import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from config import HIGH, MODELS, N_FOLDS, SEED, WORK
from features import add_features
from rating_lib import compute, player_match_table

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

MIN_BALLS = 100        # high-pressure balls needed to be ranked
MIN_MATCH_BALLS = 3    # pressure balls in a match for it to count towards consistency
WEIGHTS = {"skill": 0.5, "resilience": 0.25, "consistency": 0.25}

# ---------- 1. Load, add context features ----------
bb = pd.read_csv(WORK / "balls_pressure.csv", low_memory=False)
bb = add_features(bb)
bb["is_spin"] = bb.pace_or_spin.eq("Spin").astype(int)
bb["rrr_f"] = bb.rrr_c.fillna(0)          # no required rate in the first innings
bb["out"] = bb.bowler_wicket.astype(int)  # dismissals a batter/bowler is responsible for
CTX = ["innings", "overid", "wkts_before", "balls_left_c", "crr_c", "rrr_f",
       "is_ipl", "is_spin", "pressure_pct"]

# ---------- 2. Expected-outcome models: what an average player does in this exact situation ----------
reg = lambda: HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.05,
                                            max_leaf_nodes=15, min_samples_leaf=300,
                                            random_state=SEED)
clf = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=300, random_state=SEED)


def oof_predict(make, X, y, groups, proba=False):
    out = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        m = make().fit(X.iloc[tr], y.iloc[tr])
        out[va] = m.predict_proba(X.iloc[va])[:, 1] if proba else m.predict(X.iloc[va])
    return out


legal = bb.legal.astype(bool)
bb["x_out"] = oof_predict(clf, bb[CTX], bb.out, bb.match_id, proba=True)
bb["x_bowl_runs"] = oof_predict(reg, bb[CTX], bb.bowler_runs, bb.match_id)
bb.loc[legal, "x_bat_runs"] = oof_predict(reg, bb.loc[legal, CTX], bb.loc[legal, "runs"],
                                          bb.loc[legal, "match_id"])

chk = bb[legal].groupby(["innings", "high_pressure"]).agg(
    runs_per100=("runs", lambda x: x.mean() * 100), x_runs_per100=("x_bat_runs", lambda x: x.mean() * 100),
    outs_per100=("out", lambda x: x.mean() * 100), x_outs_per100=("x_out", lambda x: x.mean() * 100))
print("EXPECTED vs ACTUAL (legal balls) - each pair should be close")
print(chk.round(2).to_string())

# ---------- 3. What a wicket costs, in runs, at this stage ----------
xt = joblib.load(MODELS / "xtotal.joblib")


def runs_to_come(d, extra_wkt):
    s = d[["kind", "score_before", "wkts_before", "balls_left", "runs_needed"]].copy()
    s["wkts_before"] = (s.wkts_before + extra_wkt).clip(upper=10)
    s = add_features(s)
    r = xt["model"].predict(s[xt["cols"]]).clip(min=0)
    return np.where(s.wkts_before >= 10, 0, r)


bb["wkt_cost"] = (runs_to_come(bb, 0) - runs_to_come(bb, 1)).clip(min=0)
print("\nAVERAGE RUN COST OF A WICKET")
print(bb.groupby(["innings", "phase"]).wkt_cost.mean().round(1).unstack())

# ---------- 4. Impact per ball: runs above expected, wickets valued in runs ----------
bb["bat_impact"] = ((bb.runs - bb.x_bat_runs) - bb.wkt_cost * (bb.out - bb.x_out)).where(legal)
bb["bowl_impact"] = (bb.x_bowl_runs - bb.bowler_runs) + bb.wkt_cost * (bb.out - bb.x_out)

# ---------- 4b. Self-made pressure check ----------
# Does a player's OWN earlier performance in the match decide whether his later balls
# count as pressure? If so, pressure balls are partly selected by his form that day.
bb["bat_prior"] = (bb.bat_impact.fillna(0).groupby([bb.match_id, bb.innings, bb.striker_id]).cumsum()
                   - bb.bat_impact.fillna(0))
bb["bowl_prior"] = (bb.bowl_impact.groupby([bb.match_id, bb.bowler_id]).cumsum() - bb.bowl_impact)
def self_made_check(bat_flag, bowl_flag, title):
    print(f"\nSELF-MADE PRESSURE CHECK ({title}): player's own impact earlier in this match (runs)")
    print("  (similar values for pressure and normal balls = labels are not driven by his own form)")
    for side, prior, flag, d in (("batters", "bat_prior", bat_flag, bb[legal]),
                                 ("bowlers", "bowl_prior", bowl_flag, bb)):
        t = d.groupby(["innings", flag])[prior].mean().unstack(flag).round(2)
        t.columns = ["normal", "pressure"]
        print(f"{side}:\n{t.to_string()}")


self_made_check("high_pressure", "high_pressure_bowl", "original labels")

# ---------- 4c. Take the player's own contribution out of first-innings labels ----------
# Batting first, pressure = distance from par, and a player's own earlier balls move it.
# Re-label each ball as if that player had been exactly average so far in the match.
# (Chases are left alone: the check shows their labels are not self-made.)
i1 = bb.innings == 1
for side, swing_col, prior, pct_col in (("bat", "swing", "bat_prior", "pressure_pct"),
                                        ("bowl", "swing_bowl", "bowl_prior", "pressure_pct_bowl")):
    adj = bb[swing_col] + bb[prior]
    bb["pct_" + side] = bb[pct_col]
    bb.loc[i1, "pct_" + side] = adj[i1].groupby(bb.kind[i1]).rank(pct=True)
    bb["hp_" + side] = bb["pct_" + side] >= HIGH

self_made_check("hp_bat", "hp_bowl", "own contribution removed")

# ---------- 5. Ratings: shrinkage, three components, combined score (see rating_lib.py) ----------
def rate(df, pid, name_col, imp, pct_col, label):
    pm = player_match_table(df, pid, imp, pct_col, HIGH)
    out, k = compute(pm)
    hp = df[~df.assoc_match.astype(bool) & (df[pct_col] >= HIGH)]
    out["ipl_share"] = hp.groupby(pid).kind.apply(lambda x: x.eq("IPL").mean() * 100)
    out["name"] = df.groupby(pid)[name_col].agg(lambda x: x.mode().iat[0])

    print(f"\n{label}: {len(out)} eligible | balls needed before half the score is the player's own: "
          f"normal level {k['k_normal']:.0f}, pressure change {k['k_resilience']:.0f}")
    cols = ["rank", "name", "p_balls", "p_matches", "skill", "normal", "resilience",
            "consistency", "ipl_share", "rating"]
    print(out[cols].head(15).round(2).to_string())
    out.to_csv(WORK / f"ratings_{label.lower()}.csv")
    return out


bat = rate(bb[legal], "striker_id", "striker_name", "bat_impact", "pct_bat", "BATTERS")
bowl = rate(bb, "bowler_id", "bowler_name", "bowl_impact", "pct_bowl", "BOWLERS")
print("\nskill/normal = runs above expected per 100 balls (bowlers: runs saved)")
print("resilience = skill - normal | consistency = % of pressure matches above expected")

keep = ["match_id", "year", "innings", "kind", "phase", "overid", "assoc_match", "legal",
        "striker_id", "striker_name", "bowler_id", "bowler_name", "runs", "bowler_runs",
        "bat_impact", "bowl_impact", "pct_bat", "pct_bowl"]
bb[keep].to_csv(WORK / "balls_rated.csv", index=False)
print("\nSaved:", WORK / "ratings_batters.csv", "|", WORK / "ratings_bowlers.csv",
      "|", WORK / "balls_rated.csv")