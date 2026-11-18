import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGS, HIGH, MODELS, WORK
from features import add_features

pd.set_option("display.width", 200)
# HIGH = 0.75  # top 25% of balls = high pressure
# Which candidate defines pressure in each innings. None = the CV winner.
PRESSURE_MODEL = {1: "behind_par", 2: "logistic_poly"}

# ---------- 1. Load ----------
bb = pd.read_csv(WORK / "balls_clean.csv", low_memory=False)
wp = pd.read_csv(WORK / "winprob_oof.csv")
bb = bb.merge(wp, on=["match_id", "innings", "innings_ball_num"], how="left")
models = joblib.load(MODELS / "winprob.joblib")
NAMES = list(models[1]["models"])

# ---------- 2. Win chance after each possible outcome of this ball ----------
OUTCOMES = {"wicket": (0, 1), "four": (4, 0)}
STATE = ["kind", "score_before", "wkts_before", "balls_left", "runs_needed"]


def wp_after(d, inn, name, runs, wkt):
    s = d[STATE].copy()
    s["score_before"] = s.score_before + runs
    s["wkts_before"] = (s.wkts_before + wkt).clip(upper=10)
    s["balls_left"] = s.balls_left - 1
    if inn == 2:
        s["runs_needed"] = s.runs_needed - runs
    s = add_features(s)
    m = models[inn]
    p = m["models"][name].predict_proba(s[m["cols"]])[:, 1]
    if inn == 2:  # chase finished: the result is known, not predicted
        all_out_or_no_balls = (s.wkts_before >= 10) | ((s.balls_left <= 0) & (s.runs_needed > 0))
        p = np.where(all_out_or_no_balls, 0.0, p)
        p = np.where(s.runs_needed <= 0, 1.0, p)
    return p

# ---------- 3. One pressure definition per model: swing = WP(four) - WP(wicket) ----------
for name in NAMES:
    for inn in (1, 2):
        idx = bb.innings == inn
        four = wp_after(bb[idx], inn, name, *OUTCOMES["four"])
        wkt = wp_after(bb[idx], inn, name, *OUTCOMES["wicket"])
        bb.loc[idx, "swing_" + name] = four - wkt
    pct = bb.groupby(["innings", "kind"])["swing_" + name].rank(pct=True)
    bb["hp_" + name] = pct >= HIGH

# ---------- 3b. Two more first-innings candidates from the expected-total model ----------
xt = joblib.load(MODELS / "xtotal.joblib")
i1 = bb.innings == 1


def expected_total(d, runs=0, wkt=0, step=0):
    s = d[STATE].copy()
    s["score_before"] = s.score_before + runs
    s["wkts_before"] = (s.wkts_before + wkt).clip(upper=10)
    s["balls_left"] = s.balls_left - step
    s = add_features(s)
    rem = xt["model"].predict(s[xt["cols"]]).clip(min=0)
    rem = np.where((s.wkts_before >= 10) | (s.balls_left <= 0), 0, rem)  # innings over
    return s.score_before.values + rem


def win_from_total(total, d):
    X = pd.DataFrame({"final_total": total, "is_ipl": (d.kind == "IPL").astype(int).values})
    return xt["win_model"].predict_proba(X)[:, 1]


d1 = bb[i1]
bb.loc[i1, "swing_xtotal"] = (win_from_total(expected_total(d1, 4, 0, 1), d1)
                              - win_from_total(expected_total(d1, 0, 1, 1), d1))
bb.loc[i1, "swing_behind_par"] = d1.kind.map(xt["par"]).values - expected_total(d1)
for name in ("xtotal", "behind_par"):
    pct = bb[i1].groupby("kind")["swing_" + name].rank(pct=True)
    bb["hp_" + name] = False
    bb.loc[i1, "hp_" + name] = pct >= HIGH


# ---------- 4. Score each definition, innings by innings ----------
lg = bb[bb.legal].copy()
lg["boundary"] = lg.is_four | lg.is_six
lg["dot"] = lg.runs.eq(0)
CANDIDATES = {1: NAMES + ["xtotal", "behind_par"], 2: NAMES}
CHECKS = {  # situations every fan agrees on: (mask, what we want)
    1: {"collapse_flagged": ((lg.wkts_before >= 4) & (lg.overid <= 10), "high"),
        "cruising_flagged": ((lg.wkts_before <= 1) & lg.overid.between(9, 15) & (lg.crr >= 9), "low")},
    2: {"tight_finish_flagged": ((lg.overid >= 17) & lg.wp_oof.between(0.25, 0.75), "high"),
        "decided_flagged": (((lg.wp_oof < 0.05) | (lg.wp_oof > 0.95)), "low")},
}

for inn in (1, 2):
    sub = lg[lg.innings == inn]
    rows = []
    for name in CANDIDATES[inn]:
        f = sub["hp_" + name].astype(bool)
        g = (sub.assign(flag=f).groupby(["phase", "flag"])
             .agg(sr=("runs", "mean"), outs=("is_wicket", "mean"), dots=("dot", "mean"))
             .unstack("flag"))
        r = {"candidate": name,
             "pp_share": sub.loc[f, "phase"].eq("powerplay").mean() * 100,
             "death_share": sub.loc[f, "phase"].eq("death").mean() * 100,
             "sr_gap": ((g.sr[True] - g.sr[False]) * 100).mean(),
             "outs_gap_per100": ((g.outs[True] - g.outs[False]) * 100).mean(),
             "dot_gap_pct": ((g.dots[True] - g.dots[False]) * 100).mean()}
        for check, (mask, _) in CHECKS[inn].items():
            r[check] = f[mask[lg.innings == inn]].mean() * 100
        rows.append(r)
    print(f"\nINNINGS {inn}: PRESSURE DEFINITIONS COMPARED")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

print("""
  pp/death_share: % of pressure balls in powerplay / death
  sr/outs/dot gap: pressure minus normal balls INSIDE the same phase (gaps averaged over phases)
  collapse_flagged:     4+ down inside 10 overs, % flagged (want high)
  cruising_flagged:     0-1 down, overs 9-15, run rate 9+, % flagged (want low)
  tight_finish_flagged: chase overs 17-20, win chance 25-75%, % flagged (want high)
  decided_flagged:      chase win chance <5% or >95%, % flagged (want ~0)""")


# ---------- 5. Choose the definition ----------
chosen = {inn: PRESSURE_MODEL[inn] or models[inn]["chosen"] for inn in (1, 2)}
print("\nPressure model used:", chosen)
bb["swing"] = np.where(bb.innings == 1, bb["swing_" + chosen[1]], bb["swing_" + chosen[2]])
bb["pressure_pct"] = bb.groupby(["innings", "kind"]).swing.rank(pct=True)
bb["high_pressure"] = bb.pressure_pct >= HIGH
lg["high_pressure"] = bb.loc[lg.index, "high_pressure"]

# Bowling side. In a chase both sides face the same stakes, so the label is shared.
# Batting first, "behind par" is the batting side's trouble but the bowlers' comfort:
# bowlers are under pressure when the batting side is AHEAD of par, so mirror it.
mirror = chosen[1] == "behind_par"
bb["swing_bowl"] = np.where((bb.innings == 1) & mirror, -bb.swing, bb.swing)
bb["pressure_pct_bowl"] = bb.groupby(["innings", "kind"]).swing_bowl.rank(pct=True)
bb["high_pressure_bowl"] = bb.pressure_pct_bowl >= HIGH
print("Bowling-side label mirrored in innings 1:", mirror)

# ---------- 6. Proof for the chosen definition ----------
proof = lg.groupby(["innings", "phase", "high_pressure"]).agg(
    balls=("runs", "size"),
    strike_rate=("runs", lambda x: x.mean() * 100),
    outs_per_100=("is_wicket", lambda x: x.mean() * 100),
    dot_pct=("dot", lambda x: x.mean() * 100),
    boundary_pct=("boundary", lambda x: x.mean() * 100),
).round(1)
print("\nCHOSEN DEFINITION: PRESSURE vs NORMAL, SAME INNINGS AND PHASE (legal balls)")
print(proof.to_string())

hp = lg[lg.high_pressure & ~lg.assoc_match.astype(bool)]
bat_n, bowl_n = hp.groupby("striker_id").size(), hp.groupby("bowler_id").size()
print("\nPLAYERS BY HIGH-PRESSURE LEGAL BALLS (associate matches excluded)")
for n in (40, 60, 80, 100, 150):
    print(f">= {n:>3}: {(bat_n >= n).sum():>4} batters | {(bowl_n >= n).sum():>4} bowlers")

# ---------- 7. Chart and save ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
p = proof.reset_index()
p["grp"] = "Inn " + p.innings.astype(str) + "\n" + p.phase
order = [f"Inn {i}\n{ph}" for i in (1, 2) for ph in ("powerplay", "middle", "death")]
for ax, col, title in zip(axes, ["strike_rate", "outs_per_100"],
                          ["Strike rate", "Dismissals per 100 balls"]):
    t = p.pivot(index="grp", columns="high_pressure", values=col).reindex(order)
    t.columns = ["Normal", "High pressure"]
    t.plot.bar(ax=ax, color=["#9aa5b1", "#c0392b"], rot=0, legend=(col == "strike_rate"))
    ax.set(title=title + " (same innings and phase)", xlabel="")
axes[0].legend(loc="lower center", ncol=2, framealpha=1)
fig.tight_layout()
fig.savefig(FIGS / "pressure_proof.png", dpi=150)

drop = ["is_ipl", "balls_left_c", "runs_needed_c", "rrr_c", "proj_score", "crr_c"]
bb.drop(columns=[c for c in drop if c in bb.columns]).to_csv(WORK / "balls_pressure.csv", index=False)
print("\nSaved:", WORK / "balls_pressure.csv", "|", FIGS / "pressure_proof.png")

