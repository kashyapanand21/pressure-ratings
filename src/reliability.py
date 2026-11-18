import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import FIGS, HIGH, MIN_BALLS, SEED, TEST_YEAR, WEIGHTS, WORK
from rating_lib import compute, player_match_table

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
N_BOOT = 500
MIN_TARGET_BALLS = 30   # pressure balls in the test year to be scored in the future test
MIN_NAIVE_BALLS = 60    # balls needed for the simple comparison rankings

# ---------- 1. Load and set up both sides ----------
b = pd.read_csv(WORK / "balls_rated.csv", low_memory=False)
b = b[~b.assoc_match.astype(bool)]
b["legal"] = b.legal.astype(bool)
match_year = b.groupby("match_id").year.first()
SIDES = {
    "batters": dict(df=b[b.legal], pid="striker_id", name="striker_name", imp="bat_impact", pct="pct_bat"),
    "bowlers": dict(df=b, pid="bowler_id", name="bowler_name", imp="bowl_impact", pct="pct_bowl"),
}
for s in SIDES.values():
    s["names"] = s["df"].groupby(s["pid"])[s["name"]].agg(lambda x: x.mode().iat[0])
    s["pm"] = player_match_table(s["df"], s["pid"], s["imp"], s["pct"], HIGH)
    s["base"], info = compute(s["pm"])
    s["tv"] = info["tv"]  # how much players really differ, from the full data: held fixed below

def ones(ids):
    return pd.Series(1.0, index=ids)


def simple_rate(d, pid, side, min_n=MIN_NAIVE_BALLS):
    """Strike rate (batters) or economy, negated so higher = better (bowlers)."""
    if side == "batters":
        g = d.groupby(pid).agg(r=("runs", "sum"), n=("runs", "size"))
        return (g.r / g.n * 100)[g.n >= min_n]
    g = d.groupby(pid).agg(r=("bowler_runs", "sum"), n=("legal", "sum"))
    return (-g.r / g.n * 6)[g.n >= min_n]

# ---------- 2. Future test: do 2022-25 rankings predict 2026 pressure performance? ----------
print(f"FUTURE TEST: rank on matches before {TEST_YEAR}, score on {TEST_YEAR} pressure balls")
print("  spearman: rank agreement with the target (-1 to 1, higher = better prediction)")
print("  top10_target: average target value of each method's top 10 (same pool of players)")
future_rows = []
pre = ones(match_year.index[match_year < TEST_YEAR])
for side, s in SIDES.items():
    d, pid = s["df"], s["pid"]
    r_pre, _ = compute(s["pm"], match_w=pre)
    r_pre_v1, _ = compute(s["pm"], match_w=pre, prior="zero")
    d_pre = d[d.year < TEST_YEAR]
    d26 = d[(d.year == TEST_YEAR) & (d[s["pct"]] >= HIGH)]

    g = d26.groupby(pid)[s["imp"]].agg(["mean", "size"])
    targets = {"impact_2026": (g["mean"] * 100)[g["size"] >= MIN_TARGET_BALLS],
               "raw_2026": simple_rate(d26, pid, side, MIN_TARGET_BALLS)}
    predictors = {"our rating": r_pre.rating, "our skill only": r_pre.skill,
                  "our rating v1 (old)": r_pre_v1.rating, "our skill v1 (old)": r_pre_v1.skill,
                  "death-overs " + ("strike rate" if side == "batters" else "economy"):
                      simple_rate(d_pre[d_pre.phase == "death"], pid, side),
                  "overall " + ("strike rate" if side == "batters" else "economy"):
                      simple_rate(d_pre, pid, side)}
    pool = pd.concat(predictors.values(), axis=1).dropna().index  # players every method can rate
    for t_name, target in targets.items():
        for p_name, pred in predictors.items():
            both = pd.concat([pred.reindex(pool), target], axis=1, keys=["p", "t"]).dropna()
            rho = spearmanr(both.p, both.t).statistic if len(both) > 2 else np.nan
            future_rows.append({"side": side, "target": t_name, "method": p_name, "players": len(both),
                                "spearman": rho, "top10_target": both.nlargest(10, "p").t.mean()})
future = pd.DataFrame(future_rows).round(3)
print(future.to_string(index=False))
print("  impact_2026 = runs above expected per 100 pressure balls | raw_2026 = plain strike rate / "
      "minus economy on pressure balls")

# ---------- 3. Split-half: odd vs even matches ----------
print("\nSPLIT-HALF: rate players on odd-numbered and even-numbered matches separately")
ids = match_year.index
halves = (ones(ids[ids % 2 == 1]), ones(ids[ids % 2 == 0]))
for side, s in SIDES.items():
    a, _ = compute(s["pm"], match_w=halves[0], min_balls=MIN_BALLS // 2, fixed_tv=s["tv"])
    c, _ = compute(s["pm"], match_w=halves[1], min_balls=MIN_BALLS // 2, fixed_tv=s["tv"])
    common = a.index.intersection(c.index)
    msg = [f"{col} {spearmanr(a.loc[common, col], c.loc[common, col]).statistic:.2f}"
           for col in ("rating", "skill", "normal", "resilience", "consistency")]
    print(f"{side}: {len(common)} players in both halves | agreement: " + " | ".join(msg))

# ---------- 4. Bootstrap: how often is each player in the top 10? ----------
print(f"\nBOOTSTRAP: {N_BOOT} re-samples of matches (with replacement)")
rng = np.random.default_rng(SEED)
all_ids = match_year.index.values
for s in SIDES.values():
    s["boot"] = pd.Series(0.0, index=s["base"].index)
for i in range(N_BOOT):
    w = pd.Series(rng.choice(all_ids, len(all_ids))).value_counts()
    for s in SIDES.values():
        r, _ = compute(s["pm"], match_w=w, fixed_tv=s["tv"])
        top = r.index[:10]
        s["boot"] = s["boot"].add(pd.Series(1.0, index=top), fill_value=0)
for s in SIDES.values():
    s["boot"] = s["boot"] / N_BOOT * 100


# ---------- 5. Sensitivity: does the top 10 survive other reasonable choices? ----------
VARIANTS = {
    "base": dict(),
    "threshold 70%": dict(high=0.70),
    "threshold 80%": dict(high=0.80),
    "skill only": dict(weights={"skill": 1, "resilience": 0, "consistency": 0}),
    "equal weights": dict(weights={"skill": 1 / 3, "resilience": 1 / 3, "consistency": 1 / 3}),
    "skill 0.7": dict(weights={"skill": 0.7, "resilience": 0.15, "consistency": 0.15}),
    "no resilience": dict(weights={"skill": 2 / 3, "resilience": 0, "consistency": 1 / 3}),
    "min balls 80": dict(min_balls=80),
    "min balls 150": dict(min_balls=150),
    "old method (v1)": dict(prior="zero"),
}
print("\nSENSITIVITY: overlap of each variant's top 10 with the base top 10")
for side, s in SIDES.items():
    s["variant_hits"] = pd.Series(0, index=s["base"].index)
    base_top = set(s["base"].index[:10])
    line = []
    for v_name, v in VARIANTS.items():
        pm = s["pm"] if "high" not in v else player_match_table(
            s["df"], s["pid"], s["imp"], s["pct"], v["high"])
        same_model = "high" not in v and "prior" not in v  # new threshold / old method: re-estimate
        r, _ = compute(pm, weights=v.get("weights", WEIGHTS), min_balls=v.get("min_balls", MIN_BALLS),
                       prior=v.get("prior", "own"), fixed_tv=s["tv"] if same_model else None)
        top = r.index[:10]
        s["variant_hits"] = s["variant_hits"].add(pd.Series(1, index=top), fill_value=0)
        line.append(f"{v_name}: {len(base_top & set(top))}/10")
    print(f"{side}: " + " | ".join(line))


# ---------- 6. Decision table ----------
for side, s in SIDES.items():
    t = s["base"].head(20).copy()
    t.insert(1, "name", s["names"].reindex(t.index))
    t["boot_top10_pct"] = s["boot"].reindex(t.index).fillna(0)
    t["variants_top10"] = s["variant_hits"].reindex(t.index).fillna(0).astype(int)
    cols = ["rank", "name", "p_balls", "p_matches", "skill", "resilience", "consistency",
            "rating", "boot_top10_pct", "variants_top10"]
    print(f"\nDECISION TABLE - {side.upper()} (variants_top10 out of {len(VARIANTS)})")
    print(t[cols].round(2).to_string())
    t[cols].to_csv(WORK / f"decision_{side}.csv")

# ---------- 7. Charts ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (side, s) in zip(axes, SIDES.items()):
    top = s["boot"].sort_values(ascending=False).head(15)[::-1]
    ax.barh(s["names"].reindex(top.index), top.values, color="#c0392b")
    ax.axvline(50, color="grey", ls="--", lw=1)
    ax.set(title=f"{side.capitalize()}: % of {N_BOOT} re-samples in the top 10", xlim=(0, 100))
fig.tight_layout()
fig.savefig(FIGS / "bootstrap_top10.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, side in zip(axes, SIDES):
    f = future[(future.side == side) & (future.target == "impact_2026")]
    f = f[::-1]
    ax.barh(f.method, f.spearman,
            color=["#c0392b" if m.startswith("our") else "#9aa5b1" for m in f.method])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title=f"{side.capitalize()}: how well {TEST_YEAR - 4}-{TEST_YEAR - 1} predicts "
                 f"{TEST_YEAR} pressure impact", xlabel="Spearman rank agreement")
fig.tight_layout()
fig.savefig(FIGS / "future_test.png", dpi=150)
future.to_csv(WORK / "future_test.csv", index=False)
print("\nSaved:", WORK / "decision_batters.csv", "|", WORK / "decision_bowlers.csv", "|",
      FIGS / "bootstrap_top10.png", "|", FIGS / "future_test.png")