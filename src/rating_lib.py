import pandas as pd

from config import MIN_BALLS, MIN_MATCH_BALLS, WEIGHTS


def player_match_table(df, pid, imp, pct_col, high):
    """One row per player, match and pressure flag: sum, sum of squares and count of impact."""
    d = df[~df.assoc_match.astype(bool) & df[imp].notna()]
    return (d.assign(hp=d[pct_col] >= high, sq=d[imp] ** 2)
             .groupby([pid, "match_id", "hp"])
             .agg(s=(imp, "sum"), s2=("sq", "sum"), n=(imp, "size"))
             .reset_index().rename(columns={pid: "pid"}))

# def _shrunk(t, min_n):
#     g = t.groupby("pid")[["S", "S2", "N"]].sum()
#     mean = g.S / g.N
#     noise = g.S2.sum() / g.N.sum() - (g.S.sum() / g.N.sum()) ** 2
#     big = g.N >= min_n
#     true_var = max(mean[big].var() - (noise / g.N[big]).mean(), 1e-9)
#     k = noise / true_var
#     return mean * g.N / (g.N + k) * 100, g.N, k

# def compute(pm, match_w=None, min_balls=MIN_BALLS, min_match_balls=MIN_MATCH_BALLS,
#             weights=WEIGHTS):
#     """Ratings from a player-match table. match_w: weight per match_id (missing = 0)."""
#     t = pm.copy()
#     t["W"] = 1.0 if match_w is None else t.match_id.map(match_w).fillna(0).values
#     t = t[t.W > 0]
#     t["S"], t["S2"], t["N"] = t.s * t.W, t.s2 * t.W, t.n * t.W
#     hp, nm = t[t.hp], t[~t.hp]

#     out = pd.DataFrame()
#     out["skill"], out["p_balls"], k_p = _shrunk(hp, min_balls)
#     out["normal"], out["n_balls"], k_n = _shrunk(nm, min_balls)
#     out["resilience"] = out.skill - out.normal


def _stats(t):
    g = t.groupby("pid")[["S", "S2", "N"]].sum()
    noise = g.S2.sum() / g.N.sum() - (g.S.sum() / g.N.sum()) ** 2
    return g.S / g.N, g.N, noise


def _true_var(est, noise_each, big):
    """Spread of real differences between players = spread seen - spread expected from luck."""
    big = big.reindex(est.index, fill_value=False)  # players missing from one group count as not big
    return max(est[big].var() - noise_each[big].mean(), 1e-9)

def compute(pm, match_w=None, min_balls=MIN_BALLS, min_match_balls=MIN_MATCH_BALLS,
            weights=WEIGHTS, prior="own", fixed_tv=None):
    """Ratings from a player-match table. match_w: weight per match_id (missing = 0).
    prior="own": pressure level starts from the player's own normal level (v2).
    prior="zero": pressure level starts from the league average (v1).
    fixed_tv: real-spread estimates from the full data, held fixed in re-samples and splits."""
    t = pm.copy()
    t["W"] = 1.0 if match_w is None else t.match_id.map(match_w).fillna(0).values
    t = t[t.W > 0]
    t["S"], t["S2"], t["N"] = t.s * t.W, t.s2 * t.W, t.n * t.W
    hp, nm = t[t.hp], t[~t.hp]

    mean_p, n_p, noise_p = _stats(hp)
    mean_n, n_n, noise_n = _stats(nm)
    fixed_tv = fixed_tv or {}
    tv_n = fixed_tv.get("n") or _true_var(mean_n, noise_n / n_n, n_n >= min_balls)
    normal = mean_n * tv_n / (tv_n + noise_n / n_n)          # shrunk towards average

    out = pd.DataFrame({"p_balls": n_p, "n_balls": n_n})
    if prior == "zero":
        tv_p = fixed_tv.get("p") or _true_var(mean_p, noise_p / n_p, n_p >= min_balls)
        out["skill"] = mean_p * tv_p / (tv_p + noise_p / n_p) * 100
        out["normal"] = normal * 100
        out["resilience"] = out.skill - out.normal
        k = {"k_pressure": noise_p / tv_p, "k_normal": noise_n / tv_n, "k_resilience": float("nan"),
             "tv": {"n": tv_n, "p": tv_p}}
    else:
        diff = mean_p - mean_n                               # raw pressure minus raw normal
        diff_noise = noise_p / n_p + noise_n / n_n           # luck in that difference
        tv_d = fixed_tv.get("d") or _true_var(diff, diff_noise, n_p >= min_balls)
        resil = diff * tv_d / (tv_d + diff_noise)            # shrunk towards "no change"
        out["normal"] = normal * 100
        out["resilience"] = resil * 100
        out["skill"] = (normal + resil) * 100                # best estimate of pressure level
        k = {"k_pressure": float("nan"), "k_normal": noise_n / tv_n, "k_resilience": noise_p / tv_d,
             "tv": {"n": tv_n, "d": tv_d}}
    m = hp[hp.n >= min_match_balls]
    c = m.assign(pos=(m.s > 0) * m.W).groupby("pid")[["pos", "W"]].sum()
    out["p_matches"] = c.W
    out["consistency"] = (c.pos + 5) / (c.W + 10) * 100  # pulled towards 50%

    out = out[out.p_balls >= min_balls].dropna(subset=["normal", "consistency"]).copy()
    for comp, w in weights.items():
        sd = out[comp].std()
        # a part with no real spread (e.g. resilience fully shrunk to 0) carries no information:
        # z-scoring it would blow rounding dust up into fake scores, so it counts as 0 for everyone
        out["z_" + comp] = (out[comp] - out[comp].mean()) / sd if sd > 1e-3 else 0.0
    out["rating"] = sum(w * out["z_" + comp] for comp, w in weights.items())
    out = out.sort_values("rating", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.index.name = "player_id"
    return out, k