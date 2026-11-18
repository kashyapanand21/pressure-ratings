import re
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA, WORK

bbb = pd.read_csv(DATA / "ball_by_ball.csv", low_memory=False)
m = pd.read_csv(DATA / "matches.csv")
c = pd.read_csv(DATA / "competitions.csv")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

# 1. Which matches are usable
ASSOC = {"United Arab Emirates", "Netherlands", "Hong Kong", "Scotland", "Canada",
         "United States Of America", "Singapore", "Kuwait", "Nepal", "Oman", "Japan"}
mm = m.merge(c, on="comp_id", how="left")
mm["kind"] = np.where(mm.comp_name.str.startswith("IPL"), "IPL",
                      np.where(mm.match_format == "T20I", "T20I", "OTHER_T20"))
n_inn = bbb[bbb.innings.isin([1, 2])].groupby("match_id").innings.nunique()
mm["usable"] = (mm.match_id.map(n_inn).eq(2) & mm.win_team_id.notna()
                & mm.revised_target.isna() & (mm.kind != "OTHER_T20"))
mm["year"] = pd.to_datetime(mm.match_date).dt.year
mm["assoc_match"] = mm.home_team.isin(ASSOC) | mm.away_team.isin(ASSOC)
keep = mm.loc[mm.usable, ["match_id", "kind", "year", "stage", "knock_out", "venue_id",
                          "win_team_id", "result", "assoc_match"]]

bb = bbb[bbb.innings.isin([1, 2])].merge(keep, on="match_id", how="inner")
bb = bb.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
print("usable matches:", keep.shape[0], "| rows:", len(bb))

# 2. Ball types
et = bb.extras_type.fillna("none")
bb["is_wide"] = et.eq("wide")
bb["is_noball"] = et.isin(["Overstep", "other"])
bb["legal"] = ~(bb.is_wide | bb.is_noball)
bb["legal_i"] = bb.legal.astype(int)
bb["wk"] = bb.is_wicket.astype(int)

# 3. Approximate team runs per ball (two versions, the data decides)
EXTRA_VAL = {"wide": 1.14, "Overstep": 1.0, "other": 1.0, "byes": 2.0,
             "legbyes": 1.3, "Penalty": 5.0}
base = et.map(EXTRA_VAL).fillna(0.0)
bb["tot_a"] = bb.runs + base
bb["tot_b"] = bb.tot_a + np.where(bb.is_noball, bb.extras_2_runs, 0)

margin = (keep.set_index("match_id").result
          .str.extract(r"by\s+(\d+)\s+runs?", flags=re.IGNORECASE)[0].astype(float).dropna())
tots = bb.groupby(["match_id", "innings"])[["tot_a", "tot_b"]].sum().unstack("innings")
errs = {}
print("\nTotals check against 'won by N runs' (" + str(len(margin)) + " matches):")
for col in ["tot_a", "tot_b"]:
    diff = (tots[(col, 1)] - tots[(col, 2)]).reindex(margin.index)
    err = (margin - diff).abs()
    errs[col] = err.mean()
    print(f"  {col}: mean abs error {err.mean():.2f} | within 1 run: {int((err <= 1).sum())}")
best = min(errs, key=errs.get)
print("using:", best)
bb["total_runs"] = bb[best]


# 4. Match state BEFORE each ball
def cum_before(col):
    return bb.groupby(["match_id", "innings"], sort=False)[col].cumsum() - bb[col]


bb["score_before"] = cum_before("total_runs")
bb["wkts_before"] = cum_before("wk")
bb["legal_before"] = cum_before("legal_i")
bb["balls_left"] = 120 - bb.legal_before

inn1_total = bb[bb.innings == 1].groupby("match_id").total_runs.sum()
bb["target"] = np.where(bb.innings == 2, bb.match_id.map(inn1_total).round() + 1, np.nan)
bb["runs_needed"] = bb.target - bb.score_before
bb["rrr"] = bb.runs_needed / bb.balls_left.replace(0, np.nan) * 6
bb["crr"] = bb.score_before / bb.legal_before.replace(0, np.nan) * 6
bb["phase"] = pd.cut(bb.overid, [0, 6, 15, 20], labels=["powerplay", "middle", "death"]).astype(str)
bb["bat_won"] = bb.win_team_id.eq(bb.batting_team_id)

# 5. Player credit columns
BOWLER_WICKETS = ["Bowled", "Caught", "LBW", "Stump", "Hit Wicket"]
bb["bowler_wicket"] = bb.is_wicket & bb.dismissal_type.isin(BOWLER_WICKETS)
bb["bowler_runs"] = (bb.runs + np.where(bb.is_wide, EXTRA_VAL["wide"], 0)
                     + np.where(bb.is_noball, 1.0, 0))

# 6. Sanity checks
print("\nSANITY CHECKS")
print("over number agrees with legal-ball count:",
      round(float((bb.legal_before // 6 + 1 == bb.overid).mean()), 4))
inn = bb.groupby(["match_id", "innings"]).agg(legal=("legal_i", "sum"), wk=("wk", "sum"))
i1 = inn.xs(1, level="innings")
print("1st innings with <120 legal balls and <10 wickets:", int(((i1.legal < 120) & (i1.wk < 10)).sum()))

tot1 = bb[bb.innings == 1].groupby("match_id").agg(total=("total_runs", "sum"), kind=("kind", "first"))
print("\n1st innings total by competition:")
print(tot1.groupby("kind").total.describe().round(1))
chase = bb[bb.innings == 2].groupby("match_id").agg(won=("bat_won", "first"), kind=("kind", "first"))
print("\nChasing team win rate:")
print(chase.groupby("kind").won.agg(["mean", "size"]).round(3))
print("\nMissing values in key columns:")
print(bb[["score_before", "wkts_before", "balls_left", "target", "rrr", "bat_won"]].isna().sum())

cols = ["match_id", "innings", "kind", "year", "stage", "knock_out", "venue_id", "assoc_match",
        "innings_ball_num", "overid", "phase", "batting_team_id", "bowling_team_id",
        "striker_id", "striker_name", "non_striker_id", "bowler_id", "bowler_name",
        "batting_order", "batting_type", "bowling_type", "pace_or_spin",
        "runs", "total_runs", "bowler_runs", "is_four", "is_six", "is_wide", "is_noball", "legal",
        "is_wicket", "dismissal_type", "dismissed_player_id", "bowler_wicket",
        "score_before", "wkts_before", "legal_before", "balls_left",
        "target", "runs_needed", "rrr", "crr", "bat_won"]
bb[cols].to_csv(WORK / "balls_clean.csv", index=False)
print("\nSaved:", WORK / "balls_clean.csv", "| rows:", len(bb))