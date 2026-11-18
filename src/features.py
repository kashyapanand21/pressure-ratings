FEATS = {
    1: {"cols": ["score_before", "wkts_before", "balls_left_c", "proj_score", "is_ipl"],
        "mono": [1, -1, 1, 1, 0]},
    2: {"cols": ["runs_needed_c", "wkts_before", "balls_left_c", "rrr_c", "is_ipl"],
        "mono": [-1, -1, 1, -1, 0]},
}


def add_features(df):
    df["is_ipl"] = (df.kind == "IPL").astype(int)
    df["balls_left_c"] = df.balls_left.clip(1, 120)
    df["runs_needed_c"] = df.runs_needed.clip(lower=0)
    df["rrr_c"] = (df.runs_needed_c / df.balls_left_c * 6).clip(upper=36)
    df["crr_c"] = (df.score_before / (120 - df.balls_left_c).clip(lower=1) * 6).clip(upper=18)
    df["proj_score"] = (df.score_before
                        + df.score_before / (120 - df.balls_left_c + 1) * df.balls_left_c)
    return df