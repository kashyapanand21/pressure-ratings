import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import MODELS, N_FOLDS, SEED, TEST_YEAR, WORK
from features import add_features

pd.set_option("display.width", 200)

# ---------- 1. First-innings balls with the target: runs still to come ----------
bb = pd.read_csv(WORK / "balls_clean.csv", low_memory=False)
i1 = add_features(bb[bb.innings == 1].copy())
i1["final_total"] = i1.match_id.map(i1.groupby("match_id").total_runs.sum())
i1["remaining"] = i1.final_total - i1.score_before

XT_COLS = ["wkts_before", "balls_left_c", "crr_c", "is_ipl"]
XT_MONO = [-1, 1, 1, 0]


def make_models():
    return {
        "ridge_poly": make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                    Ridge(alpha=1.0)),
        "gbm": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
            l2_regularization=1.0, monotonic_cst=XT_MONO, random_state=SEED),
    }

# ---------- 2. Compare models: CV on earlier years, then test on the future ----------
X, y, g = i1[XT_COLS], i1.remaining, i1.match_id
train = i1.year < TEST_YEAR
rows = []

# Baseline: average runs still to come for this many balls left (ignores wickets and form)
by_ball = y[train].groupby(i1.balls_left_c[train]).mean()
base_pred = i1.balls_left_c.map(by_ball).fillna(y[train].mean())
for split, mask in (("cv", train), ("test_" + str(TEST_YEAR), ~train)):
    rows.append({"model": "balls_left_only", "split": split,
                 "mae_runs": mean_absolute_error(y[mask], base_pred[mask])})

for name in make_models():
    oof = cross_val_predict(make_models()[name], X[train], y[train], groups=g[train],
                            cv=GroupKFold(n_splits=N_FOLDS))
    rows.append({"model": name, "split": "cv", "mae_runs": mean_absolute_error(y[train], oof)})
    mdl = make_models()[name].fit(X[train], y[train])
    rows.append({"model": name, "split": "test_" + str(TEST_YEAR),
                 "mae_runs": mean_absolute_error(y[~train], mdl.predict(X[~train]))})

res = pd.DataFrame(rows).round(2)
print("EXPECTED REMAINING RUNS, FIRST INNINGS (mean absolute error in runs, lower is better)")
print(res.to_string(index=False))
cv = res[(res.split == "cv") & (res.model != "balls_left_only")]
best = cv.loc[cv.mae_runs.idxmin(), "model"]
print("Chosen:", best)
final_model = make_models()[best].fit(X, y)

# ---------- 3. Final total -> chance of winning the match ----------
mt = i1.groupby("match_id").agg(final_total=("final_total", "first"), is_ipl=("is_ipl", "first"),
                                y=("bat_won", "first")).astype(float)
win_model = LogisticRegression(max_iter=1000)
p = cross_val_predict(win_model, mt[["final_total", "is_ipl"]], mt.y, cv=N_FOLDS,
                      method="predict_proba")[:, 1]
print(f"\nTOTAL -> WIN MODEL ({len(mt)} matches): AUC {roc_auc_score(mt.y, p):.3f}"
      f" | log loss {log_loss(mt.y, p):.3f}")
win_model.fit(mt[["final_total", "is_ipl"]], mt.y)
grid = pd.DataFrame([(t, k) for k in (1, 0) for t in (140, 160, 180, 200, 220)],
                    columns=["final_total", "is_ipl"])
grid["win_chance"] = win_model.predict_proba(grid)[:, 1].round(2)
grid["comp"] = np.where(grid.is_ipl == 1, "IPL", "T20I")
print(grid.pivot(index="final_total", columns="comp", values="win_chance"))

par = i1.groupby("match_id").agg(t=("final_total", "first"), k=("kind", "first")).groupby("k").t.median()
print("\nPar (median first-innings total):", par.round(1).to_dict())

joblib.dump({"model": final_model, "name": best, "cols": XT_COLS, "win_model": win_model,
             "par": par.to_dict()}, MODELS / "xtotal.joblib")
print("Saved:", MODELS / "xtotal.joblib")