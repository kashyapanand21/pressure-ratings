import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from config import FIGS, MODELS, N_FOLDS, SEED, TEST_YEAR, WORK
from features import FEATS, add_features

# ---------- 1. Load and build features ----------
bb = pd.read_csv(WORK / "balls_clean.csv", low_memory=False)
bb = add_features(bb)
bb["y"] = bb.bat_won.astype(int)

def make_models(mono):
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "logistic_poly": make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                       LogisticRegression(max_iter=5000)),
        "gbm": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
            l2_regularization=1.0, monotonic_cst=mono, random_state=SEED),
    }


def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p)}

results, final_models = [], {}
bb["wp_oof"] = np.nan

for inn, spec in FEATS.items():
    d = bb[bb.innings == inn]
    X, y, g = d[spec["cols"]], d.y, d.match_id
    train = d.year < TEST_YEAR

    base = y[train].mean()
    for split, mask in (("cv", train), ("test_" + str(TEST_YEAR), ~train)):
        r = scores(y[mask], np.full(mask.sum(), base))
        results.append({"innings": inn, "model": "constant", "split": split, **r})

    for name in make_models(spec["mono"]):
        oof = np.zeros(train.sum())
        gkf = GroupKFold(n_splits=N_FOLDS)
        for tr, va in gkf.split(X[train], y[train], g[train]):
            mdl = make_models(spec["mono"])[name]
            mdl.fit(X[train].iloc[tr], y[train].iloc[tr])
            oof[va] = mdl.predict_proba(X[train].iloc[va])[:, 1]
        results.append({"innings": inn, "model": name, "split": "cv", **scores(y[train], oof)})

        mdl = make_models(spec["mono"])[name]
        mdl.fit(X[train], y[train])
        p_test = mdl.predict_proba(X[~train])[:, 1]
        results.append({"innings": inn, "model": name, "split": "test_" + str(TEST_YEAR),
                        **scores(y[~train], p_test)})

res = pd.DataFrame(results).round(4)
print("MODEL COMPARISON (lower log_loss / brier is better, higher auc is better)")
print(res.to_string(index=False))
res.to_csv(WORK / "winprob_model_comparison.csv", index=False)

# ---------- 3. Final model: the CV winner, out-of-fold win chance for every ball ----------
cv = res[(res.split == "cv") & (res.model != "constant")]
best = cv.loc[cv.groupby("innings").log_loss.idxmin()].set_index("innings").model.to_dict()
print("\nChosen model per innings:", best)

for inn, spec in FEATS.items():
    d = bb[bb.innings == inn]
    X, y, g = d[spec["cols"]], d.y, d.match_id
    oof = np.zeros(len(d))
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        mdl = make_models(spec["mono"])[best[inn]]
        mdl.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = mdl.predict_proba(X.iloc[va])[:, 1]
    bb.loc[d.index, "wp_oof"] = oof

    final_models[inn] = {"cols": spec["cols"], "chosen": best[inn],
                         "models": {name: mdl.fit(X, y)
                                    for name, mdl in make_models(spec["mono"]).items()}}

joblib.dump(final_models, MODELS / "winprob.joblib")

# ---------- 4. Calibration check ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
print("\nCALIBRATION (out-of-fold, all years)")
for ax, inn in zip(axes, (1, 2)):
    d = bb[bb.innings == inn]
    bins = pd.cut(d.wp_oof, np.linspace(0, 1, 11), include_lowest=True)
    cal = d.groupby(bins, observed=True).agg(pred=("wp_oof", "mean"), actual=("y", "mean"),
                                             balls=("y", "size"))
    print(f"\nInnings {inn}:\n{cal.round(3).to_string()}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.plot(cal.pred, cal.actual, "o-")
    ax.set(title=f"Innings {inn}", xlabel="Predicted win chance", ylabel="Actual win rate",
           xlim=(0, 1), ylim=(0, 1))
fig.tight_layout()
fig.savefig(FIGS / "winprob_calibration.png", dpi=150)

out = bb[["match_id", "innings", "innings_ball_num", "wp_oof"]]
out.to_csv(WORK / "winprob_oof.csv", index=False)
print("\nSaved:", MODELS / "winprob.joblib", "|", WORK / "winprob_oof.csv",
      "|", FIGS / "winprob_calibration.png")