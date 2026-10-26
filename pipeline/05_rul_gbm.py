"""Stage 5 - remaining-useful-life (RUL) models on tabular features.

Leakage rules: every feature uses only the engine's own past (trailing windows, a baseline from its
first cycles); cross-validation folds are grouped by engine, so no engine is ever in train and
validation at once. The official NASA test sets are touched once, at the end.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import PROCESSED, SEED, WAREHOUSE, Timer

INFORMATIVE = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
RUL_CAP = 125
FOLDS = 5


def build_features() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    cols = ["engine_id", "subset", "split", "unit", "cycle", "rul", "regime"]
    for s in INFORMATIVE:
        cols += [f"{s}_n", f"{s}_m5", f"{s}_m15", f"{s}_m30", f"{s}_sd15", f"{s}_slope30"]
    df = con.execute(f"SELECT {', '.join(cols)} FROM fact_engine_cycle ORDER BY engine_id, cycle").df()
    con.close()

    # Self-referenced deviation: smoothed level minus the engine's own early-life baseline
    # (mean of its first min(cycle, 20) normalized readings) - causal by construction.
    g = df.groupby("engine_id", sort=False)
    for s in INFORMATIVE:
        x = df[f"{s}_n"]
        csum = g[f"{s}_n"].cumsum()
        n = g.cumcount() + 1
        first20 = x.where(n <= 20, 0.0).groupby(df.engine_id).cumsum()
        baseline = first20 / np.minimum(n, 20)
        df[f"{s}_dev"] = df[f"{s}_m15"] - baseline
        del csum
    # Cycles flown in the hottest/highest regimes so far (usage severity).
    df["frac_high_regime"] = (df.regime >= 4).astype(float).groupby(df.engine_id).cumsum() / (g.cumcount() + 1)
    df["target"] = df.rul.clip(upper=RUL_CAP)
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"engine_id", "subset", "split", "unit", "rul", "target", "regime"}
    return [c for c in df.columns if c not in drop]


def nasa_score(err: np.ndarray) -> float:
    """PHM08 scoring function: late predictions (err > 0) are penalized more than early ones."""
    return float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))


def main() -> None:
    with Timer("features"):
        df = build_features()
    feats = feature_cols(df)
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    print(f"  {len(feats)} features; train rows {len(tr):,}; test rows {len(te):,}")

    gkf = GroupKFold(n_splits=FOLDS)
    folds = list(gkf.split(tr, groups=tr.engine_id))
    oof = {m: np.zeros(len(tr)) for m in ("mean", "ridge", "lgbm")}
    test_pred = {m: np.zeros(len(te)) for m in oof}
    best_iters = []

    params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=80,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  verbose=-1, seed=SEED, num_threads=os.cpu_count())

    with Timer("cross-validation"):
        for k, (i_tr, i_va) in enumerate(folds):
            a, b = tr.iloc[i_tr], tr.iloc[i_va]
            oof["mean"][i_va] = a.target.mean()
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(a[feats], a.target)
            oof["ridge"][i_va] = ridge.predict(b[feats])
            dtr = lgb.Dataset(a[feats], a.target)
            dva = lgb.Dataset(b[feats], b.target)
            booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            best_iters.append(booster.best_iteration)
            oof["lgbm"][i_va] = booster.predict(b[feats], num_iteration=booster.best_iteration)
            print(f"  fold {k + 1}: lgbm OOF RMSE {np.sqrt(np.mean((oof['lgbm'][i_va] - b.target) ** 2)):.2f}"
                  f" @ {booster.best_iteration} trees")

    with Timer("final fit"):
        n_iter = int(np.mean(best_iters) * 1.1)
        final = lgb.train(params, lgb.Dataset(tr[feats], tr.target), num_boost_round=n_iter)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[feats], tr.target)
        test_pred["mean"][:] = tr.target.mean()
        test_pred["ridge"] = ridge.predict(te[feats])
        test_pred["lgbm"] = final.predict(te[feats])
        final.save_model(str(PROCESSED / "lgbm_rul.txt"))

        # P(RUL <= 30): an alerting classifier for the maintenance decision.
        clf_params = {**params, "objective": "binary"}
        clf = lgb.train(clf_params, lgb.Dataset(tr[feats], (tr.rul <= 30).astype(int)), num_boost_round=n_iter)
        oof_clf = np.zeros(len(tr))
        for i_tr, i_va in folds:
            a, b = tr.iloc[i_tr], tr.iloc[i_va]
            m = lgb.train(clf_params, lgb.Dataset(a[feats], (a.rul <= 30).astype(int)), num_boost_round=n_iter)
            oof_clf[i_va] = m.predict(b[feats])
        test_clf = clf.predict(te[feats])

        # TreeSHAP contributions on a sample of test rows (built into LightGBM).
        sample = te.sample(4000, random_state=SEED)
        contrib = final.predict(sample[feats], pred_contrib=True)[:, :-1]
        shap = pd.DataFrame(np.abs(contrib), columns=feats).mean().sort_values(ascending=False)
        gain = pd.Series(final.feature_importance("gain"), index=feats).sort_values(ascending=False)

    out_tr = tr[["engine_id", "subset", "unit", "cycle", "rul", "target"]].copy()
    for m in oof:
        out_tr[f"oof_{m}"] = oof[m]
    out_tr["oof_p30"] = oof_clf
    out_tr["fold"] = -1
    for k, (_, i_va) in enumerate(folds):
        out_tr.loc[i_va, "fold"] = k
    out_te = te[["engine_id", "subset", "unit", "cycle", "rul", "target"]].copy()
    for m in test_pred:
        out_te[f"pred_{m}"] = test_pred[m]
    out_te["p30"] = test_clf
    out_tr.to_parquet(PROCESSED / "rul_oof_gbm.parquet")
    out_te.to_parquet(PROCESSED / "rul_test_gbm.parquet")
    shap.rename("mean_abs_shap").to_frame().join(gain.rename("gain")).to_parquet(PROCESSED / "rul_importance.parquet")

    last = out_te.groupby("engine_id").tail(1)
    for m in test_pred:
        e = last[f"pred_{m}"].clip(0, None) - last.rul
        print(f"  TEST {m:6s} RMSE {np.sqrt(np.mean(e ** 2)):6.2f}  score {nasa_score(e.to_numpy()):10.0f}  "
              + "  ".join(f"{s}:{np.sqrt(np.mean(e[last.subset == s] ** 2)):.2f}" for s in sorted(last.subset.unique())))
    print("  top SHAP:", shap.head(8).round(2).to_dict())


if __name__ == "__main__":
    main()
