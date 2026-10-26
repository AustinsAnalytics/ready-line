"""Stage 7 - evaluate, ensemble and calibrate the RUL models.

* Ensemble weight chosen on out-of-fold predictions only, then applied to the test sets.
* Split-conformal prediction intervals (Mondrian bins by predicted RUL) calibrated on OOF residuals;
  coverage checked on the untouched NASA test sets.
* P(RUL <= 30 cycles) alert classifier: ROC AUC and calibration.
* Exports the per-engine prognostic "profiles" the fleet simulator replays.
"""
from __future__ import annotations

import base64
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from common import PROCESSED, SEED, SENSOR_INFO, SUBSETS, export

RUL_CAP = 125
BINS = [0, 15, 30, 50, 75, 100, 126]
MODELS = {"mean": "Training mean (baseline)", "ridge": "Ridge regression", "lgbm": "LightGBM",
          "cnn": "1D-CNN (PyTorch, GPU)", "ens": "Ensemble (LightGBM + CNN)"}
PROFILE_LEN = 100
rng = np.random.default_rng(SEED)


def nasa_score(err: np.ndarray) -> float:
    return float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))


def load():
    tr = pd.read_parquet(PROCESSED / "rul_oof_gbm.parquet").merge(
        pd.read_parquet(PROCESSED / "rul_oof_cnn.parquet"), on=["engine_id", "cycle"])
    te = pd.read_parquet(PROCESSED / "rul_test_gbm.parquet").merge(
        pd.read_parquet(PROCESSED / "rul_test_cnn.parquet"), on=["engine_id", "cycle"])
    return tr, te


def choose_weight(tr: pd.DataFrame) -> float:
    ws = np.linspace(0, 1, 21)
    rmse = [np.sqrt(np.mean((w * tr.oof_lgbm + (1 - w) * tr.oof_cnn - tr.target) ** 2)) for w in ws]
    return float(ws[int(np.argmin(rmse))])


GROUPS = {"FD001": "single", "FD003": "single", "FD002": "multi", "FD004": "multi"}


def conformal(tr: pd.DataFrame, col: str, levels=(0.8, 0.9)) -> dict:
    """Asymmetric Mondrian split-conformal. Cells = operating profile (single- vs multi-condition
    fleet) x predicted-RUL bin; within each cell, quantiles of the OOF residual (target - pred)."""
    pred = tr[col].clip(0, RUL_CAP)
    res = tr.target - pred
    b = np.digitize(pred, BINS[1:-1])
    grp = tr.subset.map(GROUPS).to_numpy()
    table = {}
    for lv in levels:
        a = 1 - lv
        table[lv] = {}
        for g in ("single", "multi"):
            rows = []
            for k in range(len(BINS) - 1):
                r = res[(b == k) & (grp == g)].to_numpy()
                n = len(r)
                # finite-sample correction for split conformal
                qlo = np.quantile(r, max(0.0, np.floor((n + 1) * a / 2) / n))
                qhi = np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - a / 2)) / n))
                rows.append((float(qlo), float(qhi)))
            table[lv][g] = rows
    return table


def apply_interval(pred: np.ndarray, subset: np.ndarray, table: dict, lv: float):
    p = np.clip(pred, 0, RUL_CAP)
    b = np.digitize(p, BINS[1:-1])
    g = pd.Series(subset).map(GROUPS).to_numpy()
    lo = np.array([table[lv][gg][k][0] for gg, k in zip(g, b)])
    hi = np.array([table[lv][gg][k][1] for gg, k in zip(g, b)])
    return np.clip(p + lo, 0, None), p + hi


def main() -> None:
    tr, te = load()
    w = choose_weight(tr)
    print(f"  ensemble weight (LightGBM share) chosen on OOF: {w:.2f}")
    tr["oof_ens"] = w * tr.oof_lgbm + (1 - w) * tr.oof_cnn
    te["pred_ens"] = w * te.pred_lgbm + (1 - w) * te.pred_cnn

    # ------------------------------------------------------------------ metrics
    last = te.groupby("engine_id").tail(1).copy()
    metrics = []
    for m, label in MODELS.items():
        e_all = last[f"pred_{m}"].clip(0, None) - last.rul
        row = {"model": m, "label": label,
               "oof_rmse": float(np.sqrt(np.mean((tr[f"oof_{m}"] - tr.target) ** 2))),
               "rmse": float(np.sqrt(np.mean(e_all ** 2))), "mae": float(np.mean(np.abs(e_all))),
               "score": nasa_score(e_all.to_numpy())}
        dz = last.rul <= 60
        row["rmse_decision_zone"] = float(np.sqrt(np.mean(e_all[dz] ** 2)))
        for s in SUBSETS:
            m_s = last.subset == s
            row[f"rmse_{s}"] = float(np.sqrt(np.mean(e_all[m_s] ** 2)))
            row[f"score_{s}"] = nasa_score(e_all[m_s].to_numpy())
        metrics.append(row)
        print(f"  {label:28s} RMSE {row['rmse']:6.2f} | decision-zone {row['rmse_decision_zone']:5.2f} | "
              + " ".join(f"{s}:{row[f'rmse_{s}']:.2f}" for s in SUBSETS))

    # ------------------------------------------------------------------ conformal
    table = conformal(tr, "oof_ens")
    cover = {}
    for lv in (0.8, 0.9):
        lo, hi = apply_interval(last.pred_ens.to_numpy(), last.subset.to_numpy(), table, lv)
        truth = last.rul.clip(upper=RUL_CAP).to_numpy()
        lo_all, hi_all = apply_interval(te.pred_ens.to_numpy(), te.subset.to_numpy(), table, lv)
        truth_all = te.rul.clip(upper=RUL_CAP).to_numpy()
        cover[str(lv)] = {"last_cycle": float(np.mean((truth >= lo) & (truth <= hi))),
                          "all_cycles": float(np.mean((truth_all >= lo_all) & (truth_all <= hi_all))),
                          "mean_width": float(np.mean(hi - lo))}
        last[f"lo{int(lv * 100)}"], last[f"hi{int(lv * 100)}"] = lo, hi
        te[f"lo{int(lv * 100)}"], te[f"hi{int(lv * 100)}"] = lo_all, hi_all
    print("  conformal coverage:", cover)
    truth = last.rul.clip(upper=RUL_CAP)
    pb = np.digitize(last.pred_ens.clip(0, RUL_CAP), BINS[1:-1])
    cover_bins = []
    for k in range(len(BINS) - 1):
        m = pb == k
        cover_bins.append({"bin": f"{BINS[k]}–{BINS[k + 1] - 1}", "n": int(m.sum()),
                           "c80": float(((truth >= last.lo80) & (truth <= last.hi80))[m].mean()),
                           "c90": float(((truth >= last.lo90) & (truth <= last.hi90))[m].mean())})

    # ------------------------------------------------------------------ alert classifier
    y_all = (te.rul <= 30).astype(int)
    auc_all = roc_auc_score(y_all, te.p30)
    auc_last = roc_auc_score((last.rul <= 30).astype(int), last.p30)
    auc_oof = roc_auc_score((tr.rul <= 30).astype(int), tr.oof_p30)
    fpr, tpr, _ = roc_curve(y_all, te.p30)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 80).astype(int))
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(te.p30, bins[1:-1])
    calib = [{"p": float(te.p30[idx == k].mean()), "obs": float(y_all[idx == k].mean()), "n": int((idx == k).sum())}
             for k in range(10) if (idx == k).sum() > 0]
    alert = {"auc_all": auc_all, "auc_last": auc_last, "auc_oof": auc_oof,
             "brier": brier_score_loss(y_all, te.p30), "roc": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
             "calibration": calib, "prevalence": float(y_all.mean())}
    print(f"  P(RUL<=30) AUC test-all {auc_all:.3f}, test-last {auc_last:.3f}, OOF {auc_oof:.3f}")

    # ------------------------------------------------------------------ explainability
    imp = pd.read_parquet(PROCESSED / "rul_importance.parquet")
    names = imp.index.to_series()
    imp["sensor"] = names.str.extract(r"^(s\d+)_", expand=False).fillna(names)
    imp["family"] = np.select(
        [imp.index.str.endswith("_dev"), imp.index.str.contains("slope"), imp.index.str.contains("_sd"),
         imp.index.str.contains(r"_m\d+$") | imp.index.str.endswith("_n")],
        ["Deviation from own baseline", "Trend (30-cycle slope)", "Volatility (15-cycle sd)", "Level (raw / rolling mean)"],
        "Usage & age")
    by_sensor = imp.groupby("sensor").mean_abs_shap.sum().sort_values(ascending=False)
    by_family = imp.groupby("family").mean_abs_shap.sum().sort_values(ascending=False)
    explain = {
        "by_sensor": [{"sensor": s, "abbr": SENSOR_INFO.get(s, (s,))[0], "name": SENSOR_INFO.get(s, (s, s))[1],
                       "shap": float(v)} for s, v in by_sensor.items()],
        "by_family": [{"family": f, "shap": float(v)} for f, v in by_family.items()],
        "top_features": [{"feature": f, "shap": float(v)} for f, v in imp.mean_abs_shap.head(15).items()],
    }

    # ------------------------------------------------------------------ scatter + trajectories
    scatter = last[["engine_id", "subset", "rul", "pred_ens", "lo80", "hi80"]].rename(
        columns={"rul": "truth", "pred_ens": "pred"}).to_dict("records")
    picks = []
    for s in SUBSETS:
        cands = last[last.subset == s].sort_values("rul")
        picks += cands.iloc[np.linspace(0, len(cands) - 1, 6).astype(int)].engine_id.tolist()
    traj = []
    for eid in picks:
        g = te[te.engine_id == eid]
        step = max(1, len(g) // 120)
        g = g.iloc[::step]
        traj.append({"id": eid, "subset": eid[:5], "cycle": g.cycle.tolist(), "truth": g.rul.tolist(),
                     "pred": g.pred_ens.round(1).tolist(), "lo": g.lo80.round(1).tolist(), "hi": g.hi80.round(1).tolist(),
                     "p30": g.p30.round(3).tolist(), "final_truth": int(g.rul.iloc[-1])})

    band = []
    for lo_b, hi_b in [(0, 15), (15, 30), (30, 60), (60, 100), (100, 125), (125, 400)]:
        m = (te.rul >= lo_b) & (te.rul < hi_b)
        e = (te.pred_ens - te.rul.clip(upper=RUL_CAP))[m]
        band.append({"band": f"{lo_b}–{hi_b if hi_b < 400 else '∞'}", "n": int(m.sum()), "rmse": float(np.sqrt(np.mean(e ** 2))),
                     "bias": float(e.mean()), "p10": float(e.quantile(0.1)), "p90": float(e.quantile(0.9))})

    # ------------------------------------------------------------------ simulator profiles
    clusters = pd.read_csv(PROCESSED / "fault_clusters.csv", index_col=0)["fault_cluster"]
    prof_rows, packed = [], {m: bytearray() for m in ("ens", "ridge", "lgbm")}
    for eid, g in tr.groupby("engine_id", sort=True):
        life = int(g.cycle.max())
        tail = g.iloc[-PROFILE_LEN:]
        pad = PROFILE_LEN - len(tail)
        row = {"engine_id": eid, "life": life, "subset": eid[:5], "cluster": int(clusters.get(eid, 0))}
        for m in packed:
            v = np.clip(np.round(tail[f"oof_{m}"].to_numpy()), 0, RUL_CAP).astype(np.uint8)
            v = np.r_[np.full(pad, RUL_CAP, np.uint8), v]
            packed[m] += v.tobytes()
            row[m] = v
        prof_rows.append(row)
    prof = pd.DataFrame(prof_rows)
    np.savez_compressed(PROCESSED / "sim_profiles.npz", life=prof.life.to_numpy(), cluster=prof.cluster.to_numpy(),
                        **{m: np.stack(prof[m].to_list()) for m in packed})
    export("profiles", {"n": len(prof), "len": PROFILE_LEN, "cap": RUL_CAP, "life": prof.life.tolist(),
                        "models": {m: base64.b64encode(bytes(b)).decode() for m, b in packed.items()}})
    last.to_parquet(PROCESSED / "test_last_cycle.parquet")
    pd.DataFrame({"pred": tr.oof_ens.clip(0, RUL_CAP), "target": tr.target, "rul": tr.rul,
                  "subset": tr.subset}).to_parquet(PROCESSED / "oof_ens.parquet")

    export("models", {"metrics": metrics, "ensemble_weight": w, "conformal": {"bins": BINS, "table": {str(k): v for k, v in table.items()},
                      "coverage": cover, "coverage_bins": cover_bins}, "alert": alert, "explain": explain, "scatter": scatter,
                      "trajectories": traj, "bands": band, "cap": RUL_CAP,
                      "n_test_engines": int(last.engine_id.nunique()), "n_train_engines": int(tr.engine_id.nunique())})


if __name__ == "__main__":
    main()
