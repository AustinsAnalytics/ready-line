"""Stage 9 - demand forecasting for engine maintenance actions.

A. Fleet-level monthly engine "remove & replace" demand: seasonal naive vs ETS vs SARIMAX with a
   COVID intervention regressor; holdout 2024-2025, then a 2026 forecast with intervals.
B. Part-level intermittent demand (many months with zero demand): Croston, SBA and TSB vs naive
   baselines across every engine part with enough history, scored by RMSSE and bias.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import WAREHOUSE, Timer, export

warnings.filterwarnings("ignore")
HOLDOUT_START = "2024-01-01"


def monthly(query: str) -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    df = con.execute(query).df()
    con.close()
    return df


def mase(y, f, insample, m=12):
    scale = np.mean(np.abs(insample[m:] - insample[:-m]))
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))) / scale)


def covid(idx: pd.DatetimeIndex) -> np.ndarray:
    return ((idx >= "2020-04-01") & (idx <= "2021-03-01")).astype(float)[:, None]


# ----------------------------------------------------------------------------- A
def fleet_level() -> dict:
    df = monthly("""
        SELECT month, count(*) AS n FROM fact_sdr
        WHERE is_engine AND action = 'Replace' GROUP BY month ORDER BY month
    """)
    s = df.set_index(pd.to_datetime(df.month)).n.asfreq("MS").fillna(0)
    tr, te = s[:"2023-12-01"], s[HOLDOUT_START:]
    h = len(te)
    fc = {}
    fc["Seasonal naive"] = np.tile(tr[-12:].values, int(np.ceil(h / 12)))[:h]
    ets = ExponentialSmoothing(tr, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()
    fc["Exponential smoothing (ETS)"] = ets.forecast(h).values
    sar = SARIMAX(tr, exog=covid(tr.index), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    sar_fc = sar.get_forecast(h, exog=covid(te.index))
    fc["SARIMAX + COVID regressor"] = sar_fc.predicted_mean.values
    fc["Ensemble (ETS + SARIMAX)"] = (fc["Exponential smoothing (ETS)"] + fc["SARIMAX + COVID regressor"]) / 2
    scores = []
    for name, f in fc.items():
        scores.append({"model": name, "mase": mase(te.values, f, tr.values),
                       "rmse": float(np.sqrt(np.mean((te.values - f) ** 2))),
                       "mape": float(np.mean(np.abs(te.values - f) / np.maximum(te.values, 1)))})
    scores.sort(key=lambda r: r["mase"])
    print("  fleet-level holdout:", [(r["model"], round(r["mase"], 2)) for r in scores])

    # Refit on everything, forecast 2026 with 80% / 95% intervals.
    full = SARIMAX(s, exog=covid(s.index), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    fut_idx = pd.date_range("2026-01-01", periods=12, freq="MS")
    f = full.get_forecast(12, exog=covid(fut_idx))
    ci80, ci95 = f.conf_int(alpha=0.2), f.conf_int(alpha=0.05)
    return {
        "months": [d.strftime("%Y-%m") for d in s.index], "y": s.values.tolist(),
        "holdout_start": HOLDOUT_START[:7], "holdout": {k: v.tolist() for k, v in fc.items()},
        "holdout_ci": {"lo": sar_fc.conf_int(alpha=0.2).iloc[:, 0].tolist(), "hi": sar_fc.conf_int(alpha=0.2).iloc[:, 1].tolist()},
        "scores": scores,
        "future": {"months": [d.strftime("%Y-%m") for d in fut_idx], "mean": f.predicted_mean.tolist(),
                   "lo80": ci80.iloc[:, 0].tolist(), "hi80": ci80.iloc[:, 1].tolist(),
                   "lo95": ci95.iloc[:, 0].tolist(), "hi95": ci95.iloc[:, 1].tolist(),
                   "annual": float(f.predicted_mean.sum())},
    }


# ----------------------------------------------------------------------------- B
def croston(y: np.ndarray, h: int, alpha=0.1, variant="croston") -> np.ndarray:
    z = p = None
    q = 1
    for v in y:
        if v > 0:
            if z is None:
                z, p = v, q
            else:
                z = z + alpha * (v - z)
                p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
    if z is None:
        return np.zeros(h)
    f = z / p
    if variant == "sba":
        f *= 1 - alpha / 2
    return np.full(h, f)


def tsb(y: np.ndarray, h: int, alpha=0.1, beta=0.05) -> np.ndarray:
    nz = y[y > 0]
    if len(nz) == 0:
        return np.zeros(h)
    z, p = nz.mean(), (y > 0).mean()
    for v in y:
        p = p + beta * ((v > 0) - p)
        if v > 0:
            z = z + alpha * (v - z)
    return np.full(h, z * p)


def adi_cv2(y: np.ndarray) -> tuple[float, float]:
    nz = y[y > 0]
    adi = len(y) / max(len(nz), 1)
    cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
    return adi, cv2


def part_level() -> dict:
    df = monthly("""
        SELECT part_name, month, count(*) AS n FROM fact_sdr
        WHERE is_engine AND action = 'Replace'
          AND part_name NOT IN ('UNKNOWN', 'NONE', '') AND part_name IS NOT NULL
        GROUP BY ALL
    """)
    df["month"] = pd.to_datetime(df.month)
    idx = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
    totals = df.groupby("part_name").n.sum()
    parts = totals[totals >= 40].index
    rows, examples = [], {}
    methods = ["Naive (last value)", "Moving average (12)", "Croston", "SBA", "TSB"]
    for part in parts:
        y = df[df.part_name == part].set_index("month").n.reindex(idx, fill_value=0).to_numpy(float)
        tr, te = y[:-12], y[-12:]
        if tr[-36:].sum() == 0:
            continue
        adi, cv2 = adi_cv2(tr)
        cls = ("Smooth" if adi < 1.32 and cv2 < 0.49 else "Erratic" if adi < 1.32 else
               "Intermittent" if cv2 < 0.49 else "Lumpy")
        fcs = {"Naive (last value)": np.full(12, tr[-1]), "Moving average (12)": np.full(12, tr[-12:].mean()),
               "Croston": croston(tr, 12), "SBA": croston(tr, 12, variant="sba"), "TSB": tsb(tr, 12)}
        # RMSSE (the M5-competition metric): squared errors scaled by in-sample naive error.
        # Unlike MAE-based MASE it does not reward forecasting zero for intermittent series.
        scale = np.mean(np.diff(tr) ** 2)
        if scale == 0:
            continue
        r = {"part": part, "total": int(y.sum()), "adi": adi, "cv2": cv2, "class": cls,
             "zero_share": float((tr == 0).mean())}
        for m in methods:
            r[m] = float(np.sqrt(np.mean((te - fcs[m]) ** 2) / scale))
            r[f"{m}|bias"] = float((fcs[m].sum() - te.sum()) / max(te.sum(), 1))
        rows.append(r)
        examples[part] = {"y": y.tolist(), "fc": {m: float(fcs[m][0]) for m in methods}}
    res = pd.DataFrame(rows)
    summary = []
    for m in methods:
        summary.append({"method": m, "mean_rmsse": res[m].mean(), "median_rmsse": res[m].median(),
                        "median_abs_bias": res[f"{m}|bias"].abs().median(),
                        "wins": int((res[methods].idxmin(axis=1) == m).sum())})
    by_class = res.groupby("class").apply(lambda g: {m: g[m].median() for m in methods} | {"n": len(g)}).to_dict()
    top_parts = res.sort_values("total", ascending=False).head(12).part.tolist()
    print(f"  {len(res)} part series; classes {res['class'].value_counts().to_dict()}")
    print("  ", [(s["method"], round(s["median_rmsse"], 3), round(s["median_abs_bias"], 3), s["wins"]) for s in summary])
    return {"n_series": len(res), "summary": summary, "by_class": by_class,
            "classes": res["class"].value_counts().to_dict(),
            "scatter": res[["part", "adi", "cv2", "class", "total"]].to_dict("records"),
            "examples": {p: examples[p] for p in top_parts}, "months": [d.strftime("%Y-%m") for d in idx]}


def main() -> None:
    with Timer("fleet-level forecast"):
        a = fleet_level()
    with Timer("intermittent demand"):
        b = part_level()
    export("forecast", {"fleet": a, "parts": b})


if __name__ == "__main__":
    main()
