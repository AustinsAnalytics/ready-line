"""Stage 3 - exploratory analysis of the engine fleet.

1. Operating regimes and the "signal in the noise" problem (why raw multi-regime sensors look random).
2. Sensor suitability ranking (monotonicity, trendability, prognosability).
3. A data-driven health index for every training engine.
4. Unsupervised discovery of latent fault modes, with a parametric-bootstrap likelihood-ratio
   test for "two fault modes vs one" - checked against the dataset's documentation.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import diptest
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import SEED, SENSOR_INFO, SENSORS, SUBSETS, WAREHOUSE, Timer, export

INFORMATIVE = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
rng = np.random.default_rng(SEED)


def load() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    cols = ", ".join(["engine_id", "subset", "split", "unit", "cycle", "rul", "max_cycle", "regime",
                      "setting_1", "setting_2", "setting_3", *SENSORS, *[f"{s}_n" for s in SENSORS],
                      *[f"{s}_m15" for s in INFORMATIVE]])
    df = con.execute(f"SELECT {cols} FROM fact_engine_cycle ORDER BY engine_id, cycle").df()
    con.close()
    return df


# ----------------------------------------------------------------------------- 1
def regimes_and_noise(df: pd.DataFrame) -> dict:
    tr = df[(df.split == "train") & (df.subset == "FD002")]
    sample = tr.sample(1800, random_state=SEED)
    scatter = sample[["setting_1", "setting_2", "setting_3", "regime"]].to_dict("records")

    reg = (df[df.split == "train"].groupby("regime")
           .agg(alt=("setting_1", "mean"), mach=("setting_2", "mean"), tra=("setting_3", "mean"), rows=("cycle", "size"))
           .reset_index())

    # One FD002 engine, raw vs regime-normalized Ps30 (s11) - the clearest HPC degradation channel.
    eng_id = tr.groupby("engine_id").max_cycle.first().sort_values().index[len(tr.engine_id.unique()) // 2]
    e = tr[tr.engine_id == eng_id]
    noise = {
        "engine_id": eng_id,
        "sensor": "s11",
        "cycle": e.cycle.tolist(),
        "regime": e.regime.tolist(),
        "raw": e.s11.tolist(),
        "norm": e.s11_n.tolist(),
        "smooth": e.s11_m15.tolist(),
    }
    # How much variance in raw s11 is explained by regime alone (one-way ANOVA R^2).
    grand = tr.s11.mean()
    ss_between = tr.groupby("regime").s11.apply(lambda x: len(x) * (x.mean() - grand) ** 2).sum()
    ss_total = ((tr.s11 - grand) ** 2).sum()
    noise["regime_r2"] = ss_between / ss_total
    # Correlation of each version with cycles-remaining, for this engine.
    noise["corr_raw"] = stats.spearmanr(e.s11, -e.rul).statistic
    noise["corr_norm"] = stats.spearmanr(e.s11_n, -e.rul).statistic
    noise["corr_smooth"] = stats.spearmanr(e.s11_m15, -e.rul).statistic
    return {"scatter": scatter, "regimes": reg.to_dict("records"), "noise": noise}


# ----------------------------------------------------------------------------- 2
def sensor_ranking(df: pd.DataFrame) -> list[dict]:
    tr = df[df.split == "train"]
    rows = []
    for s in SENSORS:
        col = f"{s}_n"
        mono, trend, start, end = [], [], [], []
        for _, g in tr.groupby("engine_id"):
            x = g[col].rolling(15, min_periods=1).mean().to_numpy()
            d = np.diff(x)
            if len(d) == 0 or np.allclose(x, x[0]):
                mono.append(0.0)
                trend.append(0.0)
            else:
                mono.append(abs((d > 0).sum() - (d < 0).sum()) / len(d))
                trend.append(abs(stats.spearmanr(g.cycle, x).statistic))
            start.append(x[:10].mean())
            end.append(x[-10:].mean())
        start, end = np.array(start), np.array(end)
        spread = np.mean(np.abs(end - start))
        prog = float(np.exp(-np.std(end) / spread)) if spread > 1e-9 else 0.0
        mono_v, trend_v = float(np.nanmean(mono)), float(np.nanmean(trend))
        abbr, name, unit = SENSOR_INFO[s]
        rows.append({
            "sensor": s, "abbr": abbr, "name": name, "unit": unit,
            "monotonicity": mono_v, "trendability": trend_v, "prognosability": prog,
            "score": (mono_v + trend_v + prog) / 3, "informative": s in INFORMATIVE,
        })
    return sorted(rows, key=lambda r: -r["score"])


# ----------------------------------------------------------------------------- 3
def health_index(df: pd.DataFrame) -> dict:
    """Linear health index: map sensors to 1 (first 30 cycles) / 0 (last 30 cycles), FD001."""
    tr = df[(df.split == "train") & (df.subset == "FD001")].copy()
    feats = [f"{s}_n" for s in INFORMATIVE]
    healthy = tr[tr.cycle <= 30]
    failed = tr[tr.rul < 30]
    X = pd.concat([healthy[feats], failed[feats]])
    y = np.r_[np.ones(len(healthy)), np.zeros(len(failed))]
    lr = LinearRegression().fit(X, y)
    tr["hi"] = lr.predict(tr[feats])
    tr["hi_s"] = tr.groupby("engine_id").hi.transform(lambda x: x.rolling(10, min_periods=1).mean())

    curves = []
    for eid, g in tr.groupby("engine_id"):
        g = g.iloc[::3]
        curves.append({"id": eid, "life": int(g.max_cycle.iloc[0]),
                       "t": (-g.rul).tolist(), "hi": g.hi_s.round(3).tolist()})
    weights = sorted(
        [{"sensor": s, "abbr": SENSOR_INFO[s][0], "w": float(w)} for s, w in zip(INFORMATIVE, lr.coef_)],
        key=lambda r: -abs(r["w"]))
    end_hi = tr[tr.rul == 0].hi_s
    return {"curves": curves, "weights": weights, "threshold": float(end_hi.mean()),
            "threshold_sd": float(end_hi.std())}


# ----------------------------------------------------------------------------- 4
def signatures(df: pd.DataFrame, subset: str) -> tuple[pd.DataFrame, np.ndarray]:
    tr = df[(df.split == "train") & (df.subset == subset)]
    feats = [f"{s}_n" for s in INFORMATIVE]
    rows, ids, lives = [], [], []
    for eid, g in tr.groupby("engine_id"):
        rows.append(g[feats].iloc[-20:].mean().to_numpy() - g[feats].iloc[:30].mean().to_numpy())
        ids.append(eid)
        lives.append(int(g.max_cycle.iloc[0]))
    sig = pd.DataFrame(rows, columns=INFORMATIVE)
    sig["engine_id"], sig["life"] = ids, lives
    return sig, StandardScaler().fit_transform(sig[INFORMATIVE])


def fit_gmms(Z: np.ndarray, kmax: int = 4) -> list[GaussianMixture]:
    return [GaussianMixture(k, covariance_type="full", n_init=8, random_state=SEED).fit(Z) for k in range(1, kmax + 1)]


def bootstrap_lrt(Z: np.ndarray, B: int = 199) -> tuple[float, float, float]:
    """Parametric bootstrap LRT for k=1 vs k=2 Gaussian mixtures (McLachlan 1987).
    Chi-square asymptotics fail here (boundary of parameter space), so simulate the null."""
    def lr_stat(X):
        g1 = GaussianMixture(1, covariance_type="full", random_state=SEED).fit(X)
        g2 = GaussianMixture(2, covariance_type="full", n_init=4, random_state=SEED).fit(X)
        return 2 * len(X) * (g2.score(X) - g1.score(X)), g1

    observed, g1 = lr_stat(Z)
    null = []
    for b in range(B):
        Xb = rng.multivariate_normal(g1.means_[0], g1.covariances_[0], size=len(Z))
        null.append(lr_stat(Xb)[0])
    p = (1 + np.sum(np.array(null) >= observed)) / (B + 1)
    return float(observed), float(p), float(np.quantile(null, 0.95))


def fault_modes(df: pd.DataFrame) -> dict:
    out = {"subsets": []}
    for subset in SUBSETS:
        sig, Zfull = signatures(df, subset)
        pca = PCA(n_components=3, random_state=SEED).fit(Zfull)
        Z = pca.transform(Zfull)
        gmms = fit_gmms(Z)
        bics = [g.bic(Z) for g in gmms]
        k_best = int(np.argmin(bics) + 1)
        lr, p, lr_null95 = bootstrap_lrt(Z)
        # Hartigan's dip test on the first principal component: H0 = unimodal signatures.
        dip, dip_p = diptest.diptest(Z[:, 0])
        g2 = gmms[1]
        labels = g2.predict(Z)
        # Name clusters so cluster 0 is the larger one.
        if (labels == 0).sum() < (labels == 1).sum():
            labels = 1 - labels
        sig["cluster"] = labels
        entry = {
            "subset": subset, "n": len(sig), "bic": bics, "k_best": k_best,
            "lrt_stat": lr, "lrt_p": p, "lrt_null95": lr_null95,
            "delta_bic_12": bics[1] - bics[0], "dip": dip, "dip_p": dip_p, "explained": pca.explained_variance_ratio_.tolist(),
            "points": [{"id": e, "x": float(z[0]), "y": float(z[1]), "c": int(c), "life": int(l)}
                       for e, z, c, l in zip(sig.engine_id, Z, labels, sig.life)],
        }
        if k_best >= 2:
            a, b = sig[sig.cluster == 0], sig[sig.cluster == 1]
            tests = []
            for s in INFORMATIVE:
                t = stats.ttest_ind(a[s], b[s], equal_var=False)
                tests.append({"sensor": s, "abbr": SENSOR_INFO[s][0], "mean_0": a[s].mean(), "mean_1": b[s].mean(),
                              "t": t.statistic, "p": t.pvalue})
            # Benjamini-Hochberg false-discovery-rate control across the 14 sensors.
            ps = np.array([t["p"] for t in tests])
            order = np.argsort(ps)
            q = np.empty_like(ps)
            q[order] = np.minimum.accumulate((ps[order] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1]
            for t, qq in zip(tests, q):
                t["q"] = float(min(qq, 1.0))
            entry["sensor_tests"] = sorted(tests, key=lambda r: r["p"])
            mw = stats.mannwhitneyu(a.life, b.life)
            entry["life_by_cluster"] = {"n0": len(a), "n1": len(b), "mean0": a.life.mean(), "mean1": b.life.mean(),
                                        "median0": a.life.median(), "median1": b.life.median(), "mw_p": mw.pvalue}
            entry["labels"] = dict(zip(sig.engine_id, labels.tolist()))
        out["subsets"].append(entry)
        print(f"  {subset}: dBIC(1->2) = {bics[1] - bics[0]:+.1f}; dip p = {dip_p:.4f}; "
              f"LR = {lr:.1f} (null 95% = {lr_null95:.1f})")
    return out


def main() -> None:
    with Timer("load"):
        df = load()
    with Timer("regimes"):
        reg = regimes_and_noise(df)
    print(f"  regime explains {reg['noise']['regime_r2']:.1%} of raw s11 variance")
    with Timer("sensor ranking"):
        rank = sensor_ranking(df)
    with Timer("health index"):
        hi = health_index(df)
    with Timer("fault modes"):
        fm = fault_modes(df)
    labels = {}
    for e in fm["subsets"]:
        labels.update(e.pop("labels", {}))
    pd.Series(labels, name="fault_cluster").to_csv(WAREHOUSE.parent / "processed" / "fault_clusters.csv")
    export("explore", {"regimes": reg, "sensors": rank, "health": hi, "fault_modes": fm}, n=3)


if __name__ == "__main__":
    main()
