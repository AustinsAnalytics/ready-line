"""Stage 4 - formal hypothesis tests and reliability modeling.

H1  Engines wear out (Weibull shape > 1)        -> is preventive / predictive removal worth anything?
H2  Operating environment changes engine life    -> FD001 (1 condition) vs FD002 (6 conditions)
H3  The discovered fault mode changes life       -> latent cluster from stage 3 vs life
H4  Engine problems interrupt flights more often -> logistic regression on 600k+ FAA SDRs
H5  COVID-19 changed reporting volume            -> interrupted time series on monthly SDR counts

Plus the classic age-replacement optimization that H1 makes possible.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, WeibullFitter
from lifelines.statistics import logrank_test
from scipy import integrate, optimize, stats

from common import JASC_CHAPTERS, PROCESSED, SEED, WAREHOUSE, Timer, export

rng = np.random.default_rng(SEED)


def lives() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    df = con.execute("SELECT engine_id, subset, split, total_life FROM dim_engine WHERE split = 'train'").df()
    con.close()
    clusters = pd.read_csv(PROCESSED / "fault_clusters.csv", index_col=0)["fault_cluster"]
    df["cluster"] = df.engine_id.map(clusters)
    return df


def boot_mean_diff(a, b, B=5000):
    a, b = np.asarray(a), np.asarray(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return float(np.mean(a) - np.mean(b)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# ---------------------------------------------------------------------------- H1
def h1_wearout(df: pd.DataFrame) -> dict:
    out = {"groups": []}
    groups = {"HPC fault only (FD001+FD002)": df[df.subset.isin(["FD001", "FD002"])],
              "HPC + fan faults (FD003+FD004)": df[df.subset.isin(["FD003", "FD004"])],
              "All 709 training engines": df}
    for name, g in groups.items():
        t = g.total_life.to_numpy(float)
        wf = WeibullFitter().fit(t)
        beta, eta = float(wf.rho_), float(wf.lambda_)
        summ = wf.summary
        beta_lo, beta_hi = float(summ.loc["rho_", "coef lower 95%"]), float(summ.loc["rho_", "coef upper 95%"])
        ll_w = float(wf.log_likelihood_)
        ll_e = float(np.sum(stats.expon.logpdf(t, scale=t.mean())))
        lr = 2 * (ll_w - ll_e)
        p = float(stats.chi2.sf(lr, 1))
        # 3-parameter Weibull (failure-free period), fit by MLE for the simulation.
        c3, loc3, scale3 = stats.weibull_min.fit(t, 2.0, loc=0.8 * t.min(), scale=t.mean() - 0.8 * t.min())
        ks3 = stats.kstest(t, "weibull_min", args=(c3, loc3, scale3))
        ks2 = stats.kstest(t, "weibull_min", args=(beta, 0, eta))
        km = KaplanMeierFitter().fit(t)
        grid = np.linspace(0, t.max() * 1.05, 120)
        entry = {
            "name": name, "n": len(t), "mean": t.mean(), "sd": t.std(ddof=1), "min": t.min(), "max": t.max(),
            "beta": beta, "beta_lo": beta_lo, "beta_hi": beta_hi, "eta": eta,
            "lr_vs_exponential": lr, "p": p,
            "w3": {"beta": c3, "gamma": loc3, "eta": scale3, "ks_p": ks3.pvalue}, "w2_ks_p": ks2.pvalue,
            "km": {"t": km.survival_function_.index.tolist(), "s": km.survival_function_.iloc[:, 0].tolist()},
            "grid": grid.tolist(),
            "s_w2": stats.weibull_min.sf(grid, beta, 0, eta).tolist(),
            "s_w3": stats.weibull_min.sf(grid, c3, loc3, scale3).tolist(),
            "h_w3": (stats.weibull_min.pdf(grid, c3, loc3, scale3) /
                     np.maximum(stats.weibull_min.sf(grid, c3, loc3, scale3), 1e-12)).tolist(),
        }
        # Weibull probability plot (median ranks, Bernard's approximation).
        ts = np.sort(t)
        F = (np.arange(1, len(ts) + 1) - 0.3) / (len(ts) + 0.4)
        entry["prob_plot"] = {"x": np.log(ts).tolist(), "y": np.log(-np.log(1 - F)).tolist()}
        out["groups"].append(entry)
        print(f"  {name}: beta={beta:.2f} [{beta_lo:.2f},{beta_hi:.2f}] eta={eta:.1f}; LR={lr:.0f}, p={p:.1e}; "
              f"3p: beta={c3:.2f} gamma={loc3:.1f} eta={scale3:.1f} KS p={ks3.pvalue:.3f} (2p KS p={ks2.pvalue:.3f})")
    out["decision"] = "reject" if all(g["p"] < 0.05 for g in out["groups"]) else "fail to reject"
    return out


def age_replacement(beta: float, gamma: float, eta: float, ratios=(2, 3, 5, 8, 12, 20)) -> dict:
    """Cost rate of an age-replacement policy: C(T) = [Cp R(T) + Cf F(T)] / integral_0^T R(t) dt.
    Cp = planned removal cost (normalized to 1), Cf = unplanned failure cost = ratio * Cp."""
    R = lambda x: stats.weibull_min.sf(x, beta, gamma, eta)  # noqa: E731
    mttf = stats.weibull_min.mean(beta, gamma, eta)
    curves = []
    Ts = np.linspace(gamma + 5, stats.weibull_min.ppf(0.999, beta, gamma, eta), 140)
    cum = np.array([integrate.quad(R, 0, T, limit=200)[0] for T in Ts])
    for k in ratios:
        cost = (1 * R(Ts) + k * (1 - R(Ts))) / cum
        res = optimize.minimize_scalar(
            lambda T: (R(T) + k * (1 - R(T))) / integrate.quad(R, 0, T, limit=200)[0],
            bounds=(gamma + 1, Ts[-1]), method="bounded")
        run_to_failure = k / mttf
        curves.append({"ratio": k, "cost": cost.tolist(), "t_star": float(res.x), "c_star": float(res.fun),
                       "rtf": run_to_failure, "saving": 1 - float(res.fun) / run_to_failure,
                       "p_fail_before": float(1 - R(res.x))})
    return {"T": Ts.tolist(), "curves": curves, "mttf": mttf}


# ---------------------------------------------------------------------------- H2/H3
def two_sample(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    lr = logrank_test(a, b)
    mw = stats.mannwhitneyu(a, b)
    ks = stats.ks_2samp(a, b)
    d, lo, hi = boot_mean_diff(a, b)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"a": label_a, "b": label_b, "n_a": len(a), "n_b": len(b), "mean_a": a.mean(), "mean_b": b.mean(),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "logrank_chi2": float(lr.test_statistic), "logrank_p": float(lr.p_value),
            "mw_p": float(mw.pvalue), "ks_p": float(ks.pvalue),
            "diff": d, "diff_lo": lo, "diff_hi": hi, "cohens_d": (a.mean() - b.mean()) / pooled,
            "hist_a": np.histogram(a, bins=np.arange(100, 575, 25))[0].tolist(),
            "hist_b": np.histogram(b, bins=np.arange(100, 575, 25))[0].tolist(),
            "bins": np.arange(100, 575, 25).tolist()}


# ---------------------------------------------------------------------------- H4
def h4_interruptions() -> dict:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    df = con.execute("""
        SELECT chapter, is_engine, interruption::INT AS y, stage_group, year,
               CASE WHEN aircraft_make IN ('BOEING','AIRBUS','CNDAIR','EMB','DOUG','CESSNA','BOMBDR')
                    THEN aircraft_make ELSE 'OTHER' END AS make
        FROM fact_sdr
        WHERE stage_group IN ('Take-off & climb', 'Cruise', 'Descent & landing')
    """).df()
    con.close()
    df["engine"] = df.is_engine.astype(int)
    model = smf.glm("y ~ engine + C(stage_group) + C(make) + C(year)", data=df,
                    family=sm.families.Binomial()).fit(cov_type="HC1")
    b, se = model.params["engine"], model.bse["engine"]
    crude = pd.crosstab(df.engine, df.y)
    chi2, p_chi, *_ = stats.chi2_contingency(crude)

    # Chapter-level interruption rates with Wilson intervals (forest plot).
    rows = []
    for ch, g in df.groupby("chapter"):
        n, k = len(g), int(g.y.sum())
        if n < 400:
            continue
        lo, hi = sm.stats.proportion_confint(k, n, method="wilson")
        rows.append({"chapter": ch, "title": JASC_CHAPTERS.get(ch, f"Chapter {ch}"), "n": n, "rate": k / n,
                     "lo": lo, "hi": hi, "is_engine": bool(g.is_engine.iloc[0])})
    rows.sort(key=lambda r: -r["rate"])
    print(f"  H4: n={len(df):,}; adjusted OR(engine)={np.exp(b):.2f} [{np.exp(b - 1.96 * se):.2f},"
          f" {np.exp(b + 1.96 * se):.2f}], p={model.pvalues['engine']:.1e}")
    return {"n": len(df), "base_rate": df.y.mean(), "engine_rate": df[df.engine == 1].y.mean(),
            "other_rate": df[df.engine == 0].y.mean(),
            "or": float(np.exp(b)), "or_lo": float(np.exp(b - 1.96 * se)), "or_hi": float(np.exp(b + 1.96 * se)),
            "p": float(model.pvalues["engine"]), "chi2": float(chi2), "chi2_p": float(p_chi),
            "controls": ["stage of flight", "aircraft make", "year"], "chapters": rows}


# ---------------------------------------------------------------------------- H5
def h5_covid() -> dict:
    m = pd.read_parquet(PROCESSED / "sdr_monthly")
    s = m.groupby("month").reports.sum().sort_index()
    s.index = pd.to_datetime(s.index)
    s = s["2016-01-01":"2025-12-31"]
    df = pd.DataFrame({"y": s.values, "month": s.index})
    df["t"] = np.arange(len(df))
    df["moy"] = df.month.dt.month
    df["acute"] = ((df.month >= "2020-04-01") & (df.month <= "2021-03-01")).astype(int)
    df["post"] = (df.month >= "2021-04-01").astype(int)
    df["t_post"] = np.where(df.post == 1, df.t - df.t[df.post == 1].min(), 0)
    df["log_y"] = np.log(df.y)
    fit = smf.ols("log_y ~ t + C(moy) + acute + post + t_post", data=df).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    pre = df.month < "2020-04-01"
    cf = smf.ols("log_y ~ t + C(moy)", data=df[pre]).fit()
    df["counterfactual"] = np.exp(cf.predict(df))
    df["fitted"] = np.exp(fit.fittedvalues)
    eff = fit.params["acute"]
    lo, hi = fit.conf_int().loc["acute"]
    print(f"  H5: acute-period level shift {np.expm1(eff):+.1%} [{np.expm1(lo):+.1%}, {np.expm1(hi):+.1%}],"
          f" p={fit.pvalues['acute']:.1e}")
    return {"months": df.month.dt.strftime("%Y-%m").tolist(), "y": df.y.tolist(),
            "fitted": df.fitted.tolist(), "counterfactual": df.counterfactual.tolist(),
            "acute_effect": float(np.expm1(eff)), "acute_lo": float(np.expm1(lo)), "acute_hi": float(np.expm1(hi)),
            "acute_p": float(fit.pvalues["acute"]),
            "post_effect": float(np.expm1(fit.params["post"])), "post_p": float(fit.pvalues["post"]),
            "trend_pre_annual": float(np.expm1(fit.params["t"] * 12)), "r2": float(fit.rsquared)}


def main() -> None:
    df = lives()
    with Timer("H1 Weibull"):
        h1 = h1_wearout(df)
    allg = h1["groups"][2]["w3"]
    with Timer("age replacement"):
        ar = age_replacement(allg["beta"], allg["gamma"], allg["eta"])
    for c in ar["curves"]:
        print(f"  Cf/Cp={c['ratio']}: T*={c['t_star']:.0f} cycles, saving {c['saving']:.1%} vs run-to-failure")
    with Timer("H2/H3"):
        g = lambda s: df[df.subset == s].total_life.to_numpy(float)  # noqa: E731
        h2 = two_sample(g("FD001"), g("FD002"), "FD001 · 1 operating condition", "FD002 · 6 operating conditions")
        h3a = two_sample(g("FD001"), g("FD003"), "FD001 · HPC fault", "FD003 · HPC + fan faults")
        fd34 = df[df.subset.isin(["FD003", "FD004"])]
        h3 = two_sample(fd34[fd34.cluster == 0].total_life.to_numpy(float), fd34[fd34.cluster == 1].total_life.to_numpy(float),
                        "Discovered mode A", "Discovered mode B")
        print(f"  H2 logrank p={h2['logrank_p']:.3f}; H3 (modes) logrank p={h3['logrank_p']:.1e},"
              f" diff={h3['diff']:.0f} [{h3['diff_lo']:.0f},{h3['diff_hi']:.0f}]")
    with Timer("H4 logistic"):
        h4 = h4_interruptions()
    with Timer("H5 ITS"):
        h5 = h5_covid()
    export("hypotheses", {"h1": h1, "age_replacement": ar, "h2": h2, "h3_subsets": h3a, "h3": h3, "h4": h4, "h5": h5})


if __name__ == "__main__":
    main()
