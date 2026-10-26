"""Stage 10 - fleet readiness experiments with the GPU Monte Carlo simulator.

1. Calibrate the non-engine background so the baseline squadron's MC rate matches the target.
2. CPU vs GPU scaling benchmark.
3. Maintenance-policy comparison (threshold tuned on one random stream, scored on another).
4. Value of prognostics: readiness as a function of model error.
5. Spares x depot-turnaround readiness surfaces.
6. Analysis of alternatives (courses of action) with cost.
(The day-by-day squadron board for the website is traced separately in 10b_board.py.)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from common import PROCESSED, Timer, export
from sim_engine import Library, SimConfig, simulate, summarize

PROFILES = PROCESSED / "sim_profiles.npz"
BASE_MC = 0.60          # calibration target for the status-quo squadron
PBL_TARGET = 0.65       # notional PBL mission-capable requirement
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")
_libs: dict = {}
REPS = {"count": 0}      # every simulated squadron-year, for the record


def lib(model="ens", alpha=1.0, device=DEV) -> Library:
    key = (model, round(alpha, 4), device.type)
    if key not in _libs:
        _libs[key] = Library(PROFILES, model, alpha, device)
    return _libs[key]


def run(cfg: SimConfig, R: int, seed: int, device=DEV, **kw) -> dict:
    REPS["count"] += R * cfg.days / 365
    return simulate(cfg, R, lib(cfg.model, cfg.alpha, device), device, seed=seed, **kw)


def calibrate(base: SimConfig) -> float:
    lo, hi = 0.0, 0.25
    for _ in range(12):
        q = (lo + hi) / 2
        mc = run(base.with_(q_other=q), 800, seed=11).get("mc").mean()
        lo, hi = (q, hi) if mc > BASE_MC else (lo, q)
    q = (lo + hi) / 2
    print(f"  calibrated q_other = {q:.4f}")
    return q


def benchmark(base: SimConfig) -> list[dict]:
    rows = []
    for R in (256, 1024, 4096, 16384):
        row = {"R": R}
        for name, dev in (("cpu", CPU), (DEV.type, DEV)):
            if name == "cpu" and R > 4096:
                cfg_b = base.with_(days=120, warmup=0)
                t0 = time.perf_counter()
                run(cfg_b, R, 1, device=dev)
                row[name] = (time.perf_counter() - t0) * (base.days + base.warmup) / 120
                row[f"{name}_extrapolated"] = True
                continue
            run(base.with_(days=20, warmup=0), R, 1, device=dev)  # warm caches / kernels
            t0 = time.perf_counter()
            run(base, R, 1, device=dev)
            row[name] = time.perf_counter() - t0
        row["aircraft_days"] = R * base.n_aircraft * (base.days + base.warmup)
        rows.append(row)
        print(f"  R={R:6d}: " + ", ".join(f"{k}={v:.2f}s" for k, v in row.items() if isinstance(v, float)))
    return rows


def tune_threshold(cfg: SimConfig, grid=range(2, 42, 3), R=1200, metric="cost") -> tuple[int, list]:
    scores = []
    for th in grid:
        r = run(cfg.with_(threshold=th), R, seed=101)
        val = r["cost_yr"].mean() if metric == "cost" else -r["mc"].mean()
        scores.append({"threshold": th, "mc": float(r["mc"].mean()), "cost": float(r["cost_yr"].mean()),
                       "unsched": float(r["unsched_yr"].mean())})
    best = min(scores, key=lambda s: s["cost"] if metric == "cost" else -s["mc"])
    return best["threshold"], scores


def policies(base: SimConfig, R=4000) -> dict:
    out = []
    th_ens, curve_ens = tune_threshold(base.with_(policy="predictive", model="ens"))
    th_ridge, _ = tune_threshold(base.with_(policy="predictive", model="ridge"))
    th_oracle, _ = tune_threshold(base.with_(policy="predictive", model="ens", alpha=0.0), grid=range(1, 21, 2))
    hard_scores = []
    for T in range(130, 231, 10):
        r = run(base.with_(policy="hard_time", hard_time=T), 1200, seed=101)
        hard_scores.append({"T": T, "mc": float(r["mc"].mean()), "cost": float(r["cost_yr"].mean())})
    T_best = min(hard_scores, key=lambda s: s["cost"])["T"]
    specs = [
        ("Run to failure", "Reactive: fly every engine until it fails", base.with_(policy="reactive")),
        ("Hard-time removal", f"Pull every engine at {T_best} cycles (cost-optimal age)", base.with_(policy="hard_time", hard_time=T_best)),
        ("Predictive · ridge", f"Pull when a linear model predicts ≤ {th_ridge} cycles left", base.with_(policy="predictive", model="ridge", threshold=th_ridge)),
        ("Predictive · ensemble", f"Pull when the LightGBM + CNN ensemble predicts ≤ {th_ens} cycles left", base.with_(policy="predictive", model="ens", threshold=th_ens)),
        ("Perfect foresight", f"Oracle that knows the true RUL; pull at ≤ {th_oracle} cycles left", base.with_(policy="predictive", alpha=0.0, threshold=th_oracle)),
    ]
    for name, desc, cfg in specs:
        r = run(cfg, R, seed=202)
        s = summarize(r, PBL_TARGET)
        s.update(name=name, desc=desc, policy=cfg.policy, threshold=cfg.threshold, hard_time=cfg.hard_time,
                 rmse_decision=lib(cfg.model, cfg.alpha).rmse_decision if cfg.policy == "predictive" else None,
                 hist=np.histogram(r["mc"], bins=np.linspace(0.40, 0.80, 41))[0].tolist())
        out.append(s)
        print(f"  {name:24s} MC {s['mc']['mean']:.3f}  unsched/yr {s['unsched_yr']['mean']:5.2f}  "
              f"planned/yr {s['planned_yr']['mean']:5.2f}  cost {s['cost_yr']['mean']:6.2f}  P(MC>={PBL_TARGET}) {s['p_meet_target']:.2f}")
    return {"policies": out, "threshold_curve": curve_ens, "hard_time_curve": hard_scores,
            "thresholds": {"ens": th_ens, "ridge": th_ridge, "oracle": th_oracle, "hard_time": T_best},
            "hist_bins": np.linspace(0.40, 0.80, 41).tolist()}


def value_of_prognostics(base: SimConfig, alphas=(0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)) -> list[dict]:
    rows = []
    for a in alphas:
        cfg = base.with_(policy="predictive", model="ens", alpha=a)
        th, _ = tune_threshold(cfg, grid=range(2, 62, 4), R=800)
        r = run(cfg.with_(threshold=th), 2500, seed=303)
        rows.append({"alpha": a, "rmse": lib("ens", a).rmse_decision, "threshold": th,
                     "mc": float(r["mc"].mean()), "mc_se": float(r["mc"].std() / np.sqrt(len(r["mc"]))),
                     "unsched": float(r["unsched_yr"].mean()), "cost": float(r["cost_yr"].mean()),
                     "life_used": float(np.nanmean(r["life_used"]))})
        print(f"  alpha={a:4.2f} rmse={rows[-1]['rmse']:5.2f} th={th:2d} MC={rows[-1]['mc']:.3f} cost={rows[-1]['cost']:.2f}")
    return rows


def surfaces(base: SimConfig, th: int) -> dict:
    spares = list(range(0, 8))
    tats = [60, 90, 120, 150, 180]
    out = {"spares": spares, "tats": tats}
    for name, cfg in (("reactive", base.with_(policy="reactive")),
                      ("predictive", base.with_(policy="predictive", threshold=th))):
        grid = []
        for tat in tats:
            row = []
            for s in spares:
                r = run(cfg.with_(n_spares=s, tat_unsched=tat, tat_planned=tat * 0.625), 600, seed=404)
                row.append({"mc": float(r["mc"].mean()), "await": float(r["await_eng"].mean()),
                            "cost": float(r["cost_yr"].mean())})
            grid.append(row)
        out[name] = grid
    return out


PHM_COST = 0.6       # $M/yr: prognostics program (data pipeline, model upkeep, analysts)
DEPOT_INCENTIVE = 2.5  # $M/yr: PBL incentive paid for a 25% faster depot turnaround


def aoa(base: SimConfig, th: int, R=4000) -> list[dict]:
    phm, inc = PHM_COST, DEPOT_INCENTIVE
    fast = dict(tat_unsched=90, tat_planned=56, cost_program_year=inc)
    coas = [
        ("COA 0", "Status quo", "Run to failure, 3 spare engines, 120-day depot turnaround", base.with_(policy="reactive")),
        ("COA 1", "Buy 2 spare engines", "Run to failure with 5 spares", base.with_(policy="reactive", n_spares=5)),
        ("COA 2", "Faster depot (PBL incentive)", f"Turnaround cut 25% (120 → 90 days) for a ${inc}M/yr incentive",
         base.with_(policy="reactive", **fast)),
        ("COA 3", "Predictive maintenance", f"Ensemble model, pull at ≤ {th} cycles; ${phm}M/yr prognostics program",
         base.with_(policy="predictive", threshold=th, cost_phm_year=phm)),
        ("COA 4", "Predictive + 1 spare", "COA 3 plus one more spare engine",
         base.with_(policy="predictive", threshold=th, cost_phm_year=phm, n_spares=4)),
        ("COA 5", "Predictive + faster depot", "COA 3 plus the faster depot",
         base.with_(policy="predictive", threshold=th, cost_phm_year=phm, **fast)),
        ("COA 6", "Predictive + spare + depot", "COA 3 plus one spare and the faster depot",
         base.with_(policy="predictive", threshold=th, cost_phm_year=phm, n_spares=4, **fast)),
    ]
    rows = []
    for cid, name, desc, cfg in coas:
        r = run(cfg, R, seed=505)
        s = summarize(r, PBL_TARGET)
        s.update(id=cid, name=name, desc=desc, n_spares=cfg.n_spares, tat=cfg.tat_unsched, policy=cfg.policy,
                 program_cost=cfg.cost_phm_year + cfg.cost_program_year,
                 hist=np.histogram(r["mc"], bins=np.linspace(0.40, 0.80, 41))[0].tolist())
        rows.append(s)
        print(f"  {cid} {name:28s} MC {s['mc']['mean']:.3f} [{s['mc']['p05']:.3f},{s['mc']['p95']:.3f}] "
              f"P>={PBL_TARGET}: {s['p_meet_target']:.2f}  cost {s['cost_yr']['mean']:.2f}")
    return rows


def sensitivity(base: SimConfig, th: int, R=1500) -> list[dict]:
    """How much of the predictive-maintenance benefit survives if a ground rule is wrong?"""
    cases = [
        ("Baseline ground rules", {}),
        ("Planned depot visit only 15% shorter", {"tat_planned": 102.0}),
        ("Planned depot visit no shorter", {"tat_planned": 120.0}),
        ("Planned depot visit 50% shorter", {"tat_planned": 60.0}),
        ("Unscheduled removal 2× planned cost", {"cost_unsched": 2.4}),
        ("Unscheduled removal 5× planned cost", {"cost_unsched": 6.0}),
        ("Depot 25% slower overall", {"tat_unsched": 150.0, "tat_planned": 94.0}),
        ("Higher flying tempo (+20%)", {"p_fly": 0.9}),
        ("Prognostic error doubled", {"alpha": 2.0}),
        ("Two spare engines instead of three", {"n_spares": 2}),
    ]
    rows = []
    for label, kw in cases:
        react_kw = {k: v for k, v in kw.items() if k != "alpha"}
        r0 = run(base.with_(policy="reactive", **react_kw), R, seed=707)
        r1 = run(base.with_(policy="predictive", threshold=th, **kw), R, seed=707)
        rows.append({"case": label, "mc_reactive": float(r0["mc"].mean()), "mc_predictive": float(r1["mc"].mean()),
                     "mc_gain": float(r1["mc"].mean() - r0["mc"].mean()),
                     "cost_reactive": float(r0["cost_yr"].mean()), "cost_predictive": float(r1["cost_yr"].mean()),
                     "cost_delta": float(r1["cost_yr"].mean() - r0["cost_yr"].mean())})
        print(f"  {label:38s} MC gain {rows[-1]['mc_gain'] * 100:+5.1f} pts  cost Δ {rows[-1]['cost_delta']:+6.1f} $M/yr")
    return rows


def main() -> None:
    base = SimConfig()
    print(f"  device: {DEV}")
    with Timer("calibration"):
        q = calibrate(base)
    base = base.with_(q_other=q)
    with Timer("benchmark"):
        bench = benchmark(base)
    with Timer("policies"):
        pol = policies(base)
    th = pol["thresholds"]["ens"]
    with Timer("value of prognostics"):
        vop = value_of_prognostics(base)
    with Timer("surfaces"):
        surf = surfaces(base, th)
    with Timer("analysis of alternatives"):
        alts = aoa(base, th)
    with Timer("sensitivity"):
        sens = sensitivity(base, th)
    # Reference results for validating the in-browser JavaScript port of the simulator.
    ref = []
    for cfg in (base.with_(policy="reactive"), base.with_(policy="predictive", threshold=th),
                base.with_(policy="hard_time", hard_time=pol["thresholds"]["hard_time"]),
                base.with_(policy="reactive", n_spares=6, tat_unsched=90, tat_planned=56)):
        r = run(cfg, 3000, seed=606)
        ref.append({"config": cfg.dict(), "mc": float(r["mc"].mean()), "mc_sd": float(r["mc"].std()),
                    "unsched": float(r["unsched_yr"].mean()), "planned": float(r["planned_yr"].mean()),
                    "await": float(r["await_eng"].mean())})
    export("simulation", {"base": base.dict(), "pbl_target": PBL_TARGET, "base_mc_target": BASE_MC,
                          "device": DEV.type, "benchmark": bench, "policies": pol, "value": vop,
                          "surfaces": surf, "aoa": alts, "sensitivity": sens, "reference": ref,
                          "squadron_years": REPS["count"]})


if __name__ == "__main__":
    main()
