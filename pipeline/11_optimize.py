"""Stage 11 - prescriptive optimization (Pyomo + HiGHS; any MILP solver plugs in).

A. Spares-mix optimization (METRIC / VARI-METRIC family): choose stock levels for the engine and
   12 engine LRUs to maximize expected aircraft availability under a budget. Demand mix comes from
   real FAA SDR replacement counts; engine demand comes from the fleet simulator under each
   maintenance policy. Solved exactly as a multiple-choice knapsack MILP, cross-checked against
   classic greedy marginal analysis, for a full cost-availability frontier.
B. Engine induction scheduling: given the ensemble's predicted RUL (and its calibrated error
   distribution) for 200 engines in service, pick the week each engine goes to the depot over a
   12-week horizon, subject to depot slots and spare-engine availability. The plan is then scored
   against the true RULs NASA withheld.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy import stats

from common import PROCESSED, SITE_DATA, WAREHOUSE, Timer, export

N_AIRCRAFT = 24
SOLVER = "appsi_highs"   # swap for "gurobi", "cplex" or "cbc" - the model is unchanged

# Notional fighter-engine LRUs. sdr_parts = FAA SDR part names pooled into each LRU's demand weight.
LRUS = [
    ("ECU", "Engine control unit (FADEC/EEC)", ["EEC", "FADEC"], 420, 45),
    ("FMU", "Fuel metering unit", ["HMU", "FMU", "FUEL CONTROL"], 350, 50),
    ("VGA", "Variable-geometry actuator", ["ACTUATOR"], 90, 35),
    ("BLV", "Bleed valve", ["BLEED VALVE"], 65, 30),
    ("STV", "Start valve", ["START VALVE"], 40, 25),
    ("ATS", "Air turbine starter", ["STARTER"], 140, 40),
    ("AIV", "Anti-ice valve", ["ANTI-ICE VALVE"], 35, 25),
    ("SNS", "Sensors & transmitters", ["SENSOR", "TRANSMITTER"], 18, 20),
    ("TCP", "Thermocouples & probes", ["PROBE", "THERMOCOUPLE"], 22, 20),
    ("HAR", "Wire harness", ["WIRE HARNESS"], 30, 30),
    ("PMP", "Fuel & oil pumps", ["PUMP"], 85, 35),
    ("IGN", "Ignition exciter & igniters", ["IGNITER", "EXCITER"], 16, 15),
]
LRU_RATE_PER_AC_YEAR = 3.0     # GR&A: total engine-LRU removals per aircraft-year
ENGINE_COST_K = 15_000          # GR&A: notional spare engine (power module) cost, $K


# --------------------------------------------------------------------------- A
def ebo(mu: float, smax: int) -> np.ndarray:
    """Expected backorders for stock 0..smax with Poisson(mu) pipeline (Palm's theorem)."""
    xs = np.arange(0, int(mu * 4 + smax + 60))
    pmf = stats.poisson.pmf(xs, mu)
    return np.array([np.sum(np.maximum(xs - s, 0) * pmf) for s in range(smax + 1)])


def items_for(policy: dict, sdr_counts: pd.Series) -> list[dict]:
    w = np.array([sum(sdr_counts.get(p, 0) for p in parts) for _, _, parts, _, _ in LRUS], float)
    rates = w / w.sum() * LRU_RATE_PER_AC_YEAR
    items = []
    for (code, name, parts, cost, tat), rate, wt in zip(LRUS, rates, w):
        items.append({"code": code, "name": name, "cost": cost, "tat": tat, "rate": rate,
                      "sdr_reports": int(wt), "sdr_parts": parts})
    items.append({"code": "ENG", "name": "Spare engine (power module)", "cost": ENGINE_COST_K,
                  "tat": policy["tat"], "rate": policy["rate"], "sdr_reports": int(sdr_counts.get("ENGINE", 0)),
                  "sdr_parts": ["ENGINE"]})
    for it in items:
        it["pipeline"] = N_AIRCRAFT * it["rate"] * it["tat"] / 365.0
        it["smax"] = int(max(6, np.ceil(it["pipeline"] + 6 * np.sqrt(it["pipeline"]) + 4)))
        it["ebo"] = ebo(it["pipeline"], it["smax"])
        it["logA"] = np.log(np.clip(1 - it["ebo"] / N_AIRCRAFT, 1e-9, None))
    return items


def solve_milp(items: list[dict], budget_k: float) -> tuple[list[int], float]:
    m = pyo.ConcreteModel()
    pairs = [(i, s) for i, it in enumerate(items) for s in range(it["smax"] + 1)]
    m.y = pyo.Var(pairs, domain=pyo.Binary)
    m.one = pyo.Constraint(range(len(items)), rule=lambda m, i: sum(m.y[i, s] for s in range(items[i]["smax"] + 1)) == 1)
    m.budget = pyo.Constraint(expr=sum(items[i]["cost"] * s * m.y[i, s] for i, s in pairs) <= budget_k)
    m.obj = pyo.Objective(expr=sum(items[i]["logA"][s] * m.y[i, s] for i, s in pairs), sense=pyo.maximize)
    pyo.SolverFactory(SOLVER).solve(m)
    stock = [int(round(sum(s * pyo.value(m.y[i, s]) for s in range(it["smax"] + 1)))) for i, it in enumerate(items)]
    return stock, float(np.exp(sum(items[i]["logA"][s] for i, s in enumerate(stock))))


def greedy(items: list[dict]) -> list[dict]:
    """Marginal analysis: repeatedly buy the unit with the best availability gain per dollar."""
    stock = [0] * len(items)
    path = [{"cost": 0.0, "A": float(np.exp(sum(it["logA"][0] for it in items)))}]
    while True:
        best, gain = None, 0.0
        for i, it in enumerate(items):
            if stock[i] < it["smax"]:
                g = (it["logA"][stock[i] + 1] - it["logA"][stock[i]]) / it["cost"]
                if g > gain:
                    best, gain = i, g
        if best is None or gain < 1e-12:
            break
        stock[best] += 1
        cost = sum(s * it["cost"] for s, it in zip(stock, items))
        path.append({"cost": cost, "A": float(np.exp(sum(it["logA"][s] for s, it in zip(stock, items)))), "buy": items[best]["code"]})
        if cost > 160_000:
            break
    return path


def spares(sim: dict) -> dict:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    counts = con.execute("""SELECT part_name, count(*) n FROM fact_sdr
                            WHERE is_engine AND action = 'Replace' GROUP BY 1""").df().set_index("part_name").n
    con.close()
    pol = {p["name"]: p for p in sim["policies"]["policies"]}
    base = sim["base"]
    scen = {}
    for key, pname in (("reactive", "Run to failure"), ("predictive", "Predictive · ensemble")):
        p = pol[pname]
        un, pl = p["unsched_yr"]["mean"], p["planned_yr"]["mean"]
        tat = (un * base["tat_unsched"] + pl * base["tat_planned"]) / max(un + pl, 1e-9)
        scen[key] = {"rate": (un + pl) / N_AIRCRAFT, "tat": tat}
    out = {"scenarios": {}}
    budgets = np.arange(0, 100_001, 2_500)
    for key, policy in scen.items():
        items = items_for(policy, counts)
        frontier = []
        t0 = time.perf_counter()
        for b in budgets:
            stock, A = solve_milp(items, float(b))
            frontier.append({"budget": float(b), "A": A, "stock": stock,
                             "spent": float(sum(s * it["cost"] for s, it in zip(stock, items)))})
        solve_s = time.perf_counter() - t0
        g = greedy(items)
        # Cross-check: at each budget, the best greedy point that fits vs the exact MILP.
        gaps = []
        for f in frontier:
            fit = [p for p in g if p["cost"] <= f["budget"]]
            if fit:
                gaps.append(f["A"] - max(p["A"] for p in fit))
        gap_max = float(max(gaps))
        # Rule of thumb many shops use: stock each item at its mean pipeline, rounded up.
        rot = [int(np.ceil(it["pipeline"])) for it in items]
        rot_cost = float(sum(s * it["cost"] for s, it in zip(rot, items)))
        rot_A = float(np.exp(sum(it["logA"][min(s, it["smax"])] for s, it in zip(rot, items))))
        # cheapest MILP point that matches the rule of thumb's availability
        match = next((f for f in frontier if f["A"] >= rot_A), frontier[-1])
        out["scenarios"][key] = {
            "engine_rate": policy["rate"], "engine_tat": policy["tat"],
            "items": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in it.items() if k not in ("logA",)}
                      for it in items],
            "frontier": frontier, "greedy": g[::max(1, len(g) // 300)], "solve_seconds": solve_s,
            "greedy_max_gap": gap_max, "greedy_min_gap": float(min(gaps)),
            "rule_of_thumb": {"stock": rot, "cost": rot_cost, "A": rot_A, "milp_cost_same_A": match["spent"],
                              "saving": 1 - match["spent"] / rot_cost},
        }
        print(f"  {key}: engine rate {policy['rate']:.3f}/ac-yr, TAT {policy['tat']:.0f} d; rule-of-thumb "
              f"${rot_cost / 1000:.1f}M → A={rot_A:.3f}; MILP same A for ${match['spent'] / 1000:.1f}M "
              f"({1 - match['spent'] / rot_cost:.0%} less); {len(budgets)} MILPs in {solve_s:.1f}s; "
              f"MILP - greedy availability gap in [{min(gaps):.4f}, {gap_max:.4f}]")
    return out


# --------------------------------------------------------------------------- B
CYCLES_PER_WEEK = 4
WEEKS = 12
C_FAIL, C_PLAN, C_CYCLE = 3.6, 1.2, 0.006    # $M; wasted life valued at $6K per cycle


def rul_distribution(last: pd.DataFrame) -> list[np.ndarray]:
    """Per engine: samples of plausible true RUL = prediction + OOF residual from the same
    (operating profile, predicted-RUL) calibration cell. No test information is used."""
    oof = pd.read_parquet(PROCESSED / "oof_ens.parquet")
    groups = {"FD001": "single", "FD003": "single", "FD002": "multi", "FD004": "multi"}
    oof["g"] = oof.subset.map(groups)
    oof["res"] = oof.target - oof.pred
    bins = [0, 10, 20, 30, 40, 50, 65, 80, 100, 126]
    oof["b"] = np.digitize(oof.pred, bins[1:-1])
    cells = {k: g.res.to_numpy() for k, g in oof.groupby(["g", "b"])}
    rng = np.random.default_rng(7)
    out = []
    for _, r in last.iterrows():
        res = cells[(groups[r.subset], int(np.digitize(min(r.pred_ens, 125), bins[1:-1])))]
        out.append(np.clip(r.pred_ens + rng.choice(res, 4000), 0, None))
    return out


def schedule(last: pd.DataFrame, samples: list[np.ndarray], cap: int, spares0: int, returns_per_week: float,
             method: str) -> dict:
    n = len(last)
    t_w = np.array([CYCLES_PER_WEEK * (w - 1) for w in range(1, WEEKS + 1)], float)
    horizon = CYCLES_PER_WEEK * WEEKS
    F = np.array([[np.mean(s <= t) for t in t_w] for s in samples])               # P(fail before week w)
    Fh = np.array([np.mean(s <= horizon) for s in samples])
    waste = np.array([[np.mean(np.where(s > t, s - t, 0)) for t in t_w] for s in samples])
    cost = F * C_FAIL + (1 - F) * C_PLAN + C_CYCLE * waste                        # remove in week w
    cost_none = Fh * C_FAIL                                                        # leave in service
    x = np.zeros((n, WEEKS), int)
    if method == "milp":
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(n), range(WEEKS), domain=pyo.Binary)
        m.once = pyo.Constraint(range(n), rule=lambda m, e: sum(m.x[e, w] for w in range(WEEKS)) <= 1)
        m.cap = pyo.Constraint(range(WEEKS), rule=lambda m, w: sum(m.x[e, w] for e in range(n)) <= cap)
        m.spares = pyo.Constraint(range(WEEKS), rule=lambda m, w: sum(m.x[e, k] for e in range(n) for k in range(w + 1))
                                  <= spares0 + returns_per_week * w)
        m.obj = pyo.Objective(expr=sum(cost[e, w] * m.x[e, w] for e in range(n) for w in range(WEEKS))
                              + sum(cost_none[e] * (1 - sum(m.x[e, w] for w in range(WEEKS))) for e in range(n)))
        t0 = time.perf_counter()
        pyo.SolverFactory(SOLVER).solve(m)
        solve_s = time.perf_counter() - t0
        for e in range(n):
            for w in range(WEEKS):
                x[e, w] = int(round(pyo.value(m.x[e, w])))
    elif method == "threshold":
        # Rule: pull in the first week the predicted RUL is <= 20; overflow slips to the next week.
        solve_s = 0.0
        want = [next((w for w in range(WEEKS) if last.pred_ens.iloc[e] - t_w[w] <= 20), None) for e in range(n)]
        used = np.zeros(WEEKS, int)
        cum_cap = [spares0 + returns_per_week * w for w in range(WEEKS)]
        order = np.argsort([last.pred_ens.iloc[e] for e in range(n)])
        for e in order:
            w = want[e]
            while w is not None and w < WEEKS and (used[w] >= cap or used[: w + 1].sum() + 1 > cum_cap[w]):
                w += 1
            if w is not None and w < WEEKS:
                x[e, w] = 1
                used[w] += 1
    else:
        solve_s = 0.0
    planned = x.sum(1) == 1
    week = np.where(planned, x.argmax(1), -1)
    exp_cost = float(np.sum(np.where(planned, cost[np.arange(n), week.clip(0)], cost_none)))

    # Ground truth: NASA's withheld RUL for each engine.
    truth = last.rul.to_numpy()
    t_rm = np.where(planned, t_w[week.clip(0)], np.inf)
    fails = truth <= np.minimum(t_rm, horizon)
    wasted = np.where(planned & ~fails, truth - t_rm, 0)
    real_cost = float(np.sum(np.where(fails, C_FAIL, np.where(planned, C_PLAN + C_CYCLE * wasted, 0))))
    return {"method": method, "planned": int(planned.sum()), "expected_cost": exp_cost, "solve_seconds": solve_s,
            "realized_failures": int(fails.sum()), "realized_cost": real_cost,
            "mean_wasted": float(wasted[planned & ~fails].mean()) if (planned & ~fails).any() else 0.0,
            "per_week": x.sum(0).tolist(), "week": week.tolist()}


def induction(cap=3, spares0=4, returns=2.0) -> dict:
    last = pd.read_parquet(PROCESSED / "test_last_cycle.parquet")
    last = last[last.subset.isin(["FD001", "FD003"])].reset_index(drop=True)
    samples = rul_distribution(last)
    res = {m: schedule(last, samples, cap, spares0, returns, m) for m in ("none", "threshold", "milp")}
    for m, r in res.items():
        print(f"  {m:9s}: planned {r['planned']:3d}, expected ${r['expected_cost']:.1f}M, realized failures "
              f"{r['realized_failures']:2d}, realized ${r['realized_cost']:.1f}M, mean wasted {r['mean_wasted']:.1f} cyc")
    caps = []
    for c in (1, 2, 3, 4, 5, 6):
        r = schedule(last, samples, c, spares0, returns, "milp")
        caps.append({"cap": c, "expected_cost": r["expected_cost"], "realized_cost": r["realized_cost"],
                     "realized_failures": r["realized_failures"], "planned": r["planned"]})
    horizon = CYCLES_PER_WEEK * WEEKS
    fleet = []
    for i, r in last.iterrows():
        fleet.append({"id": r.engine_id, "pred": float(r.pred_ens), "lo": float(r.lo80), "hi": float(r.hi80),
                      "truth": int(r.rul), "p_fail_h": float(np.mean(samples[i] <= horizon)),
                      "week_milp": res["milp"]["week"][i], "week_rule": res["threshold"]["week"][i]})
    for r in res.values():
        r.pop("week")
    return {"params": {"cap": cap, "spares0": spares0, "returns_per_week": returns, "weeks": WEEKS,
                       "cycles_per_week": CYCLES_PER_WEEK, "c_fail": C_FAIL, "c_plan": C_PLAN, "c_cycle": C_CYCLE},
            "results": res, "capacity": caps, "fleet": fleet, "n": len(last)}


def main() -> None:
    sim = json.loads((SITE_DATA / "simulation.json").read_text())
    with Timer("spares optimization"):
        a = spares(sim)
    with Timer("induction scheduling"):
        b = induction()
    export("optimization", {"spares": a, "induction": b, "solver": SOLVER, "n_aircraft": N_AIRCRAFT,
                            "lru_rate": LRU_RATE_PER_AC_YEAR})


if __name__ == "__main__":
    main()
