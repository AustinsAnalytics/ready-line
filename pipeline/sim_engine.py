"""Squadron readiness simulator - vectorized Monte Carlo over thousands of replications at once.

Model (daily time step, one engine per aircraft):
  * Each mission-capable (MC) aircraft flies a sortie on a flying day with probability p_fly;
    one sortie = one C-MAPSS cycle of engine life.
  * Every engine is one of the 709 real run-to-failure engines (life + the model's actual
    out-of-fold RUL predictions over its last 100 cycles), drawn at random when installed.
  * Policies: run-to-failure; hard-time removal at a fixed age; or predictive removal when the
    model's predicted RUL drops to a threshold (and up to `opportunistic` cycles earlier if a
    spare is already on the shelf).  A planned removal with no spare leaves the aircraft
    waiting for an engine, exactly like a failure - but with a shorter, cheaper depot visit.
  * Removed engines queue for a capacity-limited depot (FIFO, lognormal turnaround) and return
    as good as new.  No spare -> the aircraft waits (NMCS, awaiting engine).
  * Non-engine discrepancies arrive per sortie and ground the aircraft (NMCM or NMCS),
    calibrated so the baseline squadron sits inside the GAO-reported fleet MC range.
Everything is a tensor of shape (replications, aircraft) or (replications, engines), so the same
code runs on CPU, Apple-silicon GPU (MPS) or CUDA.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

CAP = 125


@dataclass(frozen=True)
class SimConfig:
    n_aircraft: int = 24
    n_spares: int = 3
    days: int = 365
    warmup: int = 365
    p_fly: float = 0.75
    flying_days: int = 5
    policy: str = "reactive"          # reactive | hard_time | predictive
    hard_time: int = 140
    threshold: int = 10
    opportunistic: int = 15
    model: str = "ens"
    alpha: float = 1.0                # prognostic error scale: 0 = perfect foresight, 1 = our model
    swap_unsched: float = 4.0
    swap_planned: float = 2.0
    tat_unsched: float = 120.0
    tat_planned: float = 75.0
    tat_cv: float = 0.30
    depot_lines: int = 6
    q_other: float = 0.07
    other_nmcm_days: float = 2.0
    other_nmcs_share: float = 0.25
    other_nmcs_days: float = 10.0
    cost_unsched: float = 3.6         # $M per unscheduled removal (repair + collateral + expedite)
    cost_planned: float = 1.2         # $M per planned removal
    cost_spare_year: float = 1.5      # $M per spare engine per year (capital charge)
    cost_phm_year: float = 0.0        # $M per year for a prognostics program
    cost_program_year: float = 0.0    # $M per year for other program changes (e.g. a depot incentive)

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)

    def dict(self) -> dict:
        return asdict(self)


class Library:
    """Engine lives and RUL prediction trajectories, flattened for O(1) lookup by (profile, age)."""

    def __init__(self, npz_path, model: str, alpha: float, device: torch.device):
        z = np.load(npz_path)
        life = z["life"].astype(np.int64)
        tail = z[model].astype(np.float32)
        P, n = tail.shape
        L = int(life.max()) + 2
        ages = np.arange(L)[None, :]
        truth = np.clip(life[:, None] - ages, 0, CAP).astype(np.float32)
        pred = np.full((P, L), CAP, np.float32)
        for p in range(P):
            pred[p, life[p] - n + 1: life[p] + 1] = tail[p]
        pred_a = truth + alpha * (pred - truth)
        self.P, self.L = P, L
        self.life = torch.tensor(life, device=device)
        self.pred = torch.tensor(pred_a.ravel(), device=device)
        # decision-zone RMSE of the (scaled) predictions, for reporting
        m = truth <= 60
        m[:, 0] = False
        valid = (ages <= life[:, None]) & (ages >= (life[:, None] - n + 1))
        self.rmse_decision = float(np.sqrt(np.mean(((pred_a - truth)[m & valid]) ** 2)))


def _lognormal(shape, mean, cv, device):
    s2 = math.log(1 + cv * cv)
    mu = math.log(mean) - s2 / 2
    return torch.exp(mu + math.sqrt(s2) * torch.randn(shape, device=device))


STATUS = {0: "F", 1: "M", 2: "E", 3: "S", 4: "m", 5: "s"}


def simulate(cfg: SimConfig, R: int, lib: Library, device: torch.device, seed: int = 0, trace: bool = False) -> dict:
    torch.manual_seed(seed)
    dev = device
    A, S = cfg.n_aircraft, cfg.n_spares
    E = A + S
    D = cfg.warmup + cfg.days
    i64 = torch.int64
    zero = torch.zeros((), device=dev)

    # ---------------------------------------------------------------- state
    # Start near steady state: an engine in service at a random moment has a length-biased life
    # and a uniformly distributed age within it (renewal theory); spares start fresh.
    e_prof = torch.randint(lib.P, (R, E), device=dev)
    biased = torch.multinomial(lib.life.float().cpu(), R * A, replacement=True).view(R, A).to(dev)
    e_prof[:, :A] = biased
    e_life = lib.life[e_prof]
    e_age = torch.zeros((R, E), dtype=i64, device=dev)
    e_age[:, :A] = (torch.rand((R, A), device=dev) * e_life[:, :A].float()).long()
    e_loc = torch.zeros((R, E), dtype=i64, device=dev)      # 0 spare on shelf, 1 installed, 2 at depot
    e_loc[:, :A] = 1
    e_ready = torch.zeros((R, E), device=dev)
    a_eng = torch.arange(A, device=dev).expand(R, A).clone()
    a_down = torch.zeros((R, A), device=dev)
    a_kind = torch.zeros((R, A), dtype=i64, device=dev)     # 1 engine change, 2 other NMCM, 3 other NMCS
    a_swap = torch.full((R, A), cfg.swap_unsched, device=dev)
    line_free = torch.zeros((R, cfg.depot_lines), device=dev)
    keys = ["mc", "await_eng", "nmcm_eng", "other_m", "other_s", "sorties", "unsched", "planned",
            "life_used", "wasted", "queue_wait", "inducted", "depot_wip"]
    acc = {k: torch.zeros(R, device=dev) for k in keys}
    eng_cols = torch.arange(E, device=dev).expand(R, E)
    trace_rows, trace_left, trace_est, events = [], [], [], []

    def allocate(need):
        """Give shelf spares to aircraft in `need`, lowest aircraft index first."""
        spare = e_loc == 0
        n_sp = spare.sum(1, keepdim=True)
        rank = need.long().cumsum(1) - 1
        ok = need & (rank < n_sp)
        sp_rank = spare.long().cumsum(1) - 1
        inv = torch.full((R, E + 1), E, dtype=i64, device=dev)
        inv.scatter_(1, torch.where(spare, sp_rank, E), eng_cols)
        eng = inv.gather(1, rank.clamp(0, E))
        return ok, eng

    def install(ok, eng, down_days, t):
        nonlocal e_loc, a_eng, a_down, a_kind
        idx = torch.where(ok, eng, E)
        ext = torch.cat([e_loc, torch.zeros((R, 1), dtype=i64, device=dev)], 1)
        ext.scatter_(1, idx, torch.ones_like(idx))
        e_loc = ext[:, :E]
        a_eng = torch.where(ok, eng, a_eng)
        a_down = torch.where(ok, t + 1 + down_days, a_down)
        a_kind = torch.where(ok, torch.ones_like(a_kind), a_kind)

    def induct(mask, eng, tat_mean, t, meas):
        nonlocal line_free, e_ready, e_loc
        r = mask.long().cumsum(1) - 1
        kmax = int(mask.sum(1).max().item()) if mask.numel() else 0
        for k in range(kmax):
            sel = mask & (r == k)
            has = sel.any(1)
            ek = (sel.long() * eng).sum(1, keepdim=True)
            line = line_free.argmin(1, keepdim=True)
            free = line_free.gather(1, line)
            start = torch.clamp(free, min=float(t))
            done = start + _lognormal((R, 1), tat_mean, cfg.tat_cv, dev)
            h = has[:, None]
            line_free.scatter_(1, line, torch.where(h, done, free))
            e_ready.scatter_(1, ek, torch.where(h, done, e_ready.gather(1, ek)))
            e_loc.scatter_(1, ek, torch.where(h, torch.full_like(ek, 2), e_loc.gather(1, ek)))
            if meas:
                acc["queue_wait"] += torch.where(has, (start - t).squeeze(1), zero)
                acc["inducted"] += has.float()

    for t in range(D):
        meas = t >= cfg.warmup
        flying_day = (t % 7) < cfg.flying_days

        # 1. depot returns: zero-time engines back on the shelf, with a fresh life profile
        ret = (e_loc == 2) & (e_ready <= t)
        newp = torch.randint(lib.P, (R, E), device=dev)
        e_prof = torch.where(ret, newp, e_prof)
        e_life = torch.where(ret, lib.life[newp], e_life)
        e_age = torch.where(ret, torch.zeros_like(e_age), e_age)
        e_loc = torch.where(ret, torch.zeros_like(e_loc), e_loc)

        # 2. aircraft awaiting an engine get one if the shelf has it
        ok, eng = allocate(a_eng < 0)
        install(ok, eng, a_swap, t)

        # 3. status accounting
        has = a_eng >= 0
        down = a_down > t
        up = has & ~down
        if meas:
            acc["mc"] += up.sum(1)
            acc["await_eng"] += (~has).sum(1)
            acc["nmcm_eng"] += (has & down & (a_kind == 1)).sum(1)
            acc["other_m"] += (has & down & (a_kind == 2)).sum(1)
            acc["other_s"] += (has & down & (a_kind == 3)).sum(1)
            acc["depot_wip"] += (e_loc == 2).sum(1)

        # 4. fly
        if flying_day:
            fly = up & (torch.rand((R, A), device=dev) < cfg.p_fly)
        else:
            fly = torch.zeros((R, A), dtype=torch.bool, device=dev)
        eidx = a_eng.clamp(min=0)
        e_age.scatter_add_(1, eidx, fly.long())
        age = e_age.gather(1, eidx)
        life = e_life.gather(1, eidx)
        prof = e_prof.gather(1, eidx)
        fail = fly & (age >= life)

        none = torch.zeros_like(fail)
        if cfg.policy == "hard_time":
            forced, opp = up & ~fail & (age >= cfg.hard_time), none
        elif cfg.policy == "predictive":
            p = lib.pred[prof * lib.L + age.clamp(max=lib.L - 1)]
            forced = up & ~fail & (p <= cfg.threshold)
            opp = up & ~fail & ~forced & (p <= cfg.threshold + cfg.opportunistic)
        else:
            forced, opp = none, none

        # 5. unscheduled removals (engine failed on this sortie)
        induct(fail, eidx, cfg.tat_unsched, t, meas)
        a_eng = torch.where(fail, torch.full_like(a_eng, -1), a_eng)
        a_swap = torch.where(fail, torch.full_like(a_swap, cfg.swap_unsched), a_swap)
        ok, eng = allocate(a_eng < 0)
        install(ok, eng, a_swap, t)

        # 6. planned removals: forced ones always happen (aircraft waits if the shelf is empty);
        #    opportunistic ones only take a spare that is already on the shelf.
        swap_p = torch.full_like(a_swap, cfg.swap_planned)
        ok_f, eng_f = allocate(forced)
        induct(forced, eidx, cfg.tat_planned, t, meas)
        install(ok_f, eng_f, swap_p, t)
        wait = forced & ~ok_f
        a_eng = torch.where(wait, torch.full_like(a_eng, -1), a_eng)
        a_swap = torch.where(wait, swap_p, a_swap)
        ok_o, eng_o = allocate(opp)
        induct(ok_o, eidx, cfg.tat_planned, t, meas)
        install(ok_o, eng_o, swap_p, t)
        ok_p = forced | ok_o

        # 7. non-engine discrepancies
        oth = fly & ~fail & ~ok_p & (torch.rand((R, A), device=dev) < cfg.q_other)
        is_s = torch.rand((R, A), device=dev) < cfg.other_nmcs_share
        dur = torch.where(is_s, _lognormal((R, A), cfg.other_nmcs_days, 0.8, dev),
                          -torch.log(torch.rand((R, A), device=dev).clamp(min=1e-9)) * cfg.other_nmcm_days)
        a_down = torch.where(oth, torch.maximum(a_down, t + 1 + torch.ceil(dur)), a_down)
        a_kind = torch.where(oth, torch.where(is_s, 3, 2), a_kind)

        if meas:
            acc["sorties"] += fly.sum(1)
            acc["unsched"] += fail.sum(1)
            acc["planned"] += ok_p.sum(1)
            lf = life.clamp(min=1).float()
            acc["life_used"] += torch.where(fail, 1.0, 0.0).sum(1) + torch.where(ok_p, age / lf, zero).sum(1)
            acc["wasted"] += torch.where(ok_p, (life - age).float(), zero).sum(1)
            if trace:
                code = torch.full((A,), 1, dtype=i64)
                h0, d0, k0 = has[0].cpu(), down[0].cpu(), a_kind[0].cpu()
                code[~h0] = 3
                code[h0 & d0 & (k0 == 1)] = 2
                code[h0 & d0 & (k0 == 2)] = 4
                code[h0 & d0 & (k0 == 3)] = 5
                code[fly[0].cpu()] = 0
                trace_rows.append("".join(STATUS[int(c)] for c in code))
                # end-of-day engine state on each tail: true cycles left and the model's estimate
                cur = a_eng[0].clamp(min=0)
                ag, lf, pr = e_age[0, cur], e_life[0, cur], e_prof[0, cur]
                est = lib.pred[pr * lib.L + ag.clamp(max=lib.L - 1)]
                ok_eng = (a_eng[0] >= 0).cpu()
                trace_left.append([int(v) if o else -1 for v, o in zip((lf - ag).cpu().tolist(), ok_eng)])
                trace_est.append([int(round(v)) if o else -1 for v, o in zip(est.cpu().tolist(), ok_eng)])
                for j in torch.nonzero(fail[0]).flatten().tolist():
                    events.append({"day": t - cfg.warmup, "tail": j, "type": "failure"})
                for j in torch.nonzero(ok_p[0]).flatten().tolist():
                    events.append({"day": t - cfg.warmup, "tail": j, "type": "planned",
                                   "left": int((life[0, j] - age[0, j]).item())})

    if device.type == "mps":
        torch.mps.synchronize()
    ad = A * cfg.days
    yr = 365.0 / cfg.days
    out = {k: v.float().cpu().numpy() for k, v in acc.items()}
    removals = out["unsched"] + out["planned"]
    res = {
        "mc": out["mc"] / ad,
        "await_eng": out["await_eng"] / ad,
        "nmcm_eng": out["nmcm_eng"] / ad,
        "other_m": out["other_m"] / ad,
        "other_s": out["other_s"] / ad,
        "sorties_per_ac_yr": out["sorties"] / A * yr,
        "unsched_yr": out["unsched"] * yr,
        "planned_yr": out["planned"] * yr,
        "life_used": np.where(removals > 0, out["life_used"] / np.maximum(removals, 1), np.nan),
        "wasted_per_planned": np.where(out["planned"] > 0, out["wasted"] / np.maximum(out["planned"], 1), 0.0),
        "queue_wait": np.where(out["inducted"] > 0, out["queue_wait"] / np.maximum(out["inducted"], 1), 0.0),
        "depot_wip": out["depot_wip"] / cfg.days,
    }
    res["cost_yr"] = (cfg.cost_unsched * res["unsched_yr"] + cfg.cost_planned * res["planned_yr"]
                      + cfg.cost_spare_year * cfg.n_spares + cfg.cost_phm_year + cfg.cost_program_year)
    res["cost_per_sortie_k"] = 1000 * res["cost_yr"] / np.maximum(res["sorties_per_ac_yr"] * A, 1)
    if trace:
        res["trace"] = trace_rows
        res["left"] = trace_left
        res["est"] = trace_est
        res["events"] = events
    return res


def summarize(res: dict, target: float = 0.60) -> dict:
    out = {}
    for k, v in res.items():
        if k in ("trace", "events", "left", "est"):
            continue
        v = np.asarray(v, float)
        out[k] = {"mean": float(np.nanmean(v)), "p05": float(np.nanquantile(v, 0.05)),
                  "p50": float(np.nanquantile(v, 0.5)), "p95": float(np.nanquantile(v, 0.95)),
                  "se": float(np.nanstd(v) / math.sqrt(max(np.sum(~np.isnan(v)), 1)))}
    out["p_meet_target"] = float(np.mean(np.asarray(res["mc"]) >= target))
    return out
