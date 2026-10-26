"""Stage 10b - trace one representative simulated year, day by day, for the website's squadron board.

Runs the same squadron under run-to-failure and predictive removal with the same random seed and
picks a seed whose year is typical for both policies (MC rate close to each policy's mean), so the
board shows an ordinary year rather than a lucky one.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from common import PROCESSED, SITE_DATA, export
from sim_engine import Library, SimConfig, simulate

CPU = torch.device("cpu")


def main() -> None:
    sim = json.loads((SITE_DATA / "simulation.json").read_text())
    base = SimConfig(**sim["base"])
    th = sim["policies"]["thresholds"]["ens"]
    means = {p["policy"]: p["mc"]["mean"] for p in sim["policies"]["policies"]
             if p["name"] in ("Run to failure", "Predictive · ensemble")}
    lib = Library(PROCESSED / "sim_profiles.npz", "ens", 1.0, CPU)
    cfgs = {"reactive": base.with_(policy="reactive"), "predictive": base.with_(policy="predictive", threshold=th)}

    best, best_gap = None, 1e9
    for seed in range(2000, 2060):
        gap = sum(abs(float(simulate(c, 1, lib, CPU, seed=seed)["mc"][0]) - means[k]) for k, c in cfgs.items())
        if gap < best_gap:
            best, best_gap = seed, gap
    out = {"seed": best, "threshold": th, "days": base.days, "aircraft": base.n_aircraft, "spares": base.n_spares}
    for name, cfg in cfgs.items():
        r = simulate(cfg, 1, lib, CPU, seed=best, trace=True)
        out[name] = {"trace": r["trace"], "left": r["left"], "est": r["est"], "events": r["events"],
                     "mc": float(r["mc"][0]), "unsched": float(r["unsched_yr"][0]), "planned": float(r["planned_yr"][0]),
                     "sorties": float(r["sorties_per_ac_yr"][0] * base.n_aircraft), "mean_mc": means[name]}
        print(f"  {name}: MC {out[name]['mc']:.3f} (policy mean {means[name]:.3f}), unscheduled {out[name]['unsched']:.0f}")
    export("board", out)


if __name__ == "__main__":
    main()
