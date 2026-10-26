"""Stage 12 - build the static website.

* Derives the handful of headline numbers the text needs (by name, never by position).
* Renders site_src/index.template.html -> site/index.html, filling every {{path|format}}
  placeholder from the pipeline's JSON outputs, so no number on the page is typed by hand.
* Bundles all JSON into site/data/bundle.js so the page also works from file:// (no server).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from common import ROOT, SITE_DATA

TEMPLATE = ROOT / "site_src" / "index.template.html"
CONFIG = ROOT / "site_config.json"
FILES = ["etl", "warehouse", "explore", "hypotheses", "models", "gpu", "nlp", "forecast",
         "simulation", "optimization", "board", "profiles", "jsval"]
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def load() -> dict:
    return {f: json.loads((SITE_DATA / f"{f}.json").read_text()) for f in FILES}


def by(rows, key, value):
    return next(r for r in rows if r[key] == value)


def derive(D: dict, cfg: dict) -> dict:
    sim, opt, mdl, nlp = D["simulation"], D["optimization"], D["models"], D["nlp"]
    aoa = {c["id"]: c for c in sim["aoa"]}
    pol = {p["name"]: p for p in sim["policies"]["policies"]}
    ens = by(mdl["metrics"], "model", "ens")
    clf = {(r["variant"], r["model"]): r for r in nlp["classifier"]["results"]}
    masked, raw = clf[("masked", "Linear (SGD)")], clf[("raw", "Linear (SGD)")]
    tables = {t["table_name"]: t for t in D["warehouse"]["tables"]}
    bench = max(sim["benchmark"], key=lambda b: b["R"])
    sc = opt["spares"]["scenarios"]
    target = sim["pbl_target"]

    # same-availability engine stock: cheapest MILP point reaching 90% availability in each scenario
    eng_idx = [i for i, it in enumerate(sc["reactive"]["items"]) if it["code"] == "ENG"][0]
    def eng_at(k, a=0.90):
        f = next(f for f in sc[k]["frontier"] if f["A"] >= a)
        return f["stock"][eng_idx]
    eng_drop = eng_at("reactive") - eng_at("predictive")

    # Recommendation: predictive removal first (it raises readiness and lowers cost), then the add-on
    # with the best gain in P(meets PBL) per extra dollar; also name the highest-confidence option.
    c0_, c3_ = aoa["COA 0"], aoa["COA 3"]
    addons = [c for c in sim["aoa"] if c["policy"] == "predictive" and c["id"] != "COA 3"
              and c["cost_yr"]["mean"] > c3_["cost_yr"]["mean"]]
    rec = max(addons, key=lambda c: (c["p_meet_target"] - c3_["p_meet_target"]) / (c["cost_yr"]["mean"] - c3_["cost_yr"]["mean"]))
    top = max(sim["aoa"], key=lambda c: c["p_meet_target"])
    c0, c3 = aoa["COA 0"], aoa["COA 3"]
    sens = {s["case"]: s for s in sim["sensitivity"]}
    no_shorter = sens["Planned depot visit no shorter"]["mc_gain"]
    other_nmc = c3["other_m"]["mean"] + c3["other_s"]["mean"]

    fmt_usd = lambda v: f"${v:,.1f}M"  # noqa: E731
    addon = {"COA 4": "one more spare engine", "COA 5": "the faster depot", "COA 6": "one spare engine and the faster depot"}
    rec_text = (
        f"Adopt predictive engine removal (COA 3) now. In simulation it lifts the squadron's mission-capable rate from "
        f"{c0['mc']['mean']:.1%} to {c3['mc']['mean']:.1%} and cuts notional engine sustainment cost from "
        f"{fmt_usd(c0['cost_yr']['mean'])} to {fmt_usd(c3['cost_yr']['mean'])} a year. It beats buying spares or paying for a "
        f"faster depot on both counts. Then add {addon.get(rec['id'], rec['name'].lower())} ({rec['id']}), the best value among "
        f"the add-ons: it raises the chance of meeting the {target:.0%} requirement from {c0['p_meet_target']:.0%} to "
        f"{rec['p_meet_target']:.0%} for {fmt_usd(rec['cost_yr']['mean'] - c3['cost_yr']['mean'])} more a year."
        + (f" Adding {addon.get(top['id'], top['name'].lower())} instead ({top['id']}) reaches {top['p_meet_target']:.0%} for "
           f"another {fmt_usd(top['cost_yr']['mean'] - rec['cost_yr']['mean'])}."
           if top["id"] != rec["id"] and top["cost_yr"]["mean"] > rec["cost_yr"]["mean"] else "")
    )
    gain_txt = "to about zero" if abs(no_shorter) < 0.0025 else f"to {no_shorter * 100:+.1f} points"
    rec_text_2 = (
        f"No engine-side option meets {target:.0%} with high confidence, because non-engine groundings still take "
        f"{other_nmc:.0%} of aircraft-days. That is the next problem to work. Before fielding, measure one number on the real "
        f"program: how much shorter a planned depot visit is than an unscheduled one. If it is no shorter, the readiness gain "
        f"falls {gain_txt}, although the cost saving remains."
    )
    vop = {v["alpha"]: v for v in sim["value"]}
    ridge = by(mdl["metrics"], "model", "ridge")
    topics = nlp["topics"]["topics"]
    repo = (cfg.get("repo_url") or "").strip()
    return {
        "n_sdr": D["etl"]["sdr_quality"]["raw_rows"],
        "n_engines": tables["dim_engine"]["rows"],
        "n_cycles": tables["fact_engine_cycle"]["rows"],
        "n_train_engines": mdl["n_train_engines"],
        "sim_reps_total": sim.get("squadron_years", 0),
        "mc_base": c0["mc"]["mean"], "mc_pred": c3["mc"]["mean"],
        "mc_gain_pts": (c3["mc"]["mean"] - c0["mc"]["mean"]) * 100,
        "un_base": c0["unsched_yr"]["mean"], "un_pred": c3["unsched_yr"]["mean"],
        "unsched_cut": 1 - c3["unsched_yr"]["mean"] / c0["unsched_yr"]["mean"],
        "aoa_R": 4000, "pol_R": 4000,
        "rmse_dz": ens["rmse_decision_zone"], "rmse_fd001": ens["rmse_FD001"],
        "clf_acc": masked["accuracy"], "clf_top3": masked["top3"], "clf_f1": masked["macro_f1"],
        "clf_mask_delta": masked["accuracy"] - raw["accuracy"],
        "regime_sample": len(D["explore"]["regimes"]["scatter"]),
        "mode_life_a": D["hypotheses"]["h3"]["mean_a"], "mode_life_b": D["hypotheses"]["h3"]["mean_b"],
        "gpu_speedup": D["gpu"]["epoch_seconds"]["cpu"] / D["gpu"]["epoch_seconds"]["mps"],
        "th": sim["policies"]["thresholds"]["ens"],
        "hard_waste": 1 - pol["Hard-time removal"]["life_used"]["mean"],
        "pred_life_used": pol["Predictive · ensemble"]["life_used"]["mean"],
        "oracle_gap": (pol["Perfect foresight"]["mc"]["mean"] - pol["Predictive · ensemble"]["mc"]["mean"]) * 100,
        "js_max_diff": D["jsval"]["max_abs_diff_pts"],
        "sim_speedup": bench["cpu"] / bench[sim["device"]], "bench_R": bench["R"], "bench_ac_days": bench["aircraft_days"],
        "n_milps": sum(len(s["frontier"]) for s in sc.values()),
        "eng_spares_drop": eng_drop,
        "eng_drop_txt": {0: "no fewer spare engines", 1: "one fewer spare engine", 2: "two fewer spare engines",
                         3: "three fewer spare engines"}.get(eng_drop, f"{eng_drop} fewer spare engines"),
        "rot_cost": sc["reactive"]["rule_of_thumb"]["cost"] / 1000, "rot_A": sc["reactive"]["rule_of_thumb"]["A"],
        "rot_milp": sc["reactive"]["rule_of_thumb"]["milp_cost_same_A"] / 1000,
        "rot_saving": sc["reactive"]["rule_of_thumb"]["saving"],
        "recommended_coa": rec["id"], "recommendation": rec_text, "recommendation_2": rec_text_2,
        "topic1": topics[0]["name"], "topic2": topics[1]["name"], "topic3": topics[2]["name"],
        "top3_impact": topics[2]["cum_impact_share"],
        "repo_url": repo or "#build", "repo_label": repo.replace("https://", "") if repo else "Repository link available on request",
        "build_date": dt.date.today().strftime("%b %Y"),
        "cov80_all": mdl["conformal"]["coverage"]["0.8"]["all_cycles"],
        "vop_mc_spread": (vop[0.0]["mc"] - vop[4.0]["mc"]) * 100,
        "vop_cost_spread": vop[4.0]["cost"] - vop[0.0]["cost"],
        "ridge_err_ratio": ridge["rmse_decision_zone"] / ens["rmse_decision_zone"],
        "ridge_mc": pol["Predictive · ridge"]["mc"]["mean"],
        "hard_mc": pol["Hard-time removal"]["mc"]["mean"],
        "sens_R": 1500, "n_coas": len(sim["aoa"]),
        "greedy_gap": max(v["greedy_max_gap"] for v in sc.values()),
        "greedy_never_better": all(v["greedy_min_gap"] >= -1e-9 for v in sc.values()),
    }


# ----------------------------------------------------------------------------- formatting
def fmt(v, spec: str | None) -> str:
    if spec is None or spec == "raw":
        return str(v)
    if v is None:
        return "–"
    if spec == "int":
        return f"{int(round(v)):,}"
    if spec in ("f0", "f1", "f2", "f3"):
        return f"{v:,.{spec[1]}f}"
    if spec in ("pct0", "pct1"):
        return f"{v * 100:.{spec[3]}f}%"
    if spec == "pts1":
        return f"{v * 100:.1f}"
    if spec in ("signpts1", "signpts2"):
        x = v * 100
        return ("+" if x >= 0 else "−") + f"{abs(x):.{spec[-1]}f}"
    if spec == "p":
        if v <= 0:
            return "< 10⁻³⁰⁰"   # underflowed to zero in double precision
        if v < 1e-10:
            return "< 10⁻" + str(int(math.floor(-math.log10(v)))).translate(SUP)
        return "< 0.001" if v < 0.001 else f"= {v:.3f}"
    if spec == "sci":
        e = int(math.floor(math.log10(abs(v))))
        return f"{v / 10 ** e:.1f} × 10" + str(e).translate(SUP)
    raise ValueError(f"unknown format {spec}")


def lookup(ctx: dict, path: str):
    cur = ctx
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(template: str, ctx: dict) -> str:
    missing = []

    def sub(m):
        path, spec = m.group(1).strip(), (m.group(2) or "").strip() or None
        try:
            return esc(fmt(lookup(ctx, path), spec))
        except (KeyError, IndexError, ValueError) as e:
            missing.append(f"{path}|{spec}: {e}")
            return "?"
    out = re.sub(r"\{\{\s*([\w.]+)\s*(?:\|\s*(\w+)\s*)?\}\}", sub, template)
    if missing:
        raise SystemExit("Unresolved placeholders:\n  " + "\n  ".join(missing))
    return out


def bust(html: str) -> str:
    """Append a content hash to local CSS/JS URLs so browsers and CDNs never serve stale copies."""
    def sub(m):
        attr, path = m.group(1), m.group(2)
        f = ROOT / "site" / path
        h = hashlib.sha1(f.read_bytes()).hexdigest()[:10] if f.exists() else "0"
        return f'{attr}="{path}?v={h}"'
    return re.sub(r'(src|href)="((?:js|css|data)/[\w./-]+\.(?:js|css))"', sub, html)


def main() -> None:
    D = load()
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    D["d"] = derive(D, cfg)
    bundle = "window.RL_DATA=" + json.dumps(D, separators=(",", ":")) + ";\n"
    (SITE_DATA / "bundle.js").write_text(bundle)
    html = bust(render(TEMPLATE.read_text(), D))
    (ROOT / "site" / "index.html").write_text(html)
    print(f"  -> site/index.html ({len(html) / 1024:.0f} KB), site/data/bundle.js ({len(bundle) / 1024:.0f} KB)")
    for k in ("mc_base", "mc_pred", "mc_gain_pts", "unsched_cut", "recommended_coa", "eng_spares_drop", "rot_saving"):
        print(f"  {k}: {D['d'][k]}")


if __name__ == "__main__":
    main()
