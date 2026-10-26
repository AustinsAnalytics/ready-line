// Stage 12a - check the browser simulator (site/js/sim.js) against the Python/GPU reference runs
// exported by stage 10. Run with: node pipeline/validate_js_sim.js
"use strict";
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const RLSim = require(path.join(root, "site/js/sim.js"));
const profiles = JSON.parse(fs.readFileSync(path.join(root, "site/data/profiles.json"), "utf8"));
const sim = JSON.parse(fs.readFileSync(path.join(root, "site/data/simulation.json"), "utf8"));

const R = 3000;
const rows = [];
for (const ref of sim.reference) {
  const cfg = ref.config;
  const lib = RLSim.makeLibrary(profiles, cfg.model, cfg.alpha);
  const t0 = Date.now();
  const res = RLSim.simulate(cfg, R, lib, 4242);
  const ms = Date.now() - t0;
  const mc = RLSim.mean(res.mc);
  // standard error of the difference between two independent Monte Carlo means
  const sd = Math.sqrt(Array.from(res.mc).reduce((s, v) => s + (v - mc) ** 2, 0) / (R - 1));
  const se = Math.sqrt(sd * sd / R + ref.mc_sd * ref.mc_sd / R);
  const row = {
    policy: cfg.policy, spares: cfg.n_spares, tat: cfg.tat_unsched,
    py_mc: ref.mc, js_mc: mc, diff_pts: (mc - ref.mc) * 100, z: (mc - ref.mc) / se,
    py_unsched: ref.unsched, js_unsched: RLSim.mean(res.unsched_yr),
    py_planned: ref.planned, js_planned: RLSim.mean(res.planned_yr), ms,
  };
  rows.push(row);
  console.log(`${cfg.policy.padEnd(11)} spares=${cfg.n_spares} tat=${cfg.tat_unsched}: ` +
    `MC py ${(ref.mc * 100).toFixed(2)} js ${(mc * 100).toFixed(2)} (Δ ${row.diff_pts.toFixed(2)} pts, z=${row.z.toFixed(1)}), ` +
    `unsched py ${ref.unsched.toFixed(2)} js ${row.js_unsched.toFixed(2)}, ${ms} ms`);
}
const maxDiff = Math.max(...rows.map((r) => Math.abs(r.diff_pts)));
fs.writeFileSync(path.join(root, "site/data/jsval.json"), JSON.stringify({ R, rows, max_abs_diff_pts: maxDiff }));
console.log(`max |Δ| = ${maxDiff.toFixed(2)} MC points`);
process.exit(maxDiff < 1.0 ? 0 : 1);
