/* Ready Line - squadron readiness simulator (JavaScript port of pipeline/sim_engine.py).
 * Same model, same ground rules; runs a few hundred replications in the browser.
 * Validated against the Python/GPU reference by pipeline/validate_js_sim.js. */
(function (root) {
  "use strict";
  const CAP = 125;
  const TAIL = 100;

  const DEFAULTS = {
    n_aircraft: 24, n_spares: 3, days: 365, warmup: 365, p_fly: 0.75, flying_days: 5,
    policy: "reactive", hard_time: 140, threshold: 10, opportunistic: 15, model: "ens", alpha: 1.0,
    swap_unsched: 4.0, swap_planned: 2.0, tat_unsched: 120.0, tat_planned: 75.0, tat_cv: 0.30,
    depot_lines: 6, q_other: 0.07, other_nmcm_days: 2.0, other_nmcs_share: 0.25, other_nmcs_days: 10.0,
    cost_unsched: 3.6, cost_planned: 1.2, cost_spare_year: 1.5, cost_phm_year: 0.0,
  };

  // mulberry32: small, fast, seedable PRNG
  function rng(seed) {
    let a = seed >>> 0;
    const next = function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    next.normal = function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0, v = 0;
      while (u === 0) u = next();
      v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return next;
  }

  function decode(b64) {
    if (typeof atob === "function") {
      const s = atob(b64);
      const out = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }

  /* profiles: the exported site/data/profiles.json payload */
  function makeLibrary(profiles, model, alpha) {
    const tail = decode(profiles.models[model]);
    const life = Int32Array.from(profiles.life);
    const P = life.length;
    let total = 0;
    const cum = new Float64Array(P);
    for (let p = 0; p < P; p++) { total += life[p]; cum[p] = total; }
    return {
      P, life, alpha,
      pred(p, age) {
        const L = life[p];
        const truth = Math.min(Math.max(L - age, 0), CAP);
        const start = L - TAIL + 1;
        const raw = age >= start && age <= L ? tail[p * TAIL + (age - start)] : CAP;
        return truth + alpha * (raw - truth);
      },
      // length-biased draw (renewal-theory steady state for engines already in service)
      drawBiased(u) {
        const x = u * total;
        let lo = 0, hi = P - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < x) lo = mid + 1; else hi = mid; }
        return lo;
      },
    };
  }

  function lognormal(r, mean, cv) {
    const s2 = Math.log(1 + cv * cv);
    return Math.exp(Math.log(mean) - s2 / 2 + Math.sqrt(s2) * r.normal());
  }

  function simulate(cfgIn, R, lib, seed) {
    const cfg = Object.assign({}, DEFAULTS, cfgIn);
    const A = cfg.n_aircraft, S = cfg.n_spares, E = A + S, D = cfg.warmup + cfg.days;
    const r = rng(seed || 1);
    const out = {
      mc: new Float64Array(R), await_eng: new Float64Array(R), nmcm_eng: new Float64Array(R),
      other_m: new Float64Array(R), other_s: new Float64Array(R), sorties_per_ac_yr: new Float64Array(R),
      unsched_yr: new Float64Array(R), planned_yr: new Float64Array(R), cost_yr: new Float64Array(R),
      life_used: new Float64Array(R),
    };
    const eProf = new Int32Array(E), eLife = new Int32Array(E), eAge = new Int32Array(E);
    const eLoc = new Int8Array(E), eReady = new Float64Array(E);
    const aEng = new Int32Array(A), aDown = new Float64Array(A), aKind = new Int8Array(A), aSwap = new Float64Array(A);
    const lines = new Float64Array(cfg.depot_lines);
    const up = new Uint8Array(A), fly = new Uint8Array(A), fail = new Uint8Array(A);
    const forced = new Uint8Array(A), opp = new Uint8Array(A), planned = new Uint8Array(A);
    const pred = cfg.policy === "predictive";
    const hard = cfg.policy === "hard_time";

    for (let rep = 0; rep < R; rep++) {
      for (let e = 0; e < E; e++) {
        const p = e < A ? lib.drawBiased(r()) : Math.floor(r() * lib.P);
        eProf[e] = p; eLife[e] = lib.life[p];
        eAge[e] = e < A ? Math.floor(r() * eLife[e]) : 0;
        eLoc[e] = e < A ? 1 : 0; eReady[e] = 0;
      }
      for (let j = 0; j < A; j++) { aEng[j] = j; aDown[j] = 0; aKind[j] = 0; aSwap[j] = cfg.swap_unsched; }
      lines.fill(0);
      let mc = 0, awaitE = 0, nmcmE = 0, othM = 0, othS = 0, sorties = 0, unsched = 0, plannedN = 0, lifeUsed = 0;

      const takeSpare = () => { for (let e = 0; e < E; e++) if (eLoc[e] === 0) return e; return -1; };
      const install = (j, e, t, days) => { eLoc[e] = 1; aEng[j] = e; aDown[j] = t + 1 + days; aKind[j] = 1; };
      const induct = (e, t, tatMean) => {
        let li = 0;
        for (let k = 1; k < lines.length; k++) if (lines[k] < lines[li]) li = k;
        const start = Math.max(lines[li], t);
        const done = start + lognormal(r, tatMean, cfg.tat_cv);
        lines[li] = done; eReady[e] = done; eLoc[e] = 2;
      };

      for (let t = 0; t < D; t++) {
        const meas = t >= cfg.warmup;
        const flyingDay = (t % 7) < cfg.flying_days;
        // 1. depot returns
        for (let e = 0; e < E; e++) {
          if (eLoc[e] === 2 && eReady[e] <= t) {
            const p = Math.floor(r() * lib.P);
            eProf[e] = p; eLife[e] = lib.life[p]; eAge[e] = 0; eLoc[e] = 0;
          }
        }
        // 2. awaiting aircraft take shelf spares, lowest tail first
        for (let j = 0; j < A; j++) if (aEng[j] < 0) { const e = takeSpare(); if (e < 0) break; install(j, e, t, aSwap[j]); }
        // 3. status
        for (let j = 0; j < A; j++) {
          const has = aEng[j] >= 0, down = aDown[j] > t;
          up[j] = has && !down ? 1 : 0;
          if (meas) {
            if (up[j]) mc++;
            else if (!has) awaitE++;
            else if (aKind[j] === 1) nmcmE++;
            else if (aKind[j] === 2) othM++;
            else if (aKind[j] === 3) othS++;
          }
        }
        // 4. fly, age engines, detect failures and removal triggers
        for (let j = 0; j < A; j++) {
          fly[j] = flyingDay && up[j] && r() < cfg.p_fly ? 1 : 0;
          fail[j] = 0; forced[j] = 0; opp[j] = 0; planned[j] = 0;
          if (!up[j]) continue;
          const e = aEng[j];
          if (fly[j]) { eAge[e] += 1; if (meas) sorties++; }
          if (fly[j] && eAge[e] >= eLife[e]) { fail[j] = 1; continue; }
          if (hard) forced[j] = eAge[e] >= cfg.hard_time ? 1 : 0;
          else if (pred) {
            const est = lib.pred(eProf[e], eAge[e]);
            if (est <= cfg.threshold) forced[j] = 1;
            else if (est <= cfg.threshold + cfg.opportunistic) opp[j] = 1;
          }
        }
        // 5. unscheduled removals
        for (let j = 0; j < A; j++) {
          if (!fail[j]) continue;
          const e = aEng[j];
          induct(e, t, cfg.tat_unsched);
          aEng[j] = -1; aSwap[j] = cfg.swap_unsched;
          if (meas) { unsched++; lifeUsed += 1; }
        }
        for (let j = 0; j < A; j++) if (aEng[j] < 0) { const e = takeSpare(); if (e < 0) break; install(j, e, t, aSwap[j]); }
        // 6. planned removals: forced always, opportunistic only onto a shelf spare
        for (let j = 0; j < A; j++) {
          if (!forced[j]) continue;
          const old = aEng[j];
          if (meas) { plannedN++; lifeUsed += eAge[old] / Math.max(eLife[old], 1); }
          induct(old, t, cfg.tat_planned);
          planned[j] = 1;
          const e = takeSpare();
          if (e >= 0) install(j, e, t, cfg.swap_planned);
          else { aEng[j] = -1; aSwap[j] = cfg.swap_planned; }
        }
        for (let j = 0; j < A; j++) {
          if (!opp[j]) continue;
          const e = takeSpare();
          if (e < 0) break;
          const old = aEng[j];
          if (meas) { plannedN++; lifeUsed += eAge[old] / Math.max(eLife[old], 1); }
          induct(old, t, cfg.tat_planned);
          install(j, e, t, cfg.swap_planned);
          planned[j] = 1;
        }
        // 7. non-engine discrepancies
        for (let j = 0; j < A; j++) {
          if (!fly[j] || fail[j] || planned[j]) continue;
          if (r() < cfg.q_other) {
            const isS = r() < cfg.other_nmcs_share;
            const dur = isS ? lognormal(r, cfg.other_nmcs_days, 0.8) : -Math.log(Math.max(r(), 1e-9)) * cfg.other_nmcm_days;
            aDown[j] = Math.max(aDown[j], t + 1 + Math.ceil(dur));
            aKind[j] = isS ? 3 : 2;
          }
        }
      }
      const ad = A * cfg.days, yr = 365 / cfg.days;
      out.mc[rep] = mc / ad; out.await_eng[rep] = awaitE / ad; out.nmcm_eng[rep] = nmcmE / ad;
      out.other_m[rep] = othM / ad; out.other_s[rep] = othS / ad;
      out.sorties_per_ac_yr[rep] = sorties / A * yr;
      out.unsched_yr[rep] = unsched * yr; out.planned_yr[rep] = plannedN * yr;
      out.life_used[rep] = unsched + plannedN > 0 ? lifeUsed / (unsched + plannedN) : NaN;
      out.cost_yr[rep] = cfg.cost_unsched * out.unsched_yr[rep] + cfg.cost_planned * out.planned_yr[rep]
        + cfg.cost_spare_year * S + cfg.cost_phm_year;
    }
    return out;
  }

  function mean(a) { let s = 0, n = 0; for (const v of a) if (!Number.isNaN(v)) { s += v; n++; } return n ? s / n : NaN; }
  function quantile(a, q) {
    const s = Array.from(a).filter((v) => !Number.isNaN(v)).sort((x, y) => x - y);
    if (!s.length) return NaN;
    const i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
    return s[lo] + (s[hi] - s[lo]) * (i - lo);
  }

  const api = { DEFAULTS, makeLibrary, simulate, mean, quantile, rng };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RLSim = api;
})(typeof window !== "undefined" ? window : globalThis);
