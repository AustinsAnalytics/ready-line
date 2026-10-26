/* Ready Line - renders every figure from window.RL_DATA (written by pipeline/12_build_site.py). */
(function () {
  "use strict";
  const DATA = window.RL_DATA;
  const RL = window.RL;
  const F = RL.fmt;
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const S = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)", "var(--s7)", "var(--s8)"];
  const ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"];   // ordinal blue ramp (regimes)
  // Sequential encoding as one hue at rising opacity over the surface: low values recede toward the
  // background in either theme, so light and dark modes both read "darker/stronger = more".
  const t01 = (v, lo, hi) => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const seqOpacity = (v, lo = 0, hi = 1) => 0.06 + 0.94 * t01(v, lo, hi);
  const inkFor = (v, lo = 0, hi = 1) => (seqOpacity(v, lo, hi) > 0.6 ? "#ffffff" : "var(--ink)");

  // ---------------------------------------------------------------- theme toggle + rail
  const themeBtn = $("#theme-toggle");
  themeBtn.addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("rl-theme", next); } catch (e) { /* storage unavailable */ }
  });
  const railLinks = [...document.querySelectorAll("#rail a")];
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const en of entries) if (en.isIntersecting) {
        railLinks.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + en.target.id));
      }
    }, { rootMargin: "-30% 0px -60% 0px" });
    document.querySelectorAll("section.sec").forEach((s) => io.observe(s));
  }

  // ================================================================ 03 DISCOVER
  (function discover() {
    const X = DATA.explore;
    const regs = X.regimes.regimes;
    const totalRows = d3.sum(regs, (r) => r.rows);
    const rr = d3.scaleSqrt().domain([0, d3.max(regs, (r) => r.rows)]).range([0, 26]);
    const off = { 0: [12, -10, "start"], 1: [14, 4, "start"], 2: [16, 14, "start"], 3: [-16, -8, "end"], 4: [-18, 4, "end"], 5: [-18, 6, "end"] };
    RL.scatter($("#c-regimes"), {
      height: 280, r: 0, ring: false,
      data: regs.map((r) => ({ x: r.mach, y: r.alt, r })),
      xLabel: "Mach number", yLabel: "Altitude (kft)", xDomain: [-0.14, 1.0], yDomain: [-8, 48],
      xTickValues: [0, 0.2, 0.4, 0.6, 0.8], yTickValues: [0, 10, 20, 30, 40],
      tip: (d) => ({ title: "Regime R" + d.r.regime, rows: [{ v: `${F.f0(d.r.alt)} kft · Mach ${d.r.mach.toFixed(2)} · throttle ${F.f0(d.r.tra)}` }, { v: F.pct0(d.r.rows / totalRows), k: "of training cycles" }] }),
      before(g, x, y) {
        for (const r of regs) {
          g.append("circle").attr("cx", x(r.mach)).attr("cy", y(r.alt)).attr("r", rr(r.rows)).style("fill", ORD[r.regime]).style("opacity", 0.85)
            .style("stroke", "var(--surface)").style("stroke-width", 2);
          const [dx, dy, a] = off[r.regime];
          g.append("text").attr("class", "dlabel").attr("x", x(r.mach) + dx + (a === "start" ? rr(r.rows) * 0.6 : -rr(r.rows) * 0.6)).attr("y", y(r.alt) + dy)
            .attr("text-anchor", a).text(`R${r.regime} · TRA ${F.f0(r.tra)}`);
        }
      },
    });

    const N = X.regimes.noise;
    let view = "raw";
    const drawNoise = () => {
      const vals = N.cycle.map((c, i) => ({ x: c, raw: N.raw[i], norm: N.norm[i], smooth: N.smooth[i], regime: N.regime[i] }));
      const host = $("#c-noise");
      host.replaceChildren();
      if (view === "raw") {
        RL.scatter(host, {
          height: 250, r: 2.6, opacity: 0.9, ring: false,
          data: vals.map((d) => ({ x: d.x, y: d.raw, color: ORD[d.regime], regime: d.regime })),
          xLabel: "Cycle", yLabel: "Ps30 (psia), raw",
          tip: (d) => ({ title: "Cycle " + d.x, rows: [{ v: F.f2(d.y) + " psia", k: "regime R" + d.regime }] }),
        });
      } else {
        const series = [{ name: "Normalized", color: "var(--neutral-mark)", values: vals.map((d) => ({ x: d.x, y: d.norm })), width: 1, opacity: view === "smooth" ? 0.5 : 1 }];
        if (view === "smooth") series.push({ name: "15-cycle mean", color: "var(--s1)", values: vals.map((d) => ({ x: d.x, y: d.smooth })), width: 2.5, endLabel: true });
        RL.lineChart(host, { height: 250, series, xLabel: "Cycle", yLabel: "Ps30, z-score within regime", tipFormat: F.f2, yDomain: [d3.min(vals, (d) => d.norm) - 0.3, d3.max(vals, (d) => d.norm) + 0.3] });
      }
    };
    RL.chips($("#noise-view"), (v) => { view = v; drawNoise(); });
    drawNoise();

    const sens = X.sensors;
    RL.hbar($("#c-sensors"), {
      data: sens.map((s) => ({ label: `${s.abbr} · ${s.name}`, value: s.score, highlight: s.informative, color: s.informative ? "var(--s1)" : "var(--neutral-mark)",
        extra: [{ k: "monotonicity", v: F.f2(s.monotonicity) }, { k: "trendability", v: F.f2(s.trendability) }, { k: "prognosability", v: F.f2(s.prognosability) }] })),
      labelW: 250, rowH: 22, valueFormat: F.f2, xDomain: [0, 0.8], valueName: "suitability score", dimOthers: true, xLabel: "Suitability score (0–1)",
    });
    RL.table("c-sensors", [{ label: "Sensor", key: "abbr" }, { label: "Description", key: "name" }, { label: "Monotonicity", num: true, value: (r) => F.f2(r.monotonicity) },
      { label: "Trendability", num: true, value: (r) => F.f2(r.trendability) }, { label: "Prognosability", num: true, value: (r) => F.f2(r.prognosability) },
      { label: "Score", num: true, value: (r) => F.f2(r.score) }], sens);

    const H = X.health;
    RL.responsive($("#c-health"), (w) => {
      const h = 300, m = { top: 20, right: 20, bottom: 40, left: 46 }, iw = w - m.left - m.right, ih = h - m.top - m.bottom;
      const svg = RL.svg($("#c-health"), w, h, "Health index curves");
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const x = d3.scaleLinear().domain([-360, 0]).range([0, iw]);
      const y = d3.scaleLinear().domain([-0.4, 1.4]).range([ih, 0]);
      RL.axes(g, x, y, iw, ih, { xLabel: "Cycles before failure", yLabel: "Health index" });
      const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveMonotoneX);
      const paths = g.append("g").selectAll("path").data(H.curves).join("path")
        .attr("d", (c) => line(c.t.map((t, i) => [t, c.hi[i]]))).style("fill", "none").style("stroke", "var(--s1)").style("stroke-width", 1).style("opacity", 0.22);
      g.append("line").attr("class", "ref-line").attr("x1", 0).attr("x2", iw).attr("y1", y(H.threshold)).attr("y2", y(H.threshold));
      g.append("text").attr("class", "ann").attr("x", 6).attr("y", y(H.threshold) + 14).text(`mean index at failure ${F.f2(H.threshold)} (sd ${F.f2(H.threshold_sd)})`);
      g.append("g").selectAll("path").data(H.curves).join("path")
        .attr("d", (c) => line(c.t.map((t, i) => [t, c.hi[i]]))).style("fill", "none").style("stroke", "transparent").style("stroke-width", 8)
        .on("pointermove", function (evt, c) {
          paths.style("opacity", (p) => (p === c ? 1 : 0.12)).style("stroke-width", (p) => (p === c ? 2 : 1));
          RL.tip.show(evt, c.id, [{ v: c.life + " cycles", k: "total life" }]);
        })
        .on("pointerleave", () => { paths.style("opacity", 0.22).style("stroke-width", 1); RL.tip.hide(); });
    });

    const FM = X.fault_modes.subsets;
    const tb = $("#fm-table");
    const docs = ["1 (HPC)", "1 (HPC)", "2 (HPC + fan)", "2 (HPC + fan)"];
    FM.forEach((s, i) => {
      const tr = tb.insertRow();
      const two = s.delta_bic_12 < -10 && s.dip_p < 0.05;
      [s.subset, docs[i], (s.delta_bic_12 > 0 ? "+" : "") + F.f1(s.delta_bic_12), F.p(s.dip_p), two ? "Two modes found" : "One mode"].forEach((v, k) => {
        const td = tr.insertCell(); td.textContent = v; if (k === 2 || k === 3) td.className = "n";
        if (k === 4) { td.replaceChildren(el("span", "pill " + (two ? "acc" : ""), v)); }
      });
    });
    let fmIdx = 2;
    const drawFM = () => {
      const s = FM[fmIdx];
      const two = s.delta_bic_12 < -10;
      const lc = s.life_by_cluster;
      $("#c-faultmodes").replaceChildren();
      RL.scatter($("#c-faultmodes"), {
        height: 300, r: 4.5,
        data: s.points.map((p) => ({ x: p.x, y: p.y, color: two ? S[p.c] : "var(--neutral-mark)", id: p.id, life: p.life, c: p.c })),
        xLabel: `PC1 (${F.pct0(s.explained[0])} of variance)`, yLabel: `PC2 (${F.pct0(s.explained[1])})`,
        tip: (d) => ({ title: d.id, rows: [{ v: d.life + " cycles", k: "life" }].concat(two ? [{ v: d.c ? "Mode B" : "Mode A", color: S[d.c] }] : []) }),
        after(g, x, y, iw) {
          const txt = two ? `Mode A: ${lc.n0} engines, mean life ${F.f0(lc.mean0)} · Mode B: ${lc.n1} engines, mean life ${F.f0(lc.mean1)}`
            : `One mode: the two-group model is rejected (ΔBIC ${"+" + F.f1(s.delta_bic_12)})`;
          g.append("text").attr("class", "ann").attr("x", iw).attr("y", -8).attr("text-anchor", "end").text(txt);
        },
      });
    };
    RL.chips($("#fm-subset"), (v) => { fmIdx = +v; drawFM(); });
    drawFM();
  })();

  // ================================================================ 04 TEST
  (function test() {
    const H = DATA.hypotheses;
    const host = $("#hyps");
    const check = '<svg viewBox="0 0 12 12"><path d="M2 6.5l2.5 2.5L10 3.5" fill="none" stroke="currentColor" stroke-width="2"/></svg>';
    const dash = '<svg viewBox="0 0 12 12"><path d="M2 6h8" stroke="currentColor" stroke-width="2"/></svg>';
    const g2 = H.h1.groups[2];
    const cards = [
      { id: "H1 · reliability", title: "Engines wear out (failure rate rises with age)", h0: "Weibull shape β = 1 (constant hazard: failures are random in time)",
        h1: "β > 1 (wear-out)", test: `Likelihood-ratio test, Weibull vs. exponential, n = ${g2.n}`,
        res: `β = ${F.f2(g2.beta)} [${F.f2(g2.beta_lo)}, ${F.f2(g2.beta_hi)}], χ²₁ = ${F.f0(g2.lr_vs_exponential)}, p < 10⁻¹⁰⁰`, reject: true,
        so: `Removing engines before failure can beat run to failure, so it's worth predicting the removal. A 3-parameter Weibull fits better (KS p = ${F.f3(H.h1.groups[0].w3.ks_p)} for the HPC-only fleet), with a failure-free period of about ${F.f0(g2.w3.gamma)} cycles.`, chart: "h1" },
      { id: "H2 · environment", title: "Flying in six regimes shortens engine life", h0: "Life distribution is the same with 1 or 6 operating conditions (FD001 vs FD002, same fault)",
        h1: "Life differs", test: `Log-rank test (also Mann–Whitney, KS), n = ${H.h2.n_a} + ${H.h2.n_b}`,
        res: `Mean ${F.f1(H.h2.mean_a)} vs ${F.f1(H.h2.mean_b)} cycles; log-rank p = ${F.f3(H.h2.logrank_p)}; difference ${F.f1(H.h2.diff)} [${F.f1(H.h2.diff_lo)}, ${F.f1(H.h2.diff_hi)}]`, reject: false,
        so: "Operating conditions change what the sensors read, but not how long the engine lasts. The fix belongs in feature engineering (regime normalization), not in the life model. One life distribution can serve the whole HPC fleet.", chart: "h2" },
      { id: "H3 · latent factor", title: "The discovered fault mode predicts life", h0: "Engines in the two unsupervised clusters (FD003 + FD004) have the same life distribution",
        h1: "Life differs by mode", test: `Log-rank test, bootstrap CI of the mean difference, n = ${H.h3.n_a} + ${H.h3.n_b}`,
        res: `Mean ${F.f0(H.h3.mean_a)} vs ${F.f0(H.h3.mean_b)} cycles; difference ${F.f0(H.h3.diff)} [${F.f0(H.h3.diff_lo)}, ${F.f0(H.h3.diff_hi)}]; log-rank p < 10⁻²⁰`, reject: true,
        so: "A label nobody provided turns out to be a strong predictor of life. Fault mode has to be part of the prognostic model. Here the engine-baseline deviation features capture it without an explicit label.", chart: "h3" },
      { id: "H4 · operations", title: "Engine problems are more likely to interrupt a flight", h0: "Among problems found in flight, engine-system reports have the same odds of a precautionary procedure as other systems",
        h1: "Odds differ", test: `Logistic regression, n = ${F.int(H.h4.n)} in-flight SDRs, controls for ${H.h4.controls.join(", ")}; robust SEs`,
        res: `Adjusted odds ratio ${F.f2(H.h4.or)} [${F.f2(H.h4.or_lo)}, ${F.f2(H.h4.or_hi)}], p < 10⁻¹⁰⁰; raw rates ${F.pct1(H.h4.engine_rate)} vs ${F.pct1(H.h4.other_rate)}`, reject: true,
        so: "An in-flight engine problem is about twice as likely to end a mission early. That is why engine prediction gets priority over other systems with more reports.", chart: "h4" },
      { id: "H5 · demand", title: "The pandemic changed maintenance-report volume", h0: "No level shift in monthly SDR volume during Apr 2020 – Mar 2021",
        h1: "Level shift", test: "Interrupted time series: log counts, trend + month effects, Newey–West SEs, 120 months",
        res: `Level shift ${F.pct1(H.h5.acute_effect)} [${F.pct1(H.h5.acute_lo)}, ${F.pct1(H.h5.acute_hi)}], ${H.h5.acute_p < 0.001 ? "p < 0.001" : "p = " + F.f3(H.h5.acute_p)}`, reject: true,
        so: "Demand history contains a structural break. Forecasts that ignore it learn the wrong seasonality, so the SARIMAX demand model in 07 carries a COVID intervention term.", chart: "h5" },
    ];
    for (const c of cards) {
      const card = el("div", "hyp");
      const left = el("div");
      left.append(el("p", "hyp-id", c.id), el("h4", null, c.title));
      const dl = el("dl");
      for (const [k, v] of [["H₀", c.h0], ["H₁", c.h1], ["Test", c.test], ["Result", c.res]]) { dl.append(el("dt", null, k), el("dd", null, v)); }
      left.appendChild(dl);
      const v = el("span", "verdict " + (c.reject ? "reject" : "keep"));
      v.innerHTML = c.reject ? check : dash;
      v.appendChild(document.createTextNode(c.reject ? "Reject H₀ at α = 0.05" : "Fail to reject H₀"));
      left.appendChild(v);
      left.appendChild(el("p", "sowhat", c.so));
      const right = el("div", "hyp-fig");
      const chartEl = el("div", "chart");
      right.appendChild(chartEl);
      card.append(left, right);
      host.appendChild(card);
      hypChart(c.chart, chartEl, H);
    }
  })();

  function hypChart(kind, host, H) {
    if (kind === "h1") {
      const g = H.h1.groups[2];
      const leg = el("div", "legend"); host.before(leg);
      RL.legend(leg, [{ label: "Kaplan–Meier (data)", color: "var(--ink)" }, { label: "Weibull, 3-parameter", color: "var(--s1)" }, { label: "Weibull, 2-parameter", color: "var(--s2)" }, { label: "Exponential (H₀)", color: "var(--neutral-mark)" }]);
      const mean = g.mean;
      RL.lineChart(host, {
        height: 260, xLabel: "Cycles", yLabel: "Share of engines still running", yDomain: [0, 1], yFormat: d3.format(".0%"), tipFormat: F.pct1,
        series: [
          { name: "Exponential (H₀)", color: "var(--neutral-mark)", values: g.grid.map((t) => ({ x: t, y: Math.exp(-t / mean) })), width: 1.5 },
          { name: "Weibull 2-p", color: "var(--s2)", values: g.grid.map((t, i) => ({ x: t, y: g.s_w2[i] })) },
          { name: "Weibull 3-p", color: "var(--s1)", values: g.grid.map((t, i) => ({ x: t, y: g.s_w3[i] })) },
          { name: "Kaplan–Meier", color: "var(--ink)", values: g.km.t.map((t, i) => ({ x: t, y: g.km.s[i] })), step: true, width: 1.5, noTip: true },
        ],
      });
    } else if (kind === "h2" || kind === "h3") {
      const d = kind === "h2" ? H.h2 : H.h3;
      const leg = el("div", "legend"); host.before(leg);
      RL.legend(leg, [{ label: d.a, color: "var(--s1)" }, { label: d.b, color: "var(--s2)" }]);
      const toVals = (hist) => d.bins.slice(0, -1).map((b, i) => ({ x: b + 12.5, y: hist[i] / d3.sum(hist) }));
      RL.lineChart(host, {
        height: 230, xLabel: "Engine life (cycles)", yLabel: "Share of engines", yFormat: d3.format(".0%"), tipFormat: F.pct1,
        series: [{ name: d.a, color: "var(--s1)", values: toVals(d.hist_a), area: true }, { name: d.b, color: "var(--s2)", values: toVals(d.hist_b), area: true }],
      });
    } else if (kind === "h4") {
      const rows = H.h4.chapters.slice(0, 14);
      RL.responsive(host, (w) => {
        const rh = 20, m = { top: 6, right: 16, bottom: 34, left: Math.min(170, w * 0.45) };
        const h = m.top + m.bottom + rh * rows.length, iw = w - m.left - m.right;
        const svg = RL.svg(host, w, h, "Interruption rate by system");
        const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
        const x = d3.scaleLinear().domain([0, 0.8]).range([0, iw]);
        g.append("g").attr("class", "gridline").attr("transform", `translate(0,${rh * rows.length})`).call(d3.axisBottom(x).ticks(4).tickSize(-rh * rows.length).tickFormat(""));
        g.append("g").attr("class", "axis").attr("transform", `translate(0,${rh * rows.length})`).call(d3.axisBottom(x).ticks(4).tickFormat(d3.format(".0%")));
        g.append("text").attr("class", "axis-label").attr("x", iw).attr("y", rh * rows.length + 30).attr("text-anchor", "end").text("Share of in-flight reports with a precautionary procedure");
        rows.forEach((r, i) => {
          const cy = i * rh + rh / 2, c = r.is_engine ? "var(--s2)" : "var(--s1)";
          g.append("line").attr("x1", x(r.lo)).attr("x2", x(r.hi)).attr("y1", cy).attr("y2", cy).style("stroke", c).style("stroke-width", 2).style("stroke-linecap", "round");
          g.append("circle").attr("cx", x(r.rate)).attr("cy", cy).attr("r", 4.5).style("fill", c).style("stroke", "var(--surface)").style("stroke-width", 2);
          g.append("text").attr("class", "dlabel" + (r.is_engine ? "" : " dim")).attr("x", -8).attr("y", cy).attr("dy", "0.35em").attr("text-anchor", "end").text(r.title);
          g.append("rect").attr("class", "hit").attr("x", -m.left).attr("y", cy - rh / 2).attr("width", iw + m.left).attr("height", rh)
            .on("pointermove", (evt) => RL.tip.show(evt, `${r.title} (JASC ${r.chapter})`, [{ v: F.pct1(r.rate), k: `95% CI ${F.pct1(r.lo)}–${F.pct1(r.hi)} · n = ${F.int(r.n)}` }]))
            .on("pointerleave", RL.tip.hide);
        });
      });
      const leg = el("div", "legend"); host.before(leg);
      RL.legend(leg, [{ label: "Engine systems", color: "var(--s2)", type: "dot" }, { label: "Other systems", color: "var(--s1)", type: "dot" }]);
    } else if (kind === "h5") {
      const d = H.h5;
      const t = d.months.map((m) => d3.timeParse("%Y-%m")(m));
      const leg = el("div", "legend"); host.before(leg);
      RL.legend(leg, [{ label: "Monthly SDRs", color: "var(--s1)" }, { label: "Pre-2020 trend, projected", color: "var(--neutral-mark)" }]);
      RL.lineChart(host, {
        height: 240, xTime: true, yLabel: "Reports per month", yFormat: d3.format("~s"), tipFormat: F.int, xTipFormat: d3.timeFormat("%b %Y"),
        series: [{ name: "Counterfactual", color: "var(--neutral-mark)", values: t.map((x, i) => ({ x, y: d.counterfactual[i] })), width: 1.5 },
          { name: "Reports", color: "var(--s1)", values: t.map((x, i) => ({ x, y: d.y[i] })) }],
        yDomain: [0, d3.max(d.y) * 1.1],
        after(g, x, y, iw, ih) {
          const a = x(new Date(2020, 3, 1)), b = x(new Date(2021, 2, 1));
          g.insert("rect", ":first-child").attr("x", a).attr("width", b - a).attr("y", 0).attr("height", ih).style("fill", "var(--accent-wash)");
          g.append("text").attr("class", "ann").attr("x", b + 4).attr("y", 12).text("acute COVID period");
        },
      });
    }
  }

  // ================================================================ 05 PREDICT
  (function predict() {
    const M = DATA.models;
    const tb = $("#leaderboard tbody");
    for (const r of M.metrics) {
      const tr = tb.insertRow();
      if (r.model === "ens") tr.className = "hl";
      const cells = [r.label, F.f2(r.rmse_FD001), F.f2(r.rmse_FD002), F.f2(r.rmse_FD003), F.f2(r.rmse_FD004), F.f2(r.rmse), F.f2(r.rmse_decision_zone), F.int(r.score)];
      cells.forEach((v, i) => { const td = tr.insertCell(); td.textContent = v; if (i) td.className = "n"; });
    }

    const single = (s) => s === "FD001" || s === "FD003";
    RL.legend($("#lg-scatter"), [{ label: "Single-condition fleet (FD001, FD003)", color: "var(--s1)", type: "dot" }, { label: "Six-condition fleet (FD002, FD004)", color: "var(--s3)", type: "dot" }]);
    RL.scatter($("#c-scatter"), {
      height: 300, r: 3.4, opacity: 0.75,
      data: M.scatter.map((d) => ({ x: d.truth, y: d.pred, color: single(d.subset) ? "var(--s1)" : "var(--s3)", d })),
      xLabel: "True RUL (cycles)", yLabel: "Predicted RUL", xDomain: [0, 200], yDomain: [0, 140],
      before(g, x, y) {
        g.append("line").attr("class", "ref-line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(140)).attr("y2", y(140));
        g.append("text").attr("class", "ann").attr("x", x(118)).attr("y", y(128)).attr("text-anchor", "end").text("perfect");
        g.append("line").attr("class", "ref-line").attr("x1", x(125)).attr("x2", x(125)).attr("y1", y(0)).attr("y2", y(140));
        g.append("text").attr("class", "ann").attr("x", x(127)).attr("y", y(8)).text("cap 125");
      },
      tip: (p) => ({ title: p.d.engine_id, rows: [{ v: `${F.f1(p.y)} predicted`, k: `80% interval ${F.f0(p.d.lo80)}–${F.f0(p.d.hi80)}` }, { v: `${p.x} true` }] }),
    });

    RL.columns($("#c-bands"), {
      height: 260, data: M.bands.map((b) => ({ label: b.band, value: b.rmse, extra: [{ k: "bias", v: F.f1(b.bias) }, { k: "n cycles", v: F.int(b.n) }] })),
      yLabel: "RMSE (cycles)", xLabel: "True RUL band (cycles)", valueFormat: F.f1, valueName: "RMSE",
    });
    RL.table("c-bands", [{ label: "True RUL band", key: "band" }, { label: "Cycles", num: true, value: (r) => F.int(r.n) }, { label: "RMSE", num: true, value: (r) => F.f2(r.rmse) },
      { label: "Bias", num: true, value: (r) => F.f2(r.bias) }, { label: "10th pct error", num: true, value: (r) => F.f1(r.p10) }, { label: "90th pct error", num: true, value: (r) => F.f1(r.p90) }], M.bands);

    const pick = $("#eng-pick");
    M.trajectories.forEach((t, i) => {
      const o = el("option", null, `${t.id} · ends with ${t.final_truth} cycles left`);
      o.value = i; pick.appendChild(o);
    });
    pick.value = String(M.trajectories.findIndex((t) => t.subset === "FD001" && t.final_truth < 40) >= 0 ? M.trajectories.findIndex((t) => t.subset === "FD001" && t.final_truth < 40) : 0);
    RL.legend($("#lg-traj"), [{ label: "True RUL (capped at 125)", color: "var(--ink)" }, { label: "Ensemble prediction", color: "var(--s1)" }, { label: "80% interval", color: "var(--s1)", type: "band" }, { label: "P(≤ 30 cycles left)", color: "var(--s2)" }]);
    const drawTraj = () => {
      const t = M.trajectories[+pick.value];
      const host = $("#c-traj");
      host.replaceChildren();
      RL.responsive(host, (w) => {
        const h = 380, m = { top: 18, right: 18, bottom: 40, left: 46 }, iw = w - m.left - m.right;
        const h1 = 230, gap = 30, h2 = h - m.top - m.bottom - h1 - gap;
        const svg = RL.svg(host, w, h, "RUL trajectory for " + t.id);
        const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
        const x = d3.scaleLinear().domain(d3.extent(t.cycle)).range([0, iw]);
        const y = d3.scaleLinear().domain([0, 150]).range([h1, 0]);
        g.append("g").attr("class", "gridline").call(d3.axisLeft(y).ticks(5).tickSize(-iw).tickFormat(""));
        g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8)).select(".domain").remove();
        g.append("text").attr("class", "axis-label").attr("y", -6).text("RUL (cycles)");
        const pts = t.cycle.map((c, i) => ({ c, truth: Math.min(t.truth[i], 125), pred: t.pred[i], lo: t.lo[i], hi: t.hi[i], p: t.p30[i] }));
        g.append("path").datum(pts).style("fill", "var(--s1)").style("opacity", 0.15).attr("d", d3.area().x((d) => x(d.c)).y0((d) => y(d.lo)).y1((d) => y(Math.min(d.hi, 150))).curve(d3.curveMonotoneX));
        g.append("path").datum(pts).style("fill", "none").style("stroke", "var(--ink)").style("stroke-width", 1.5).attr("d", d3.line().x((d) => x(d.c)).y((d) => y(d.truth)));
        g.append("path").datum(pts).style("fill", "none").style("stroke", "var(--s1)").style("stroke-width", 2).attr("d", d3.line().x((d) => x(d.c)).y((d) => y(d.pred)).curve(d3.curveMonotoneX));
        const last = pts[pts.length - 1];
        g.append("circle").attr("cx", x(last.c)).attr("cy", y(last.pred)).attr("r", 4).style("fill", "var(--s1)").style("stroke", "var(--surface)").style("stroke-width", 2);
        g.append("text").attr("class", "dlabel").attr("x", x(last.c) - 6).attr("y", y(last.pred) - 10).attr("text-anchor", "end").text(`predicted ${F.f0(last.pred)} · true ${t.final_truth}`);
        const g2 = g.append("g").attr("transform", `translate(0,${h1 + gap})`);
        const y2 = d3.scaleLinear().domain([0, 1]).range([h2, 0]);
        g2.append("g").attr("class", "gridline").call(d3.axisLeft(y2).ticks(2).tickSize(-iw).tickFormat(""));
        g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(2).tickSize(0).tickPadding(8).tickFormat(d3.format(".0%"))).select(".domain").remove();
        g2.append("path").datum(pts).style("fill", "var(--s2)").style("opacity", 0.12).attr("d", d3.area().x((d) => x(d.c)).y0(h2).y1((d) => y2(d.p)));
        g2.append("path").datum(pts).style("fill", "none").style("stroke", "var(--s2)").style("stroke-width", 2).attr("d", d3.line().x((d) => x(d.c)).y((d) => y2(d.p)));
        g2.append("g").attr("class", "axis").attr("transform", `translate(0,${h2})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
        g2.append("text").attr("class", "axis-label").attr("x", iw).attr("y", h2 + 32).attr("text-anchor", "end").text("Cycle");
        g2.append("text").attr("class", "axis-label").attr("y", -6).text("Alert probability");
        const cross = g.append("line").attr("class", "crosshair").attr("y1", 0).attr("y2", h1 + gap + h2).style("opacity", 0);
        g.append("rect").attr("class", "hit").attr("width", iw).attr("height", h1 + gap + h2)
          .on("pointermove", (evt) => {
            const c = x.invert(d3.pointer(evt)[0]);
            const d = pts[Math.max(0, Math.min(pts.length - 1, d3.bisectCenter(pts.map((p) => p.c), c)))];
            cross.attr("x1", x(d.c)).attr("x2", x(d.c)).style("opacity", 1);
            RL.tip.show(evt, "Cycle " + d.c, [{ v: F.f0(d.pred), k: "predicted", color: "var(--s1)" }, { v: `${F.f0(d.lo)}–${F.f0(d.hi)}`, k: "80% interval" },
              { v: String(d.truth), k: "true (capped)", color: "var(--ink)" }, { v: F.pct0(d.p), k: "P(≤ 30 left)", color: "var(--s2)" }]);
          })
          .on("pointerleave", () => { cross.style("opacity", 0); RL.tip.hide(); });
      });
    };
    pick.addEventListener("change", drawTraj);
    drawTraj();

    const sh = M.explain.by_sensor.slice(0, 12);
    const special = { cycle: "Engine age (cycles flown)", frac_high_regime: "Share of high-altitude cycles" };
    RL.hbar($("#c-shap"), { data: sh.map((s) => ({ label: special[s.sensor] || `${s.abbr} (${s.sensor})`, value: s.shap, extra: special[s.sensor] ? [] : [{ k: s.name, v: "" }] })),
      labelW: 190, rowH: 22, valueFormat: F.f1, valueName: "mean |SHAP| (cycles)", xLabel: "Mean |SHAP| (cycles of RUL)" });

    const cb = M.conformal.coverage_bins;
    RL.responsive($("#c-coverage"), (w) => {
      const host = $("#c-coverage"), h = 260, m = { top: 22, right: 16, bottom: 44, left: 46 }, iw = w - m.left - m.right, ih = h - m.top - m.bottom;
      const svg = RL.svg(host, w, h, "Interval coverage");
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const x = d3.scaleBand().domain(cb.map((d) => d.bin)).range([0, iw]).padding(0.3);
      const y = d3.scaleLinear().domain([0.4, 1]).range([ih, 0]);
      RL.axes(g, x, y, iw, ih, { yFormat: d3.format(".0%"), xLabel: "Predicted RUL band (cycles)" });
      for (const [lv, c] of [[0.8, "var(--s1)"], [0.9, "var(--s3)"]]) {
        g.append("line").attr("class", "ref-line").attr("x1", 0).attr("x2", iw).attr("y1", y(lv)).attr("y2", y(lv));
      }
      g.append("text").attr("class", "ann").attr("x", iw).attr("y", y(0.9) - 5).attr("text-anchor", "end").text("90% nominal");
      g.append("text").attr("class", "ann").attr("x", iw).attr("y", y(0.8) - 5).attr("text-anchor", "end").text("80% nominal");
      cb.forEach((d) => {
        const cx = x(d.bin) + x.bandwidth() / 2;
        [[d.c80, "var(--s1)", -7, "80%"], [d.c90, "var(--s3)", 7, "90%"]].forEach(([v, c, dx, k]) => {
          g.append("circle").attr("cx", cx + dx).attr("cy", y(v)).attr("r", 5).style("fill", c).style("stroke", "var(--surface)").style("stroke-width", 2)
            .on("pointermove", (evt) => RL.tip.show(evt, `Predicted RUL ${d.bin} · ${d.n} engines`, [{ v: F.pct0(v), k: `${k} interval coverage`, color: c }]))
            .on("pointerleave", RL.tip.hide);
        });
      });
    });
    const lgc = el("div", "legend"); $("#c-coverage").before(lgc);
    RL.legend(lgc, [{ label: "80% interval", color: "var(--s1)", type: "dot" }, { label: "90% interval", color: "var(--s3)", type: "dot" }]);
  })();

  // ================================================================ 06 READ THE LOGS
  (function read() {
    const N = DATA.nlp;
    const C = N.classifier;
    const cm = C.confusion;
    const short = (s) => s.replace("Equipment & furnishings", "Equipment").replace("Indicating & recording", "Indicating").replace("Air conditioning", "Air cond.");
    RL.heatmap($("#c-confusion"), {
      rows: cm.titles.map((t, i) => `${cm.labels[i]} ${short(t)}`), cols: cm.labels, maxCell: 46,
      margin: { left: 150, bottom: 44 }, xLabel: "Predicted JASC chapter",
      value: (i, j) => cm.matrix[i][j], color: () => "var(--s1)", opacity: (v) => seqOpacity(Math.sqrt(v)), format: F.pct1, valueName: "of the row's reports",
      cellLabel: (v, i, j) => (i === j ? d3.format(".0%")(v) : v >= 0.05 ? d3.format(".0%")(v) : ""), labelInk: (v) => inkFor(Math.sqrt(v)),
      tipTitle: (i, j) => `Coded ${cm.labels[i]} ${cm.titles[i]} → predicted ${cm.labels[j]} ${cm.titles[j]}`,
    });
    const tb = $("#clf-table tbody");
    for (const r of C.results) {
      const tr = tb.insertRow();
      if (r.variant === "masked" && r.model.startsWith("Linear")) tr.className = "hl";
      [r.model, r.variant === "masked" ? "References masked" : "Raw text", F.pct1(r.accuracy), F.f3(r.macro_f1), F.pct1(r.top3)].forEach((v, i) => {
        const td = tr.insertCell(); td.textContent = v; if (i > 1) td.className = "n";
      });
    }
    const ex = $("#miscodes");
    for (const e of C.miscode.examples.slice(0, 6)) {
      const card = el("div", "ex");
      const meta = el("div", "meta");
      meta.append(el("span", "pill", `coded ${e.coded} ${e.coded_title}`), el("span", null, "→"), el("span", "pill acc", `model ${e.pred} ${e.pred_title}`), el("span", null, F.pct0(e.conf)));
      card.append(meta, el("q", null, e.text));
      ex.appendChild(card);
    }

    const T = N.topics.topics;
    const detail = $("#topic-detail");
    const showTopic = (t) => {
      detail.replaceChildren();
      detail.appendChild(el("span", "k", `${t.name} · ${F.int(t.n)} reports`));
      const p = el("p");
      const acts = Object.entries(t.actions).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k === "Unstated" ? "not stated" : k.toLowerCase()} ${F.pct0(v)}`).join(" · ");
      p.textContent = `${F.pct0(t.interruption_rate)} interrupted a flight · ${F.pct0(t.inflight_share)} found in flight · corrective action: ${acts}. Top terms: ${t.terms.slice(0, 8).join(", ").toLowerCase()}.`;
      detail.appendChild(p);
      const exs = el("div", "examples");
      for (const x of t.examples) { const c = el("div", "ex"); c.appendChild(el("q", null, x)); exs.appendChild(c); }
      detail.appendChild(exs);
    };
    RL.hbar($("#c-pareto"), {
      data: T.map((t, i) => ({ label: t.name, value: t.impact, t, highlight: i < 3, color: i < 3 ? "var(--s2)" : "var(--s1)",
        extra: [{ k: "reports", v: F.int(t.n) }, { k: "interrupted a flight", v: F.pct0(t.interruption_rate) }, { k: "cumulative share of impact", v: F.pct0(t.cum_impact_share) }] })),
      labelW: 260, rowH: 24, valueFormat: F.f0, valueName: "interrupted flights (reports × rate)", xLabel: "Reports that ended in a precautionary procedure",
      onClick: (d) => showTopic(d.t),
      after(g, x, y, iw) {
        T.forEach((t, i) => {
          g.append("text").attr("class", "ann").attr("x", iw + 48).attr("y", y(i) + y.bandwidth() / 2).attr("dy", "0.35em").attr("text-anchor", "end").text(F.pct0(t.cum_impact_share));
        });
        g.append("text").attr("class", "ann").attr("x", iw + 48).attr("y", -1).attr("text-anchor", "end").text("cum.");
      },
      margin: { right: 60 },
    });
    showTopic(T[0]);
    RL.table("c-pareto", [{ label: "Theme", key: "name" }, { label: "Reports", num: true, value: (r) => F.int(r.n) }, { label: "Interruption rate", num: true, value: (r) => F.pct1(r.interruption_rate) },
      { label: "Impact", num: true, value: (r) => F.f0(r.impact) }, { label: "Cumulative", num: true, value: (r) => F.pct0(r.cum_impact_share) }, { label: "Top terms", value: (r) => r.terms.slice(0, 5).join(", ").toLowerCase() }], T);

    const L = N.l100;
    $("#l100").textContent = `The SDR data include ${F.int(L.n)} reports on Lockheed aircraft from ${L.years[0]}–${L.years[1]}, almost all on the model 382G, the civil L-100 Hercules. Its most-reported systems are ${L.chapters.slice(0, 3).map((c) => c.title.toLowerCase()).join(", ")}; ${F.pct1(L.interruption_rate)} of its reports involved a precautionary procedure.`;
  })();

  // ================================================================ 07 FORECAST
  (function forecast() {
    const FC = DATA.forecast.fleet;
    const pt = d3.timeParse("%Y-%m");
    const hist = FC.months.map((m, i) => ({ x: pt(m), y: FC.y[i] }));
    const hIdx = FC.months.indexOf(FC.holdout_start);
    const hold = FC.months.slice(hIdx).map(pt);
    const ets = FC.holdout["Exponential smoothing (ETS)"], sar = FC.holdout["SARIMAX + COVID regressor"];
    const fut = FC.future.months.map(pt);
    RL.legend($("#lg-forecast"), [{ label: "Actual", color: "var(--ink-2)" }, { label: "ETS holdout forecast", color: "var(--s1)" }, { label: "SARIMAX holdout forecast", color: "var(--s2)" }, { label: "2026 forecast (SARIMAX) + 80% band", color: "var(--s3)" }]);
    RL.lineChart($("#c-forecast"), {
      height: 300, xTime: true, yLabel: "Engine R&R reports / month", tipFormat: F.f0, xTipFormat: d3.timeFormat("%b %Y"),
      series: [{ name: "Actual", color: "var(--ink-2)", values: hist, width: 1.5 },
        { name: "ETS", color: "var(--s1)", values: hold.map((x, i) => ({ x, y: ets[i] })) },
        { name: "SARIMAX", color: "var(--s2)", values: hold.map((x, i) => ({ x, y: sar[i] })) },
        { name: "2026 forecast", color: "var(--s3)", values: fut.map((x, i) => ({ x, y: FC.future.mean[i] })) }],
      bands: [{ name: "80% band", color: "var(--s3)", values: fut.map((x, i) => ({ x, lo: FC.future.lo80[i], hi: FC.future.hi80[i] })) }],
      refs: [{ x: pt(FC.holdout_start), label: "holdout" }], yDomain: [0, d3.max(FC.y) * 1.25],
    });
    const tb = $("#fc-table tbody");
    for (const s of FC.scores) { const tr = tb.insertRow(); [s.model, F.f2(s.mase), F.f1(s.rmse)].forEach((v, i) => { const td = tr.insertCell(); td.textContent = v; if (i) td.className = "n"; }); }

    const P = DATA.forecast.parts;
    RL.scatter($("#c-adi"), {
      height: 280, r: 4.5, xLog: true, yLog: true,
      data: P.scatter.filter((d) => d.cv2 > 0).map((d) => ({ x: d.adi, y: Math.max(d.cv2, 0.02), d })),
      xDomain: [1, 40], yDomain: [0.02, 4], xLabel: "Average demand interval (months, log)", yLabel: "CV² of demand size (log)",
      xFormat: d3.format("~g"), yFormat: d3.format("~g"), xTickValues: [1, 2, 5, 10, 20, 40], yTickValues: [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4],
      before(g, x, y, iw, ih) {
        g.append("line").attr("class", "ref-line").attr("x1", x(1.32)).attr("x2", x(1.32)).attr("y1", 0).attr("y2", ih);
        g.append("line").attr("class", "ref-line").attr("x1", 0).attr("x2", iw).attr("y1", y(0.49)).attr("y2", y(0.49));
        const lab = (t, xx, yy, a) => g.append("text").attr("class", "ann").attr("x", xx).attr("y", yy).attr("text-anchor", a || "start").text(t);
        lab("Smooth", 4, ih - 6); lab("Erratic", 4, 12); lab("Intermittent", iw - 4, ih - 6, "end"); lab("Lumpy", iw - 4, 12, "end");
      },
      tip: (p) => ({ title: p.d.part, rows: [{ v: p.d.class, k: `${F.int(p.d.total)} replacements` }, { v: `ADI ${F.f2(p.d.adi)} · CV² ${F.f2(p.d.cv2)}` }] }),
    });
    const best = d3.min(P.summary, (s) => s.median_rmsse);
    RL.hbar($("#c-intermittent"), {
      data: P.summary.map((s) => ({ label: s.method, value: s.median_rmsse, highlight: s.median_rmsse - best < 0.01, color: s.method.startsWith("Naive") ? "var(--neutral-mark)" : "var(--s1)",
        extra: [{ k: "best on", v: `${s.wins} of ${P.n_series} parts` }, { k: "median |bias|", v: F.pct0(s.median_abs_bias) }] })),
      labelW: 160, rowH: 26, valueFormat: F.f3, xDomain: [0, 1], valueName: "median RMSSE", xLabel: "Median RMSSE (lower is better)",
    });
  })();

  // ================================================================ 08 SIMULATE
  (function simulate() {
    const SIM = DATA.simulation;
    const pols = SIM.policies.policies;
    const target = SIM.pbl_target;
    const intervalChart = (host, rows, o) => RL.responsive(host, (w) => {
      const rh = o.rowH || 38, m = { top: 18, right: 24, bottom: 40, left: Math.min(o.labelW || 190, w * 0.42) };
      const h = m.top + m.bottom + rh * rows.length, iw = w - m.left - m.right, ih = rh * rows.length;
      const svg = RL.svg(host, w, h, o.label);
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const x = d3.scaleLinear().domain(o.domain).range([0, iw]);
      g.append("g").attr("class", "gridline").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6).tickSize(-ih).tickFormat(""));
      g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6).tickFormat(o.format || d3.format(".0%")));
      g.append("text").attr("class", "axis-label").attr("x", iw).attr("y", ih + 34).attr("text-anchor", "end").text(o.xLabel);
      if (o.ref != null) {
        g.append("line").attr("class", "ref-line").attr("x1", x(o.ref)).attr("x2", x(o.ref)).attr("y1", -8).attr("y2", ih);
        g.append("text").attr("class", "ann").attr("x", x(o.ref) + 4).attr("y", -6).text(o.refLabel);
      }
      rows.forEach((r, i) => {
        const cy = i * rh + rh / 2, c = r.color;
        g.append("line").attr("x1", x(r.lo)).attr("x2", x(r.hi)).attr("y1", cy).attr("y2", cy).style("stroke", c).style("stroke-width", 2).style("stroke-linecap", "round").style("opacity", 0.55);
        g.append("circle").attr("cx", x(r.mean)).attr("cy", cy).attr("r", 5.5).style("fill", c).style("stroke", "var(--surface)").style("stroke-width", 2);
        g.append("text").attr("class", "dlabel").attr("x", -10).attr("y", cy).attr("dy", "0.35em").attr("text-anchor", "end").text(r.label);
        g.append("text").attr("class", "dlabel dim").attr("x", x(r.hi) + 8).attr("y", cy).attr("dy", "0.35em").text((o.format || F.pct1)(r.mean));
        g.append("rect").attr("class", "hit").attr("x", -m.left).attr("y", cy - rh / 2).attr("width", iw + m.left + m.right).attr("height", rh)
          .on("pointermove", (evt) => RL.tip.show(evt, r.label, r.tip)).on("pointerleave", RL.tip.hide);
      });
    });
    const pcol = { "Run to failure": "var(--neutral-mark)", "Hard-time removal": "var(--s4)", "Predictive · ridge": "var(--s3)", "Predictive · ensemble": "var(--s1)", "Perfect foresight": "var(--s7)" };
    intervalChart($("#c-policies"), pols.map((p) => ({ label: p.name, mean: p.mc.mean, lo: p.mc.p05, hi: p.mc.p95, color: pcol[p.name],
      tip: [{ v: F.pct1(p.mc.mean), k: "mean MC" }, { v: `${F.pct1(p.mc.p05)} – ${F.pct1(p.mc.p95)}`, k: "5th–95th pct" }, { v: F.f1(p.unsched_yr.mean), k: "unscheduled removals / yr" }] })),
      { domain: [0.5, 0.72], ref: target, refLabel: `PBL ${F.pct0(target)}`, xLabel: "Mission-capable rate (simulated year)", label: "Policy comparison" });
    RL.table("c-policies", [{ label: "Policy", key: "name" }, { label: "Mean MC", num: true, value: (r) => F.pct1(r.mc.mean) }, { label: "5th pct", num: true, value: (r) => F.pct1(r.mc.p05) },
      { label: "95th pct", num: true, value: (r) => F.pct1(r.mc.p95) }], pols);
    const tb = $("#pol-table tbody");
    for (const p of pols) {
      const tr = tb.insertRow();
      if (p.name === "Predictive · ensemble") tr.className = "hl";
      const name = el("td"); name.append(el("span", null, p.name), el("span", "sub", p.desc)); tr.appendChild(name);
      [F.pct1(p.mc.mean), F.pct0(p.p_meet_target), F.f1(p.unsched_yr.mean), F.f1(p.planned_yr.mean), F.pct0(p.life_used.mean), F.usd(p.cost_yr.mean)].forEach((v) => { const td = tr.insertCell(); td.textContent = v; td.className = "n"; });
    }

    // value of prognostics: two small panels (no dual axis)
    const V = SIM.value;
    RL.responsive($("#c-value"), (w) => {
      const host = $("#c-value"), ph = 160, gap = 58, m = { top: 22, right: 24, bottom: 40, left: 56 }, iw = w - m.left - m.right;
      const svg = RL.svg(host, w, m.top + ph * 2 + gap + m.bottom, "Value of prognostics");
      const x = d3.scaleLinear().domain([0, d3.max(V, (d) => d.rmse) * 1.05]).range([0, iw]);
      const panel = (top, key, fmt, label, dom) => {
        const g = svg.append("g").attr("transform", `translate(${m.left},${m.top + top})`);
        const y = d3.scaleLinear().domain(dom).nice().range([ph, 0]);
        g.append("g").attr("class", "gridline").call(d3.axisLeft(y).ticks(4).tickSize(-iw).tickFormat(""));
        g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(8).tickFormat(fmt)).select(".domain").remove();
        g.append("g").attr("class", "axis").attr("transform", `translate(0,${ph})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
        g.append("text").attr("class", "axis-label").attr("y", -7).text(label);
        g.append("path").datum(V).style("fill", "none").style("stroke", "var(--s1)").style("stroke-width", 2).attr("d", d3.line().x((d) => x(d.rmse)).y((d) => y(d[key])));
        V.forEach((d) => {
          const ours = d.alpha === 1;
          g.append("circle").attr("cx", x(d.rmse)).attr("cy", y(d[key])).attr("r", ours ? 6 : 4).style("fill", ours ? "var(--accent)" : "var(--s1)").style("stroke", "var(--surface)").style("stroke-width", 2)
            .on("pointermove", (evt) => RL.tip.show(evt, `${d.alpha}× model error · RMSE ${F.f1(d.rmse)}`, [{ v: F.pct1(d.mc), k: "MC rate" }, { v: F.usd(d.cost), k: "cost / yr" }, { v: String(d.threshold), k: "tuned threshold" }]))
            .on("pointerleave", RL.tip.hide);
          if (ours || d.alpha === 0 || d.alpha === 4) {
            const up = key === "mc" && d.alpha !== 0;
            g.append("text").attr("class", "dlabel" + (ours ? "" : " dim"))
              .attr("x", x(d.rmse) + (d.alpha === 0 ? 8 : 0)).attr("y", y(d[key]) + (up ? -11 : 18))
              .attr("text-anchor", d.alpha === 4 ? "end" : d.alpha === 0 ? "start" : "middle")
              .text(ours ? "our model" : d.alpha === 0 ? "oracle" : "4× error");
          }
        });
        return g;
      };
      const mcs = V.map((d) => d.mc), cs = V.map((d) => d.cost);
      const baseMC = pols[0].mc.mean;
      const g1 = panel(0, "mc", d3.format(".1%"), "Mission-capable rate", [Math.min(baseMC, d3.min(mcs)) - 0.005, d3.max(mcs) + 0.005]);
      const y1 = d3.scaleLinear().domain([Math.min(baseMC, d3.min(mcs)) - 0.005, d3.max(mcs) + 0.005]).nice().range([ph, 0]);
      g1.append("line").attr("class", "ref-line").attr("x1", 0).attr("x2", iw).attr("y1", y1(baseMC)).attr("y2", y1(baseMC));
      g1.append("text").attr("class", "ann").attr("x", iw).attr("y", y1(baseMC) - 5).attr("text-anchor", "end").text(`run to failure ${F.pct1(baseMC)}`);
      const g2 = panel(ph + gap, "cost", (v) => "$" + v + "M", "Engine sustainment cost per year", [d3.min(cs) - 1, d3.max(cs) + 1]);
      g2.append("text").attr("class", "axis-label").attr("x", iw).attr("y", ph + 34).attr("text-anchor", "end").text("Decision-zone RMSE of the prognostic (cycles)");
    });

    const SU = SIM.surfaces;
    let sp = "reactive";
    const drawSurf = () => {
      $("#c-surface").replaceChildren();
      const grid = SU[sp];
      RL.heatmap($("#c-surface"), {
        rows: SU.tats.map((t) => `${t} days`), cols: SU.spares.map(String), maxCell: 52, margin: { left: 70 },
        xLabel: "Spare engines", yLabel: "Turnaround",
        value: (i, j) => grid[i][j].mc, color: () => "var(--s1)", opacity: (v) => seqOpacity(v, 0.4, 0.7), labelInk: (v) => inkFor(v, 0.4, 0.7), format: F.pct1, valueName: "mean MC",
        cellLabel: (v) => d3.format(".0%")(v), tipTitle: (i, j) => `${SU.spares[j]} spares · ${SU.tats[i]}-day turnaround`,
      });
    };
    RL.chips($("#surf-pol"), (v) => { sp = v; drawSurf(); });
    drawSurf();

    const B = SIM.benchmark;
    RL.legend($("#lg-bench"), [{ label: "CPU (M2, 8 cores)", color: "var(--neutral-mark)" }, { label: "GPU (M2, Metal/MPS)", color: "var(--s1)" }]);
    RL.lineChart($("#c-bench"), {
      height: 240, xLabel: "Replications (simulated squadron-years per run)", yLabel: "Seconds", tipFormat: (v) => F.f1(v) + " s",
      series: [{ name: "CPU", color: "var(--neutral-mark)", values: B.map((b) => ({ x: b.R, y: b.cpu })), endLabel: true },
        { name: "GPU", color: "var(--s1)", values: B.map((b) => ({ x: b.R, y: b[SIM.device] })), endLabel: true }],
      xFormat: d3.format("~s"),
    });

    // ---- lab: the in-browser simulator
    const lib = {};
    const getLib = (alpha) => { const k = alpha.toFixed(2); if (!lib[k]) lib[k] = RLSim.makeLibrary(DATA.profiles, "ens", alpha); return lib[k]; };
    const base = Object.assign({}, SIM.base);
    const ids = ["spares", "tat", "pratio", "th", "ht", "alpha", "pfly"];
    const inp = Object.fromEntries(ids.map((k) => [k, document.getElementById("lab-" + k)]));
    const out = Object.fromEntries(ids.map((k) => [k, document.getElementById("o-" + k)]));
    const policySel = document.getElementById("lab-policy");
    const resetVals = () => {
      inp.spares.value = base.n_spares; inp.tat.value = base.tat_unsched; inp.pratio.value = Math.round(100 * base.tat_planned / base.tat_unsched);
      inp.th.value = SIM.policies.thresholds.ens; inp.ht.value = SIM.policies.thresholds.hard_time; inp.alpha.value = 1; inp.pfly.value = base.p_fly; policySel.value = "predictive";
    };
    const sync = () => {
      out.spares.textContent = inp.spares.value; out.tat.textContent = inp.tat.value; out.pratio.textContent = inp.pratio.value + "%";
      out.th.textContent = inp.th.value; out.ht.textContent = inp.ht.value; out.alpha.textContent = (+inp.alpha.value).toFixed(2).replace(/0$/, "") + "×"; out.pfly.textContent = (+inp.pfly.value).toFixed(2);
    };
    let baseline = null;
    const R = 200;
    const runLab = () => {
      const btn = document.getElementById("lab-run");
      btn.disabled = true; btn.textContent = "Running…";
      setTimeout(() => {
        const t0 = performance.now();
        if (!baseline) baseline = RLSim.simulate(Object.assign({}, base, { policy: "reactive" }), R, getLib(1), 99);
        const tat = +inp.tat.value;
        const cfg = Object.assign({}, base, { policy: policySel.value, n_spares: +inp.spares.value, tat_unsched: tat, tat_planned: tat * (+inp.pratio.value) / 100,
          threshold: +inp.th.value, hard_time: +inp.ht.value, p_fly: +inp.pfly.value, cost_phm_year: policySel.value === "predictive" ? 0.6 : 0 });
        const res = RLSim.simulate(cfg, R, getLib(+inp.alpha.value), 7);
        const ms = performance.now() - t0;
        drawLab(res, cfg, ms);
        btn.disabled = false; btn.textContent = `Run ${R} simulated years`;
      }, 20);
    };
    const drawLab = (res, cfg, ms) => {
      const mc = RLSim.mean(res.mc), mc0 = RLSim.mean(baseline.mc);
      const pm = Array.from(res.mc).filter((v) => v >= target).length / R;
      const st = $("#lab-stats");
      st.replaceChildren();
      const tile = (v, l) => { const d = el("div", "stat"); d.append(el("div", "v", v), el("div", "l", l)); st.appendChild(d); };
      tile(F.pct1(mc), `Mean MC (${F.pts(mc - mc0)} pts vs status quo)`);
      tile(F.pct0(pm), `Chance of meeting the ${F.pct0(target)} requirement`);
      tile(F.f1(RLSim.mean(res.unsched_yr)), "Unscheduled engine removals / yr");
      tile(F.usd(RLSim.mean(res.cost_yr)), "Engine sustainment cost / yr");
      const bins = d3.range(0.40, 0.801, 0.01);
      const hist = (arr) => bins.slice(0, -1).map((b, i) => ({ x: b + 0.005, y: Array.from(arr).filter((v) => v >= b && v < bins[i + 1]).length / R }));
      $("#c-lab").replaceChildren();
      RL.lineChart($("#c-lab"), {
        height: 220, xLabel: "Annual MC rate", yLabel: "Share of simulated years", xFormat: d3.format(".0%"), yFormat: d3.format(".0%"), tipFormat: F.pct1, xTipFormat: (v) => F.pct0(v),
        series: [{ name: "Status quo", color: "var(--neutral-mark)", values: hist(baseline.mc), area: true, width: 1.5 }, { name: "Your scenario", color: "var(--s1)", values: hist(res.mc), area: true }],
        refs: [{ x: target, label: `PBL ${F.pct0(target)}` }], xDomain: [0.42, 0.78],
      });
      $("#lab-note").textContent = `${R} simulated years in ${F.f0(ms)} ms in your browser. Gray: status quo (run to failure, ${base.n_spares} spares, ${base.tat_unsched}-day depot). Blue: your scenario.`;
    };
    for (const k of ids) inp[k].addEventListener("input", sync);
    document.getElementById("lab-run").addEventListener("click", runLab);
    document.getElementById("lab-reset").addEventListener("click", () => { resetVals(); sync(); runLab(); });
    resetVals(); sync();
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { io.disconnect(); runLab(); } }, { rootMargin: "200px" });
      io.observe(document.getElementById("lab"));
    } else runLab();
  })();

  // ================================================================ 09 OPTIMIZE
  (function optimize() {
    const O = DATA.optimization;
    const SC = O.spares.scenarios;
    const keys = [["reactive", "Run to failure", "var(--neutral-mark)"], ["predictive", "Predictive removal", "var(--s1)"]];
    RL.legend($("#lg-frontier"), keys.map(([, l, c]) => ({ label: l, color: c })).concat([{ label: "Rule of thumb (stock = mean pipeline, rounded up)", color: "var(--s2)", type: "dot" }]));
    const budget = $("#budget"), ob = $("#o-budget");
    let chart;
    const drawFrontier = () => {
      $("#c-frontier").replaceChildren();
      chart = RL.lineChart($("#c-frontier"), {
        height: 280, xLabel: "Spares budget ($M)", yLabel: "Expected aircraft availability (engine + LRUs)", yDomain: [0, 1], yFormat: d3.format(".0%"), tipFormat: F.pct1, xFormat: (v) => v / 1000,
        xTipFormat: (v) => "$" + F.f1(v / 1000) + "M",
        series: keys.map(([k, l, c]) => ({ name: l, color: c, values: SC[k].frontier.map((f) => ({ x: f.budget, y: f.A })), step: true })),
        after(g, x, y, iw, ih) {
          for (const [k, , c] of keys) {
            const r = SC[k].rule_of_thumb;
            g.append("circle").attr("cx", x(r.cost)).attr("cy", y(r.A)).attr("r", 5.5).style("fill", "var(--s2)").style("stroke", "var(--surface)").style("stroke-width", 2);
            g.append("text").attr("class", "ann").attr("x", x(r.cost) + 8).attr("y", y(r.A) + 4).text(k === "reactive" ? "rule of thumb" : "");
          }
          const bx = x(+budget.value);
          g.append("line").attr("class", "crosshair").attr("x1", bx).attr("x2", bx).attr("y1", 0).attr("y2", ih).style("stroke", "var(--accent)").style("stroke-width", 1.5);
        },
      });
    };
    const drawAlloc = () => {
      const b = +budget.value;
      ob.textContent = "$" + F.f1(b / 1000) + "M";
      const pick = (k) => SC[k].frontier.reduce((best, f) => (f.budget <= b ? f : best), SC[k].frontier[0]);
      const fr = pick("reactive"), fp = pick("predictive");
      const items = SC.reactive.items;
      const host = $("#c-alloc");
      host.replaceChildren();
      let note = host.previousElementSibling;
      if (!note || !note.classList.contains("alloc-note")) { note = el("p", "alloc-note"); host.before(note); }
      note.textContent = `Units stocked at $${F.f1(b / 1000)}M. Expected availability ${F.pct1(fr.A)} under run to failure (gray) vs ${F.pct1(fp.A)} under predictive removal (blue).`;
      RL.responsive(host, (w) => {
        const rh = 26, m = { top: 4, right: 40, bottom: 10, left: Math.min(230, w * 0.45) };
        const h = m.top + m.bottom + rh * items.length, iw = w - m.left - m.right;
        const svg = RL.svg(host, w, h, "Optimal stock levels");
        const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
        const mx = d3.max(items.map((_, i) => Math.max(fr.stock[i], fp.stock[i]))) || 1;
        const x = d3.scaleLinear().domain([0, mx]).range([0, iw]);

        items.forEach((it, i) => {
          const cy = i * rh;
          g.append("text").attr("class", "dlabel" + (it.code === "ENG" ? "" : " dim")).attr("x", -8).attr("y", cy + rh / 2).attr("dy", "0.35em").attr("text-anchor", "end").text(`${it.name}`);
          [[fr.stock[i], "var(--neutral-mark)", 4], [fp.stock[i], "var(--s1)", 13]].forEach(([v, c, dy]) => {
            g.append("rect").attr("x", 0).attr("y", cy + dy - 4).attr("height", 8).attr("rx", 3).attr("width", Math.max(v ? 2 : 0, x(v))).style("fill", c);
            g.append("text").attr("class", "ann").attr("x", x(v) + 5).attr("y", cy + dy).attr("dy", "0.35em").text(v);
          });
          g.append("rect").attr("class", "hit").attr("x", -m.left).attr("y", cy).attr("width", w).attr("height", rh)
            .on("pointermove", (evt) => RL.tip.show(evt, `${it.name} · ${F.usdK(it.cost)} each`, [
              { v: `${fr.stock[i]} → ${fp.stock[i]}`, k: "units, run to failure → predictive" },
              { v: F.f2(SC.reactive.items[i].pipeline) + " / " + F.f2(SC.predictive.items[i].pipeline), k: "mean units in repair" },
              { v: it.sdr_reports ? F.int(it.sdr_reports) + " SDR replacements" : "–", k: it.sdr_parts.join(", ").toLowerCase() }]))
            .on("pointerleave", RL.tip.hide);
        });
      });
    };
    budget.addEventListener("input", () => { drawFrontier(); drawAlloc(); });
    drawFrontier(); drawAlloc();

    const AR = DATA.hypotheses.age_replacement;
    const ratio = $("#ratio"), orat = $("#o-ratio");
    const drawAge = () => {
      const c = AR.curves[+ratio.value];
      orat.textContent = c.ratio + "×";
      $("#c-agerep").replaceChildren();
      RL.lineChart($("#c-agerep"), {
        height: 260, xLabel: "Replacement age T (cycles)", yLabel: "Cost per cycle (× planned removal cost)", tipFormat: (v) => F.f3(v * 100) + " /100 cyc",
        series: [{ name: "Age replacement", color: "var(--s1)", values: AR.T.map((t, i) => ({ x: t, y: c.cost[i] })) }],
        refs: [{ y: c.rtf, label: `run to failure ${F.f3(c.rtf * 100)} per 100 cycles` }],
        yDomain: [0, Math.max(c.rtf * 1.25, d3.min(c.cost) * 2)],
        after(g, x, y) {
          g.append("circle").attr("cx", x(c.t_star)).attr("cy", y(c.c_star)).attr("r", 5.5).style("fill", "var(--accent)").style("stroke", "var(--surface)").style("stroke-width", 2);
          g.append("text").attr("class", "dlabel").attr("x", x(c.t_star) + 9).attr("y", y(c.c_star) + 16).text(`T* = ${F.f0(c.t_star)} cycles · ${F.pct0(c.saving)} below run to failure`);
        },
      });
    };
    ratio.addEventListener("input", drawAge);
    drawAge();

    const I = O.induction;
    const H = I.params.weeks, cpw = I.params.cycles_per_week;
    const fleet = I.fleet.filter((e) => e.week_milp >= 0 || e.week_rule >= 0 || e.p_fail_h >= 0.1).sort((a, b) => a.pred - b.pred);
    let view = "milp";
    RL.legend($("#lg-gantt"), [{ label: "MILP induction week", color: "var(--s1)", type: "box" }, { label: "Threshold-rule week", color: "var(--s4)", type: "box" }, { label: "Engine would fail (ground truth)", color: "var(--crit)", type: "dot" }]);
    const drawGantt = () => {
      const host = $("#c-gantt");
      host.replaceChildren();
      RL.responsive(host, (w) => {
        const rh = 15, m = { top: 26, right: 12, bottom: 8, left: 150 };
        const cw = Math.max(26, (w - m.left - m.right) / H);
        const W = m.left + cw * H + m.right, Hh = m.top + m.bottom + rh * fleet.length;
        const svg = RL.svg(host, W, Hh, "Induction schedule").style("width", W + "px").style("max-width", "none");
        const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
        for (let k = 0; k < H; k++) {
          g.append("text").attr("class", "ann").attr("x", k * cw + cw / 2).attr("y", -10).attr("text-anchor", "middle").text("W" + (k + 1));
          if (k % 2 === 0) g.append("rect").attr("x", k * cw).attr("y", 0).attr("width", cw).attr("height", rh * fleet.length).style("fill", "var(--paper)");
        }
        fleet.forEach((e, i) => {
          const cy = i * rh;
          g.append("text").style("font", "10.5px var(--font-mono)").style("fill", "var(--ink-2)").attr("x", -8).attr("y", cy + rh / 2).attr("dy", "0.35em").attr("text-anchor", "end")
            .text(`${e.id.replace("-test-", " #")} · ${F.f0(e.pred)}`);
          const wk = view === "rule" ? e.week_rule : e.week_milp;
          if (wk >= 0) g.append("rect").attr("x", wk * cw + 3).attr("y", cy + 3).attr("width", cw - 6).attr("height", rh - 6).attr("rx", 3).style("fill", view === "rule" ? "var(--s4)" : "var(--s1)");
          if (view === "truth" && e.truth <= H * cpw) {
            const fw = Math.floor(e.truth / cpw);
            const failsFirst = e.week_milp < 0 || fw < e.week_milp;
            g.append("circle").attr("cx", fw * cw + cw / 2).attr("cy", cy + rh / 2).attr("r", 4).style("fill", failsFirst ? "var(--crit)" : "none").style("stroke", "var(--crit)").style("stroke-width", 1.5);
          }
          g.append("rect").attr("class", "hit").attr("x", -m.left).attr("y", cy).attr("width", m.left + cw * H).attr("height", rh)
            .on("pointermove", (evt) => RL.tip.show(evt, e.id, [{ v: `${F.f0(e.pred)} cycles`, k: `predicted · 80% ${F.f0(e.lo)}–${F.f0(e.hi)}` },
              { v: F.pct0(e.p_fail_h), k: "P(fail within 12 weeks)" }, { v: e.week_milp >= 0 ? "week " + (e.week_milp + 1) : "not scheduled", k: "MILP" },
              { v: e.week_rule >= 0 ? "week " + (e.week_rule + 1) : "not scheduled", k: "rule" }].concat(view === "truth" ? [{ v: e.truth + " cycles", k: "true RUL" }] : [])))
            .on("pointerleave", RL.tip.hide);
        });
      });
    };
    RL.chips($("#sched-view"), (v) => { view = v; drawGantt(); });
    drawGantt();
    const tb = $("#sched-table tbody");
    const names = { none: "Fly to failure", threshold: "Rule: pull at ≤ 20", milp: "MILP schedule" };
    for (const k of ["none", "threshold", "milp"]) {
      const r = I.results[k];
      const tr = tb.insertRow();
      if (k === "milp") tr.className = "hl";
      [names[k], F.int(r.planned), F.usd(r.expected_cost), F.int(r.realized_failures), F.usd(r.realized_cost), r.planned ? F.f1(r.mean_wasted) + " cyc" : "–"].forEach((v, i) => {
        const td = tr.insertCell(); td.textContent = v; if (i) td.className = "n";
      });
    }
  })();

  // ================================================================ 10 DECIDE
  (function decide() {
    const SIM = DATA.simulation;
    const A = SIM.aoa;
    const base = A[0];
    const tb = $("#aoa-table tbody");
    const bestId = DATA.d.recommended_coa;
    for (const c of A) {
      const tr = tb.insertRow();
      if (c.id === bestId) tr.className = "hl";
      const d = el("td"); d.append(el("span", null, c.name), el("span", "sub", c.desc));
      const id = el("td", "mono", c.id);
      tr.append(id, d);
      [`${F.pct1(c.mc.mean)} [${F.pct1(c.mc.p05)}–${F.pct1(c.mc.p95)}]`, F.pct0(c.p_meet_target), F.f1(c.unsched_yr.mean), F.usd(c.cost_yr.mean),
        c === base ? "–" : (c.cost_yr.mean - base.cost_yr.mean >= 0 ? "+" : "−") + F.usd(Math.abs(c.cost_yr.mean - base.cost_yr.mean)).slice(0)].forEach((v) => {
        const td = tr.insertCell(); td.textContent = v; td.className = "n";
      });
    }
    RL.responsive($("#c-aoa"), (w) => {
      const host = $("#c-aoa"), h = 320, m = { top: 20, right: w < 640 ? 56 : 230, bottom: 42, left: 52 }, iw = w - m.left - m.right, ih = h - m.top - m.bottom;
      const svg = RL.svg(host, w, h, "Readiness vs cost by course of action");
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const x = d3.scaleLinear().domain([d3.min(A, (c) => c.cost_yr.mean) - 4, d3.max(A, (c) => c.cost_yr.mean) + 4]).range([0, iw]);
      const y = d3.scaleLinear().domain([d3.min(A, (c) => c.mc.p05) - 0.01, d3.max(A, (c) => c.mc.p95) + 0.01]).range([ih, 0]);
      RL.axes(g, x, y, iw, ih, { xLabel: "Engine sustainment cost per year ($M)", yLabel: "Mission-capable rate", yFormat: d3.format(".0%") });
      g.append("line").attr("class", "ref-line").attr("x1", 0).attr("x2", iw).attr("y1", y(SIM.pbl_target)).attr("y2", y(SIM.pbl_target));
      g.append("text").attr("class", "ann").attr("x", iw).attr("y", y(SIM.pbl_target) - 5).attr("text-anchor", "end").text(`PBL requirement ${F.pct0(SIM.pbl_target)}`);
      // labels: one column per cluster, nudged apart vertically, with leader lines when moved
      const labs = A.map((c) => ({ c, x: x(c.cost_yr.mean), y: y(c.mc.mean), ly: y(c.mc.mean) })).sort((a, b) => a.y - b.y);
      for (let i = 1; i < labs.length; i++) {
        const prev = labs.slice(0, i).filter((p) => Math.abs(p.x - labs[i].x) < 90);
        for (const p of prev) if (labs[i].ly - p.ly < 15) labs[i].ly = p.ly + 15;
      }
      A.forEach((c) => {
        const pred = c.policy === "predictive", col = c.id === bestId ? "var(--accent)" : pred ? "var(--s1)" : "var(--neutral-mark)";
        const L = labs.find((l) => l.c === c);
        g.append("line").attr("x1", x(c.cost_yr.mean)).attr("x2", x(c.cost_yr.mean)).attr("y1", y(c.mc.p05)).attr("y2", y(c.mc.p95)).style("stroke", col).style("stroke-width", 2).style("opacity", 0.5).style("stroke-linecap", "round");
        g.append("circle").attr("cx", x(c.cost_yr.mean)).attr("cy", y(c.mc.mean)).attr("r", 6).style("fill", col).style("stroke", "var(--surface)").style("stroke-width", 2);
        if (Math.abs(L.ly - L.y) > 2) g.append("line").attr("class", "ann-line").attr("x1", L.x + 7).attr("y1", L.y).attr("x2", L.x + 34).attr("y2", L.ly).style("opacity", 0.5);
        g.append("text").attr("class", "dlabel").attr("x", x(c.cost_yr.mean) + 38).attr("y", L.ly).attr("dy", "0.35em").text(w < 640 ? c.id : `${c.id} · ${c.name}`);
        g.append("circle").attr("class", "hit").attr("cx", x(c.cost_yr.mean)).attr("cy", y(c.mc.mean)).attr("r", 16)
          .on("pointermove", (evt) => RL.tip.show(evt, `${c.id} · ${c.name}`, [{ v: F.pct1(c.mc.mean), k: "mean MC" }, { v: F.usd(c.cost_yr.mean), k: "cost / yr" }, { v: F.pct0(c.p_meet_target), k: "P(meets PBL)" }]))
          .on("pointerleave", RL.tip.hide);
      });
    });
    const SE = SIM.sensitivity;
    RL.hbar($("#c-tornado"), {
      data: SE.map((s, i) => ({ label: s.case, value: s.mc_gain * 100, highlight: i === 0, color: i === 0 ? "var(--accent)" : "var(--s1)",
        extra: [{ k: "run to failure", v: F.pct1(s.mc_reactive) }, { k: "predictive", v: F.pct1(s.mc_predictive) }, { k: "cost change $M/yr", v: F.f1(s.cost_delta) }] })),
      labelW: 290, rowH: 26, valueFormat: (v) => (v >= 0 ? "+" : "−") + F.f1(Math.abs(v)) + " pts", valueName: "MC gain", xLabel: "MC gain from predictive removal (points)",
      xDomain: [Math.min(0, d3.min(SE, (s) => s.mc_gain * 100) * 1.15), d3.max(SE, (s) => s.mc_gain * 100) * 1.15],
    });
    RL.table("c-tornado", [{ label: "Changed ground rule", key: "case" }, { label: "MC, run to failure", num: true, value: (r) => F.pct1(r.mc_reactive) },
      { label: "MC, predictive", num: true, value: (r) => F.pct1(r.mc_predictive) }, { label: "Gain (pts)", num: true, value: (r) => F.f1(r.mc_gain * 100) },
      { label: "Cost change ($M/yr)", num: true, value: (r) => F.f1(r.cost_delta) }], SE);
  })();
})();
