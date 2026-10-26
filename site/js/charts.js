/* Ready Line - small D3 chart toolkit. Colors come from CSS custom properties so every chart
 * follows the light/dark theme without re-rendering. Text is inserted with textContent only. */
(function () {
  "use strict";
  const RL = (window.RL = window.RL || {});

  // ------------------------------------------------------------------ formatting
  const f = (d) => d3.format(d);
  RL.fmt = {
    pct0: (v) => (v == null ? "–" : f(".0%")(v)),
    pct1: (v) => (v == null ? "–" : f(".1%")(v)),
    f0: (v) => (v == null ? "–" : f(",.0f")(v)),
    f1: (v) => (v == null ? "–" : f(",.1f")(v)),
    f2: (v) => (v == null ? "–" : f(",.2f")(v)),
    f3: (v) => (v == null ? "–" : f(".3f")(v)),
    int: (v) => (v == null ? "–" : f(",")(Math.round(v))),
    usd: (v) => (v == null ? "–" : "$" + f(",.1f")(v) + "M"),
    usdK: (k) => (k >= 1000 ? "$" + f(",.1f")(k / 1000) + "M" : "$" + f(",.0f")(k) + "K"),
    p: (v) => (v < 0.001 ? "< 0.001" : f(".3f")(v)),
    pts: (v) => (v >= 0 ? "+" : "−") + f(".1f")(Math.abs(v * 100)),
  };

  // ------------------------------------------------------------------ tooltip
  const tipEl = () => document.getElementById("tip");
  RL.tip = {
    show(evt, title, rows) {
      const el = tipEl();
      el.replaceChildren();
      if (title) { const t = document.createElement("div"); t.className = "tk"; t.textContent = title; el.appendChild(t); }
      for (const r of rows || []) {
        const d = document.createElement("div");
        d.className = "tr";
        if (r.color) { const i = document.createElement("i"); i.style.background = r.color; d.appendChild(i); }
        const v = document.createElement("span"); v.className = "tv"; v.textContent = r.v; d.appendChild(v);
        if (r.k) { const k = document.createElement("span"); k.className = "tk"; k.textContent = r.k; d.appendChild(k); }
        el.appendChild(d);
      }
      RL.tip.move(evt);
      el.classList.add("on");
    },
    move(evt) {
      const el = tipEl();
      const x = evt.clientX, y = evt.clientY, w = el.offsetWidth, h = el.offsetHeight;
      const left = x + 14 + w > window.innerWidth - 8 ? x - w - 14 : x + 14;
      const top = y + 14 + h > window.innerHeight - 8 ? y - h - 14 : y + 14;
      el.style.left = Math.max(8, left) + "px";
      el.style.top = Math.max(8, top) + "px";
      el.style.transform = "none";
    },
    hide() { tipEl().classList.remove("on"); },
  };

  // ------------------------------------------------------------------ layout
  RL.responsive = function (el, draw) {
    if (!el) return;
    let last = 0, raf = 0;
    const run = () => {
      const w = Math.max(240, Math.floor(el.getBoundingClientRect().width));
      if (w === last) return;
      last = w;
      el.replaceChildren();
      draw(w);
    };
    run();
    if ("ResizeObserver" in window) {
      new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(run); }).observe(el);
    }
    return { redraw() { last = 0; run(); } };
  };

  RL.svg = function (el, w, h, label) {
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", w).attr("height", h)
      .attr("role", "img").style("width", "100%").style("height", "auto");
    if (label) svg.attr("aria-label", label);
    return svg;
  };

  RL.axes = function (g, x, y, iw, ih, o = {}) {
    if (o.grid !== false) {
      const gy = d3.axisLeft(y).ticks(o.yTicks || 5).tickSize(-iw).tickFormat("");
      if (o.yTickValues) gy.tickValues(o.yTickValues);
      g.append("g").attr("class", "gridline").call(gy);
    }
    if (o.xGrid) {
      g.append("g").attr("class", "gridline").attr("transform", `translate(0,${ih})`)
        .call(d3.axisBottom(x).ticks(o.xTicks || 6).tickSize(-ih).tickFormat(""));
    }
    const ax = d3.axisBottom(x).ticks(o.xTicks || 6).tickSizeOuter(0).tickFormat(o.xFormat || null);
    if (o.xTickValues) ax.tickValues(o.xTickValues);
    const ay = d3.axisLeft(y).ticks(o.yTicks || 5).tickSizeOuter(0).tickSize(0).tickPadding(8).tickFormat(o.yFormat || null);
    if (o.yTickValues) ay.tickValues(o.yTickValues);
    const xa = g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(ax);
    const ya = g.append("g").attr("class", "axis").call(ay);
    ya.select(".domain").remove();
    if (o.xLabel) g.append("text").attr("class", "axis-label").attr("x", iw).attr("y", ih + 34).attr("text-anchor", "end").text(o.xLabel);
    if (o.yLabel) g.append("text").attr("class", "axis-label").attr("x", 0).attr("y", -10).attr("text-anchor", "start").text(o.yLabel);
    return { xa, ya };
  };

  RL.legend = function (el, items) {
    if (!el) return;
    el.replaceChildren();
    for (const it of items) {
      const s = document.createElement("span");
      const i = document.createElement("i");
      i.className = it.type || "";
      i.style.background = it.color;
      if (it.type === "band") { i.className = "box"; i.style.opacity = "0.35"; }
      s.appendChild(i);
      s.appendChild(document.createTextNode(it.label));
      el.appendChild(s);
    }
  };

  // ------------------------------------------------------------------ table views
  const tables = {};
  RL.table = function (id, columns, rows) { tables[id] = { columns, rows }; };
  document.addEventListener("click", (e) => {
    const b = e.target.closest(".tbl-toggle");
    if (!b) return;
    const id = b.dataset.table;
    const fig = b.closest("figure");
    let wrap = fig.querySelector(`.tbl-wrap[data-for="${id}"]`);
    if (wrap) { wrap.remove(); b.textContent = "Table"; return; }
    const spec = tables[id];
    if (!spec) return;
    wrap = document.createElement("div");
    wrap.className = "tbl-wrap";
    wrap.dataset.for = id;
    const t = document.createElement("table");
    t.className = "t";
    const thead = t.createTHead().insertRow();
    for (const c of spec.columns) {
      const th = document.createElement("th");
      th.textContent = c.label;
      if (c.num) th.className = "n";
      thead.appendChild(th);
    }
    const tb = t.createTBody();
    for (const r of spec.rows) {
      const tr = tb.insertRow();
      for (const c of spec.columns) {
        const td = tr.insertCell();
        const v = typeof c.value === "function" ? c.value(r) : r[c.key];
        td.textContent = v;
        if (c.num) td.className = "n";
      }
    }
    wrap.appendChild(t);
    fig.appendChild(wrap);
    b.textContent = "Hide table";
  });

  // ------------------------------------------------------------------ chips
  RL.chips = function (groupEl, onChange) {
    if (!groupEl) return;
    groupEl.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      for (const x of groupEl.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
      onChange(b.dataset.v);
    });
  };

  // ------------------------------------------------------------------ line chart with crosshair
  /* series: [{name, color, values:[{x,y}], width?, dash?, area?}], bands: [{color, values:[{x,lo,hi}]}] */
  RL.lineChart = function (el, o) {
    return RL.responsive(el, (w) => {
      const h = o.height || 300;
      const m = Object.assign({ top: 22, right: 18, bottom: 40, left: 46 }, o.margin || {});
      const iw = w - m.left - m.right, ih = h - m.top - m.bottom;
      const svg = RL.svg(el, w, h, o.label);
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const allX = o.series.flatMap((s) => s.values.map((d) => d.x)).concat((o.bands || []).flatMap((b) => b.values.map((d) => d.x)));
      const allY = o.series.flatMap((s) => s.values.map((d) => d.y)).concat((o.bands || []).flatMap((b) => b.values.flatMap((d) => [d.lo, d.hi])));
      const x = (o.xTime ? d3.scaleTime() : d3.scaleLinear()).domain(o.xDomain || d3.extent(allX)).range([0, iw]);
      const yd = o.yDomain || [Math.min(0, d3.min(allY)), d3.max(allY) * 1.05];
      const y = d3.scaleLinear().domain(yd).nice().range([ih, 0]);
      RL.axes(g, x, y, iw, ih, o);
      for (const r of o.refs || []) {
        if (r.y != null) {
          g.append("line").attr("class", "ref-line").attr("x1", 0).attr("x2", iw).attr("y1", y(r.y)).attr("y2", y(r.y));
          g.append("text").attr("class", "ann").attr("x", iw).attr("y", y(r.y) - 5).attr("text-anchor", "end").text(r.label);
        }
        if (r.x != null) {
          g.append("line").attr("class", "ref-line").attr("y1", 0).attr("y2", ih).attr("x1", x(r.x)).attr("x2", x(r.x));
          g.append("text").attr("class", "ann").attr("x", x(r.x) + 5).attr("y", 10).text(r.label);
        }
      }
      for (const b of o.bands || []) {
        g.append("path").datum(b.values).style("fill", b.color).style("opacity", b.opacity || 0.14)
          .attr("d", d3.area().x((d) => x(d.x)).y0((d) => y(d.lo)).y1((d) => y(d.hi)).curve(d3.curveMonotoneX));
      }
      for (const s of o.series) {
        if (s.area) {
          g.append("path").datum(s.values).style("fill", s.color).style("opacity", 0.1)
            .attr("d", d3.area().x((d) => x(d.x)).y0(ih).y1((d) => y(d.y)).curve(d3.curveMonotoneX));
        }
        g.append("path").datum(s.values.filter((d) => d.y != null)).style("fill", "none").style("stroke", s.color)
          .style("stroke-width", s.width || 2).style("stroke-linejoin", "round").style("stroke-linecap", "round")
          .style("stroke-dasharray", s.dash || null).style("opacity", s.opacity || 1)
          .attr("d", d3.line().x((d) => x(d.x)).y((d) => y(d.y)).curve(s.step ? d3.curveStepAfter : d3.curveMonotoneX));
        if (s.endLabel) {
          const last = s.values[s.values.length - 1];
          g.append("circle").attr("cx", x(last.x)).attr("cy", y(last.y)).attr("r", 4).style("fill", s.color)
            .style("stroke", "var(--surface)").style("stroke-width", 2);
        }
      }
      if (o.after) o.after(g, x, y, iw, ih);
      if (o.crosshair !== false) {
        const xs = Array.from(new Set(allX.map((v) => +v))).sort((a, b) => a - b);
        const cross = g.append("line").attr("class", "crosshair").attr("y1", 0).attr("y2", ih).style("opacity", 0);
        const dots = g.append("g");
        g.append("rect").attr("class", "hit").attr("width", iw).attr("height", ih)
          .on("pointermove", (evt) => {
            const [px] = d3.pointer(evt);
            const xv = +x.invert(px);
            const i = d3.bisectCenter(xs, xv);
            const xx = xs[Math.max(0, Math.min(xs.length - 1, i))];
            cross.attr("x1", x(xx)).attr("x2", x(xx)).style("opacity", 1);
            dots.selectAll("*").remove();
            const rows = [];
            for (const s of o.series) {
              if (s.noTip) continue;
              const d = s.values.find((v) => +v.x === xx);
              if (!d || d.y == null) continue;
              dots.append("circle").attr("cx", x(d.x)).attr("cy", y(d.y)).attr("r", 4).style("fill", s.color)
                .style("stroke", "var(--surface)").style("stroke-width", 2);
              rows.push({ k: s.name, v: (o.tipFormat || RL.fmt.f1)(d.y), color: s.color });
            }
            for (const b of o.bands || []) {
              if (!b.name) continue;
              const d = b.values.find((v) => +v.x === xx);
              if (d) rows.push({ k: b.name, v: `${(o.tipFormat || RL.fmt.f1)(d.lo)} – ${(o.tipFormat || RL.fmt.f1)(d.hi)}`, color: b.color });
            }
            RL.tip.show(evt, (o.xTipFormat || ((v) => v))(o.xTime ? new Date(xx) : xx), rows);
          })
          .on("pointerleave", () => { cross.style("opacity", 0); dots.selectAll("*").remove(); RL.tip.hide(); });
      }
    });
  };

  // ------------------------------------------------------------------ horizontal bars
  /* data: [{label, value, color?, note?, highlight?}] */
  RL.hbar = function (el, o) {
    return RL.responsive(el, (w) => {
      const rowH = o.rowH || 26;
      const m = Object.assign({ top: 8, right: 56, bottom: 30, left: Math.min(o.labelW || 170, w * 0.42) }, o.margin || {});
      const h = m.top + m.bottom + rowH * o.data.length;
      const iw = w - m.left - m.right, ih = rowH * o.data.length;
      const svg = RL.svg(el, w, h, o.label);
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const x = d3.scaleLinear().domain(o.xDomain || [0, d3.max(o.data, (d) => d.value) * 1.04]).nice().range([0, iw]);
      const y = d3.scaleBand().domain(o.data.map((d, i) => i)).range([0, ih]).padding(0.28);
      g.append("g").attr("class", "gridline").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(o.xTicks || 5).tickSize(-ih).tickFormat(""));
      g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(o.xTicks || 5).tickSizeOuter(0).tickFormat(o.xFormat || null));
      const bh = Math.min(y.bandwidth(), 18);
      const rows = g.selectAll("g.row").data(o.data).join("g").attr("class", "row").attr("transform", (d, i) => `translate(0,${y(i) + (y.bandwidth() - bh) / 2})`);
      const x0 = x(Math.max(x.domain()[0], 0));
      rows.append("path").style("fill", (d) => d.color || o.color || "var(--s1)").style("opacity", (d) => (o.dimOthers && !d.highlight ? 0.45 : 1))
        .attr("d", (d) => {
          // square at the baseline, 4px rounded at the data end - in either direction
          const xv = x(d.value), wv = Math.max(1, Math.abs(xv - x0)), r = Math.min(4, wv / 2, bh / 2);
          if (xv >= x0) return `M${x0},0H${x0 + wv - r}Q${x0 + wv},0 ${x0 + wv},${r}V${bh - r}Q${x0 + wv},${bh} ${x0 + wv - r},${bh}H${x0}Z`;
          return `M${x0},0H${x0 - wv + r}Q${x0 - wv},0 ${x0 - wv},${r}V${bh - r}Q${x0 - wv},${bh} ${x0 - wv + r},${bh}H${x0}Z`;
        });
      if (x.domain()[0] < 0) g.append("line").attr("x1", x0).attr("x2", x0).attr("y1", 0).attr("y2", ih).style("stroke", "var(--axis)");
      rows.append("text").attr("class", (d) => "dlabel" + (d.highlight || !o.dimOthers ? "" : " dim")).attr("x", -8).attr("y", bh / 2).attr("dy", "0.35em").attr("text-anchor", "end")
        .text((d) => d.label).each(function () {
          const self = d3.select(this);
          let t = self.text();
          while (this.getComputedTextLength() > m.left - 12 && t.length > 4) { t = t.slice(0, -2); self.text(t + "…"); }
        });
      rows.append("text").attr("class", "dlabel dim").attr("x", (d) => Math.max(x(d.value), x0) + 6).attr("y", bh / 2).attr("dy", "0.35em")
        .text((d) => (o.valueFormat || RL.fmt.f1)(d.value));
      rows.append("rect").attr("class", "hit").attr("x", -m.left).attr("y", -4).attr("width", iw + m.left).attr("height", bh + 8)
        .style("cursor", o.onClick ? "pointer" : null)
        .on("pointermove", (evt, d) => RL.tip.show(evt, d.label, [{ v: (o.valueFormat || RL.fmt.f1)(d.value), k: o.valueName || "" }].concat(d.extra || [])))
        .on("pointerleave", RL.tip.hide)
        .on("click", (evt, d) => o.onClick && o.onClick(d));
      if (o.xLabel) g.append("text").attr("class", "axis-label").attr("x", iw).attr("y", ih + 28).attr("text-anchor", "end").text(o.xLabel);
      if (o.after) o.after(g, x, y, iw, ih);
    });
  };

  // ------------------------------------------------------------------ vertical columns
  RL.columns = function (el, o) {
    return RL.responsive(el, (w) => {
      const h = o.height || 260;
      const m = Object.assign({ top: 22, right: 12, bottom: 40, left: 44 }, o.margin || {});
      const iw = w - m.left - m.right, ih = h - m.top - m.bottom;
      const svg = RL.svg(el, w, h, o.label);
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const x = d3.scaleBand().domain(o.data.map((d) => d.label)).range([0, iw]).padding(0.3);
      const y = d3.scaleLinear().domain(o.yDomain || [0, d3.max(o.data, (d) => d.value) * 1.1]).nice().range([ih, 0]);
      RL.axes(g, x, y, iw, ih, { yFormat: o.yFormat, yLabel: o.yLabel, xLabel: o.xLabel });
      const bw = Math.min(x.bandwidth(), 34);
      const cols = g.selectAll("g.col").data(o.data).join("g").attr("transform", (d) => `translate(${x(d.label) + (x.bandwidth() - bw) / 2},0)`);
      cols.append("path").style("fill", (d) => d.color || o.color || "var(--s1)").attr("d", (d) => {
        const top = y(d.value), hh = ih - top, r = Math.min(4, hh / 2, bw / 2);
        return `M0,${ih}V${top + r}Q0,${top} ${r},${top}H${bw - r}Q${bw},${top} ${bw},${top + r}V${ih}Z`;
      });
      cols.append("text").attr("class", "dlabel dim").attr("x", bw / 2).attr("y", (d) => y(d.value) - 6).attr("text-anchor", "middle")
        .text((d) => (o.valueFormat || RL.fmt.f1)(d.value));
      cols.append("rect").attr("class", "hit").attr("x", -4).attr("width", bw + 8).attr("y", 0).attr("height", ih)
        .on("pointermove", (evt, d) => RL.tip.show(evt, d.label, [{ v: (o.valueFormat || RL.fmt.f1)(d.value), k: o.valueName || "" }].concat(d.extra || [])))
        .on("pointerleave", RL.tip.hide);
      if (o.after) o.after(g, x, y, iw, ih);
    });
  };

  // ------------------------------------------------------------------ scatter with voronoi hover
  RL.scatter = function (el, o) {
    return RL.responsive(el, (w) => {
      const h = o.height || 300;
      const m = Object.assign({ top: 22, right: 18, bottom: 40, left: 46 }, o.margin || {});
      const iw = w - m.left - m.right, ih = h - m.top - m.bottom;
      const svg = RL.svg(el, w, h, o.label);
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const x = (o.xLog ? d3.scaleLog() : d3.scaleLinear()).domain(o.xDomain || d3.extent(o.data, (d) => d.x)).nice().range([0, iw]);
      const y = (o.yLog ? d3.scaleLog() : d3.scaleLinear()).domain(o.yDomain || d3.extent(o.data, (d) => d.y)).nice().range([ih, 0]);
      RL.axes(g, x, y, iw, ih, o);
      if (o.before) o.before(g, x, y, iw, ih);
      g.append("g").selectAll("circle").data(o.data).join("circle")
        .attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("r", o.r ?? 4)
        .style("fill", (d) => d.color || "var(--s1)").style("opacity", o.opacity || 0.85)
        .style("stroke", "var(--surface)").style("stroke-width", o.ring === false ? 0 : 1.5);
      if (o.after) o.after(g, x, y, iw, ih);
      if (o.tip) {
        const delaunay = d3.Delaunay.from(o.data, (d) => x(d.x), (d) => y(d.y));
        const ring = g.append("circle").attr("r", (o.r ?? 4) + 3).style("fill", "none").style("stroke", "var(--ink)").style("stroke-width", 1.5).style("opacity", 0);
        g.append("rect").attr("class", "hit").attr("width", iw).attr("height", ih)
          .on("pointermove", (evt) => {
            const [px, py] = d3.pointer(evt);
            const i = delaunay.find(px, py);
            const d = o.data[i];
            if (!d || Math.hypot(x(d.x) - px, y(d.y) - py) > 40) { ring.style("opacity", 0); RL.tip.hide(); return; }
            ring.attr("cx", x(d.x)).attr("cy", y(d.y)).style("opacity", 1);
            const t = o.tip(d);
            RL.tip.show(evt, t.title, t.rows);
          })
          .on("pointerleave", () => { ring.style("opacity", 0); RL.tip.hide(); });
      }
    });
  };

  // ------------------------------------------------------------------ heatmap
  RL.heatmap = function (el, o) {
    return RL.responsive(el, (w) => {
      const m = Object.assign({ top: 10, right: 10, bottom: 40, left: 120 }, o.margin || {});
      const iw = w - m.left - m.right;
      const cell = Math.min(o.maxCell || 44, iw / o.cols.length);
      const ih = cell * o.rows.length;
      const h = ih + m.top + m.bottom;
      const svg = RL.svg(el, w, h, o.label);
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
      const color = o.color;
      o.rows.forEach((r, i) => {
        o.cols.forEach((c, j) => {
          const v = o.value(i, j);
          const cx = j * cell, cy = i * cell;
          g.append("rect").attr("x", cx + 1).attr("y", cy + 1).attr("width", cell - 2).attr("height", cell - 2).attr("rx", 3)
            .style("fill", v == null ? "var(--surface-2)" : color(v))
            .style("fill-opacity", v == null || !o.opacity ? 1 : o.opacity(v))
            .on("pointermove", (evt) => RL.tip.show(evt, o.tipTitle ? o.tipTitle(i, j) : `${r} · ${c}`, [{ v: (o.format || RL.fmt.f2)(v), k: o.valueName || "" }]))
            .on("pointerleave", RL.tip.hide);
          if (o.cellLabel && cell >= 26 && v != null) {
            const txt = o.cellLabel(v, i, j);
            if (txt) {
              g.append("text").attr("x", cx + cell / 2).attr("y", cy + cell / 2).attr("dy", "0.35em").attr("text-anchor", "middle")
                .style("font", "500 10px var(--font-mono)").style("pointer-events", "none")
                .style("fill", o.labelInk ? o.labelInk(v) : "var(--ink)").text(txt);
            }
          }
        });
        g.append("text").attr("class", "dlabel dim").attr("x", -8).attr("y", i * cell + cell / 2).attr("dy", "0.35em").attr("text-anchor", "end").text(r);
      });
      o.cols.forEach((c, j) => {
        g.append("text").attr("class", "axis").style("font", "11px var(--font-mono)").style("fill", "var(--muted)")
          .attr("x", j * cell + cell / 2).attr("y", ih + 16).attr("text-anchor", "middle").text(c);
      });
      if (o.xLabel) g.append("text").attr("class", "axis-label").attr("x", (cell * o.cols.length) / 2).attr("y", ih + 34).attr("text-anchor", "middle").text(o.xLabel);
      if (o.yLabel) g.append("text").attr("class", "axis-label").attr("x", -m.left + 4).attr("y", -2).text(o.yLabel);
    });
  };
})();
