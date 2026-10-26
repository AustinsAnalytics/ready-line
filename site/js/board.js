/* Ready Line - squadron status board. Replays one representative simulated year (pipeline/10b_board.py)
 * under either maintenance policy, day by day. */
(function () {
  "use strict";
  const D = window.RL_DATA && window.RL_DATA.board;
  const root = document.getElementById("board");
  if (!D || !root) return;

  const STATUS = {
    F: { label: "MC · flying", color: "var(--good)" },
    M: { label: "MC", color: "var(--good)" },
    E: { label: "Eng change", color: "var(--warn)" },
    S: { label: "No engine", color: "var(--crit)" },
    m: { label: "Maint", color: "var(--serious)" },
    s: { label: "Parts", color: "var(--serious)" },
  };
  const tailsEl = document.getElementById("tails");
  const dayInput = document.getElementById("board-day");
  const dayLbl = document.getElementById("board-daylbl");
  const playBtn = document.getElementById("board-play");
  const ticker = document.getElementById("ticker");
  const spark = document.getElementById("g-spark");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const A = D.aircraft, N = D.days;
  let policy = "predictive", day = reduce ? 180 : 0, playing = !reduce, timer = null, visible = true;

  const tiles = [];
  for (let j = 0; j < A; j++) {
    const t = document.createElement("div");
    t.className = "tail";
    const tn = document.createElement("div"); tn.className = "tn"; tn.textContent = "RL-" + String(j + 1).padStart(2, "0");
    const st = document.createElement("div"); st.className = "st";
    const dot = document.createElement("i"); const stx = document.createElement("span");
    st.append(dot, stx);
    const eng = document.createElement("div"); eng.className = "eng";
    const life = document.createElement("div"); life.className = "life";
    const bar = document.createElement("span"); life.appendChild(bar);
    t.append(tn, st, eng, life);
    tailsEl.appendChild(t);
    tiles.push({ t, stx, eng, bar });
  }

  // precompute daily MC share and cumulative counters per policy
  const pre = {};
  for (const p of ["reactive", "predictive"]) {
    const tr = D[p].trace;
    const mc = tr.map((row) => [...row].filter((c) => c === "F" || c === "M").length / A);
    const sorties = []; let s = 0;
    tr.forEach((row) => { s += [...row].filter((c) => c === "F").length; sorties.push(s); });
    pre[p] = { mc, sorties };
  }

  function sparkline(p, d) {
    const raw = pre[p].mc, w = 230, h = 46;
    const mc = raw.map((_, i) => d3.mean(raw.slice(Math.max(0, i - 13), i + 1)));   // 14-day trailing mean
    const x = (i) => (i / (N - 1)) * w, y = (v) => h - 4 - (v - 0.3) / 0.7 * (h - 8);
    const path = (arr) => arr.map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1)).join("");
    const avg = d3.mean(raw.slice(0, d + 1));
    spark.innerHTML = "";
    const ns = "http://www.w3.org/2000/svg";
    const full = document.createElementNS(ns, "path");
    full.setAttribute("d", path(mc)); full.setAttribute("fill", "none"); full.setAttribute("stroke", "var(--board-rule)"); full.setAttribute("stroke-width", "1.5");
    const now = document.createElementNS(ns, "path");
    now.setAttribute("d", path(mc.slice(0, d + 1))); now.setAttribute("fill", "none"); now.setAttribute("stroke", "var(--good)"); now.setAttribute("stroke-width", "1.5");
    const tgt = document.createElementNS(ns, "line");
    const ty = y(window.RL_DATA.simulation.pbl_target);
    tgt.setAttribute("x1", 0); tgt.setAttribute("x2", w); tgt.setAttribute("y1", ty); tgt.setAttribute("y2", ty);
    tgt.setAttribute("stroke", "var(--board-dim)"); tgt.setAttribute("stroke-dasharray", "2 3"); tgt.setAttribute("stroke-width", "1");
    spark.append(tgt, full, now);
    return avg;
  }

  function render() {
    const P = D[policy];
    const row = P.trace[day], left = P.left[day], est = P.est[day];
    for (let j = 0; j < A; j++) {
      const c = row[j], s = STATUS[c], tl = tiles[j];
      tl.t.style.setProperty("--st", s.color);
      tl.t.classList.toggle("flying", c === "F");
      tl.stx.textContent = s.label;
      if (left[j] < 0) {
        tl.eng.textContent = "No engine installed";
        tl.bar.style.setProperty("--w", "0%");
        tl.bar.classList.remove("low");
      } else {
        const e = est[j];
        tl.eng.textContent = left[j] + " cyc left" + (policy === "predictive" && e < 60 ? " · model " + e : "");
        tl.bar.style.setProperty("--w", Math.min(100, (left[j] / 250) * 100).toFixed(0) + "%");
        tl.bar.classList.toggle("low", left[j] <= 20);
      }
      tl.t.classList.remove("flash", "flash-p");
    }
    const evToday = P.events.filter((e) => e.day === day);
    for (const e of evToday) tiles[e.tail].t.classList.add(e.type === "failure" ? "flash" : "flash-p");
    const upto = P.events.filter((e) => e.day <= day);
    document.getElementById("g-un").textContent = upto.filter((e) => e.type === "failure").length;
    document.getElementById("g-pl").textContent = upto.filter((e) => e.type === "planned").length;
    document.getElementById("g-wait").textContent = [...row].filter((c) => c === "S").length;
    document.getElementById("g-sor").textContent = d3.format(",")(pre[policy].sorties[day]);
    document.getElementById("g-today").textContent = d3.format(".0%")(pre[policy].mc[day]);
    const avg = sparkline(policy, day);
    document.getElementById("g-avg").textContent = "Year to date " + d3.format(".1%")(avg) + " · target " + d3.format(".0%")(window.RL_DATA.simulation.pbl_target);
    ticker.replaceChildren();
    for (const e of upto.slice(-4).reverse()) {
      const div = document.createElement("div");
      div.className = e.type === "failure" ? "f" : "p";
      const tail = "RL-" + String(e.tail + 1).padStart(2, "0");
      div.textContent = "D" + String(e.day + 1).padStart(3, "0") + " " + tail + " " +
        (e.type === "failure" ? "failed, unscheduled" : "pulled, " + e.left + " left");
      ticker.appendChild(div);
    }
    if (!upto.length) { const d0 = document.createElement("div"); d0.textContent = "No engine removals yet"; ticker.appendChild(d0); }
    dayInput.value = day;
    dayLbl.textContent = "Day " + (day + 1);
  }

  function setPlaying(v) {
    playing = v;
    playBtn.setAttribute("aria-label", v ? "Pause" : "Play");
    playBtn.innerHTML = v ? '<svg viewBox="0 0 16 16"><path d="M4 2h3v12H4zM9 2h3v12H9z"/></svg>' : '<svg viewBox="0 0 16 16"><path d="M4 2l10 6-10 6z"/></svg>';
    clearInterval(timer);
    if (v) timer = setInterval(() => { if (!visible) return; day = (day + 1) % N; render(); }, 140);
  }

  root.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    policy = b.dataset.policy;
    root.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    render();
  }));
  playBtn.addEventListener("click", () => setPlaying(!playing));
  dayInput.addEventListener("input", () => { day = +dayInput.value; render(); });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { threshold: 0.1 }).observe(root);
  }
  render();
  setPlaying(playing);
})();
