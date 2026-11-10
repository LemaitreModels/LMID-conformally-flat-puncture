/* Interactive single-patch grid of the conformally-flat-puncture solver. Mounts into every element
   with class "lm-abt". No dependencies; colours from the Lemaitre tokens.

   The chart is that of Ansorg, Brügmann and Tichy (2004): prolate-spheroidal coordinates (A, B, phi)
   with foci at the punctures z = +-b,
     rho / b = 2 A sqrt(1 - B^2) / (1 - A^2),   z / b = (1 + A^2) B / (1 - A^2),
   so curves of constant A are confocal ellipses and curves of constant B confocal hyperbolas.
   A = 0 is the inner axis segment |z| <= b, B = +-1 the outer axis |z| >= b, A = 1 spatial infinity,
   and the corners (A, B) = (0, +-1) are the punctures. The angle is represented in s, with
     B = 2 s / (1 + s^2),   sqrt(1 - B^2) = (1 - s^2) / (1 + s^2).
   Nodes: K Chebyshev-Gauss nodes A_i = sin^2(pi (i + 1/2) / (2K)) and N Gauss-Legendre nodes in s
   (or, for comparison, Gauss-Legendre in B itself, the original grid).

   Two panels: the computational square (A, s) and the meridian plane (z, rho), mirrored to rho < 0,
   which is the half-plane phi + pi. A coordinate line sweeps through both, first in A, then in s;
   hovering over either panel shows a point and its two coordinate lines in both, and the sweep
   resumes after a pause. */
(function () {
  "use strict";

  function token(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  // Gauss-Legendre nodes on (-1, 1), ascending, by Newton on P_N
  function gaussLegendre(N) {
    const x = new Array(N);
    for (let j = 0; j < N; j++) {
      let t = Math.cos((Math.PI * (j + 0.75)) / (N + 0.5));
      for (let k = 0; k < 100; k++) {
        let p0 = 1, p1 = t;
        for (let n = 2; n <= N; n++) { const p2 = ((2 * n - 1) * t * p1 - (n - 1) * p0) / n; p0 = p1; p1 = p2; }
        const dp = (N * (t * p1 - p0)) / (t * t - 1), dt = p1 / dp;
        t -= dt;
        if (Math.abs(dt) < 1e-15) break;
      }
      x[j] = t;
    }
    return x.sort((a, b) => a - b);
  }
  const radialNodes = (K) => Array.from({ length: K }, (_, i) => Math.sin((Math.PI * (i + 0.5)) / (2 * K)) ** 2);
  const sOfB = (B) => B / (1 + Math.sqrt(Math.max(0, 1 - B * B)));
  const BofS = (s) => (2 * s) / (1 + s * s);
  // (A, s) -> (z, rho) in units of b
  function toPhys(A, s) {
    const d = 1 - A * A, sq = (1 - s * s) / (1 + s * s);
    return [((1 + A * A) * BofS(s)) / d, (2 * A * sq) / d];
  }
  // (z, rho) -> (A, s), from the distances to the two punctures
  function toChart(z, rho) {
    rho = Math.abs(rho);
    const r1 = Math.hypot(rho, z - 1), r2 = Math.hypot(rho, z + 1);
    const xi = (r1 + r2) / 2, B = Math.max(-1, Math.min(1, (2 * z) / (r1 + r2)));
    return [Math.sqrt(Math.max(0, (xi - 1) / (xi + 1))), sOfB(B)];
  }

  function mount(root) {
    root.innerHTML = "";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "The computational square in the coordinates A and s, and its image in the meridian plane: confocal ellipses and hyperbolas with the two punctures at their foci");
    const controls = document.createElement("div");
    controls.className = "lm-abt-controls";
    root.append(canvas, controls);

    const S = { K: 14, N: 12, view: 0.35, legB: false };
    const viewL = () => 1.2 * Math.pow(50, S.view);           // half-width of the meridian view, in b

    function slider(label, min, max, step, key, fmt) {
      const lab = document.createElement("label");
      const name = document.createElement("span"); name.innerHTML = label;
      const input = document.createElement("input");
      Object.assign(input, { type: "range", min, max, step, value: S[key] });
      const out = document.createElement("output"); out.textContent = fmt(S[key]);
      input.addEventListener("input", () => { S[key] = +input.value; out.textContent = fmt(S[key]); });
      lab.append(name, input, out); controls.append(lab);
    }
    slider("radial nodes", 4, 45, 1, "K", (v) => String(v));
    slider("angular nodes", 4, 40, 1, "N", (v) => String(v));
    slider("view", 0, 1, "any", "view", () => `±${viewL().toFixed(viewL() < 10 ? 1 : 0)} b`);
    const lab = document.createElement("label");
    const box = document.createElement("input"); box.type = "checkbox";
    box.addEventListener("change", () => { S.legB = box.checked; });
    const txt = document.createElement("span"); txt.innerHTML = "Legendre nodes in <i>B</i>, not <i>s</i>";
    lab.append(box, txt); controls.append(lab);

    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1, P = {}, Q = {}, wide = true;
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = root.clientWidth;
      wide = W >= 620;
      if (wide) {
        const h = Math.round(Math.min(W * 0.46, 420)), wq = Math.round(Math.min(W * 0.4, h + 40));
        Q = { x: 0, y: 0, w: wq, h }; P = { x: wq, y: 0, w: W - wq, h }; H = h + 64;
      } else {
        const h1 = Math.round(W * 0.72), h2 = Math.round(W * 0.85);
        Q = { x: 0, y: 0, w: W, h: h1 }; P = { x: 0, y: h1, w: W, h: h2 }; H = h1 + h2 + 120;
      }
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
    }
    new ResizeObserver(resize).observe(root);
    resize();

    // geometry of the two panels, recomputed each frame
    function geom(fs) {
      const side = Math.min(Q.w - fs * 4.2, Q.h - fs * 4.2) * (wide ? 1 : 0.92);
      const qx = Q.x + (Q.w - side) / 2 + fs * 0.8, qy = Q.y + fs * 1.6;
      const sq = { l: qx, r: qx + side * 0.92, t: qy, b: qy + side * 0.92 };
      const pv = { l: P.x + fs * 2.6, r: P.x + P.w - fs * 0.6, t: P.y + fs * (wide ? 1.8 : 3.0), b: P.y + P.h - fs * 2.2 };
      const L = viewL(), scale = (pv.r - pv.l) / (2 * L), cx = (pv.l + pv.r) / 2, cy = (pv.t + pv.b) / 2;
      return {
        sq, pv, L, scale,
        QX: (A) => sq.l + A * (sq.r - sq.l), QY: (s) => sq.b - ((s + 1) / 2) * (sq.b - sq.t),
        PX: (z) => cx + z * scale, PY: (r) => cy - r * scale,
        invQ: (x, y) => [(x - sq.l) / (sq.r - sq.l), -1 + (2 * (sq.b - y)) / (sq.b - sq.t)],
        invP: (x, y) => [(x - cx) / scale, (cy - y) / scale],
      };
    }

    const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let phase = 0.18, last = performance.now(), manualUntil = 0, probe = null;
    const PERIOD = 16000, PAUSE = 2500;
    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const fs = parseFloat(getComputedStyle(root).fontSize) || 16, g = geom(fs);
      if (x >= g.sq.l && x <= g.sq.r && y >= g.sq.t && y <= g.sq.b) {
        const [A, s] = g.invQ(x, y); probe = { A: Math.min(A, 0.995), s, flip: false };
      } else if (x >= g.pv.l && x <= g.pv.r && y >= g.pv.t && y <= g.pv.b) {
        const [z, rho] = g.invP(x, y), [A, s] = toChart(z, rho); probe = { A: Math.min(A, 0.995), s, flip: rho < 0 };
      } else return;
      manualUntil = performance.now() + PAUSE;
    });

    function frame(now) {
      const ink = token(root, "--lm-ink", "#1f1e1c"), muted = token(root, "--lm-muted", "#6c6a63");
      const rule = token(root, "--lm-rule", "#dcd8cc"), blue = token(root, "--lm-accent", "#2c5a85");
      const red = token(root, "--lm-accent2", "#b04632"), soft = token(root, "--lm-soft", "#f0eee6"), paper = token(root, "--lm-paper", "#faf9f5");
      const ftext = token(root, "--lm-font-text", "system-ui, sans-serif"), fmath = token(root, "--lm-font-math", "serif");
      const fs = parseFloat(getComputedStyle(root).fontSize) || 16, small = Math.round(fs * 0.82);
      const dt = now - last; last = now;
      const manual = now < manualUntil && probe;
      if (!still && !manual) phase = (phase + dt / PERIOD) % 1;
      const g = geom(fs), { sq, pv, L } = g;
      const As = radialNodes(S.K), Ss = S.legB ? gaussLegendre(S.N).map(sOfB) : gaussLegendre(S.N);

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = "round"; ctx.lineJoin = "round";

      // the highlighted lines: the probe's two coordinate lines, or the sweep
      let hl;
      if (manual) hl = { A: probe.A, s: probe.s, both: true, flip: probe.flip };
      else if (still) hl = { A: 0.5, s: 0.35, both: true, flip: false };
      else if (phase < 0.5) { const u = phase / 0.5; hl = { A: 0.985 * Math.sin((Math.PI / 2) * u) ** 2, s: null }; }
      else { const u = (phase - 0.5) / 0.5; hl = { A: null, s: -Math.cos(Math.PI * u) * 0.999 }; }

      // ---- computational square --------------------------------------------------------------------
      ctx.fillStyle = soft; ctx.fillRect(sq.l, sq.t, sq.r - sq.l, sq.b - sq.t);
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      for (const A of As) { ctx.beginPath(); ctx.moveTo(g.QX(A), sq.t); ctx.lineTo(g.QX(A), sq.b); ctx.stroke(); }
      for (const s of Ss) { ctx.beginPath(); ctx.moveTo(sq.l, g.QY(s)); ctx.lineTo(sq.r, g.QY(s)); ctx.stroke(); }
      // edges
      ctx.lineWidth = 3;
      ctx.strokeStyle = red; ctx.beginPath(); ctx.moveTo(sq.l, sq.t); ctx.lineTo(sq.l, sq.b); ctx.stroke();
      ctx.strokeStyle = blue; ctx.beginPath(); ctx.moveTo(sq.l, sq.t); ctx.lineTo(sq.r, sq.t); ctx.moveTo(sq.l, sq.b); ctx.lineTo(sq.r, sq.b); ctx.stroke();
      ctx.strokeStyle = muted; ctx.lineWidth = 2; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(sq.r, sq.t); ctx.lineTo(sq.r, sq.b); ctx.stroke(); ctx.setLineDash([]);
      // highlighted lines
      ctx.strokeStyle = ink; ctx.lineWidth = 2;
      if (hl.A !== null) { ctx.beginPath(); ctx.moveTo(g.QX(hl.A), sq.t); ctx.lineTo(g.QX(hl.A), sq.b); ctx.stroke(); }
      if (hl.s !== null) { ctx.beginPath(); ctx.moveTo(sq.l, g.QY(hl.s)); ctx.lineTo(sq.r, g.QY(hl.s)); ctx.stroke(); }
      // nodes
      ctx.fillStyle = ink;
      const rq = S.K * S.N > 900 ? 1.4 : 2.2;
      for (const A of As) for (const s of Ss) { ctx.beginPath(); ctx.arc(g.QX(A), g.QY(s), rq, 0, 2 * Math.PI); ctx.fill(); }
      // punctures
      for (const s of [-1, 1]) { ctx.fillStyle = ink; ctx.beginPath(); ctx.arc(sq.l, g.QY(s), 5, 0, 2 * Math.PI); ctx.fill(); }
      if (hl.both) { ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(g.QX(hl.A), g.QY(hl.s), 6, 0, 2 * Math.PI); ctx.stroke(); }
      // labels
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`; ctx.textAlign = "center";
      ctx.fillText("0", sq.l, sq.b + small + 4); ctx.fillText("1", sq.r, sq.b + small + 4);
      ctx.textAlign = "right";
      ctx.fillText("1", sq.l - 9, sq.t + small * 0.35); ctx.fillText("0", sq.l - 9, (sq.t + sq.b) / 2 + small * 0.35); ctx.fillText("−1", sq.l - 9, sq.b + small * 0.35);
      ctx.font = `${fs}px ${fmath}`; ctx.fillStyle = ink; ctx.textAlign = "center";
      ctx.fillText("𝐴", (sq.l + sq.r) / 2, sq.b + small + fs + 2);
      ctx.fillText("𝑠", sq.l - fs * 2.7, (sq.t + sq.b) / 2 + fs * 0.3);
      ctx.font = `${small}px ${ftext}`; ctx.fillStyle = muted; ctx.textAlign = "left";
      ctx.fillText("computational square", sq.l, sq.t - small * 0.7);

      // ---- meridian plane -----------------------------------------------------------------------------
      ctx.save(); ctx.beginPath(); ctx.rect(pv.l, pv.t, pv.r - pv.l, pv.b - pv.t); ctx.clip();
      const Lh = Math.max(L, (pv.b - pv.t) / (2 * g.scale)) * 1.6;
      function ellipse(A) {
        const d = 1 - A * A, a = (1 + A * A) / d, c = (2 * A) / d;
        if (c > Lh && a > Lh) return false;
        ctx.beginPath();
        for (let k = 0; k <= 240; k++) { const t = (2 * Math.PI * k) / 240, x = g.PX(a * Math.cos(t)), y = g.PY(c * Math.sin(t)); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.stroke(); return true;
      }
      function hyperbola(s) {
        const B = BofS(s), sq2 = (1 - s * s) / (1 + s * s);
        const mz = Math.abs(B) > 1e-12 ? Math.acosh(Math.max(1, Lh / Math.abs(B))) : Infinity;
        const mr = sq2 > 1e-12 ? Math.asinh(Lh / sq2) : Infinity, mu = Math.min(mz, mr, 12);
        for (const sg of [1, -1]) {
          ctx.beginPath();
          for (let k = 0; k <= 160; k++) { const m = (mu * k) / 160, x = g.PX(B * Math.cosh(m)), y = g.PY(sg * sq2 * Math.sinh(m)); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
          ctx.stroke();
        }
      }
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      for (const A of As) ellipse(A);
      for (const s of Ss) hyperbola(s);
      // the axis: inner segment red, outer axis blue
      ctx.lineWidth = 3;
      ctx.strokeStyle = blue; ctx.beginPath(); ctx.moveTo(g.PX(-Lh), g.PY(0)); ctx.lineTo(g.PX(-1), g.PY(0)); ctx.moveTo(g.PX(1), g.PY(0)); ctx.lineTo(g.PX(Lh), g.PY(0)); ctx.stroke();
      ctx.strokeStyle = red; ctx.beginPath(); ctx.moveTo(g.PX(-1), g.PY(0)); ctx.lineTo(g.PX(1), g.PY(0)); ctx.stroke();
      // highlighted lines
      ctx.strokeStyle = ink; ctx.lineWidth = 2;
      if (hl.A !== null) ellipse(hl.A);
      if (hl.s !== null) hyperbola(hl.s);
      // nodes, both half-planes
      ctx.fillStyle = ink;
      const rp = S.K * S.N > 900 ? 1.3 : 2;
      for (const A of As) for (const s of Ss) {
        const [z, r] = toPhys(A, s);
        if (Math.abs(z) > Lh || r > Lh) continue;
        for (const sg of [1, -1]) { ctx.beginPath(); ctx.arc(g.PX(z), g.PY(sg * r), rp, 0, 2 * Math.PI); ctx.fill(); }
      }
      for (const z of [-1, 1]) { ctx.fillStyle = ink; ctx.beginPath(); ctx.arc(g.PX(z), g.PY(0), 5, 0, 2 * Math.PI); ctx.fill(); }
      if (hl.both) {
        const [z, r] = toPhys(hl.A, hl.s);
        ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(g.PX(z), g.PY(hl.flip ? -r : r), 6, 0, 2 * Math.PI); ctx.stroke();
      }
      ctx.restore();
      // frame and labels
      ctx.strokeStyle = rule; ctx.lineWidth = 1; ctx.strokeRect(pv.l, pv.t, pv.r - pv.l, pv.b - pv.t);
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`; ctx.textAlign = "left";
      ctx.fillText("meridian plane", pv.l, pv.t - small * (wide ? 0.7 : 2.1));
      ctx.textAlign = "right";
      for (const [t, y] of [["𝜙 = 0", pv.t + small + 2], ["𝜙 = 𝜋", pv.b - 6]]) {
        const tw = ctx.measureText(t).width; ctx.fillStyle = paper; ctx.fillRect(pv.r - 9 - tw, y - small, tw + 6, small + 4); ctx.fillStyle = muted; ctx.fillText(t, pv.r - 6, y);
      }
      ctx.textAlign = "center";
      const tick = L < 3 ? 1 : L < 8 ? 2 : L < 20 ? 5 : L < 45 ? 10 : 20;
      for (let z = -Math.floor(L / tick) * tick; z <= L + 1e-9; z += tick) ctx.fillText(String(z).replace("-", "−"), g.PX(z), pv.b + small + 4);
      ctx.font = `${fs}px ${fmath}`; ctx.fillStyle = ink;
      ctx.fillText("𝑧/𝑏", (pv.l + pv.r) / 2, pv.b + small + fs + 2);
      ctx.save(); ctx.translate(pv.l - fs * 0.9, (pv.t + pv.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("𝜌/𝑏", 0, 0); ctx.restore();

      // readout of the highlighted coordinates
      ctx.font = `${small}px ${ftext}`; ctx.fillStyle = ink; ctx.textAlign = wide ? "right" : "left";
      let ro;
      if (hl.both) { const [z, r] = toPhys(hl.A, hl.s); ro = `𝐴 = ${hl.A.toFixed(3)},  𝑠 = ${hl.s.toFixed(3)}   →   𝑧 = ${z.toFixed(2)} 𝑏,  𝜌 = ${r.toFixed(2)} 𝑏`; }
      else if (hl.A !== null) ro = `𝐴 = ${hl.A.toFixed(3)}: an ellipse with foci at the punctures`;
      else ro = `𝑠 = ${hl.s.toFixed(3)}: a hyperbola with the same foci`;
      ctx.fillText(ro, wide ? pv.r : pv.l, pv.t - small * 0.7);

      // ---- legend and counts ----------------------------------------------------------------------------
      const items = [
        [red, [], "inner axis, |𝑧| ≤ 𝑏  (𝐴 = 0)"],
        [blue, [], "outer axis, |𝑧| ≥ 𝑏  (𝑠 = ±1)"],
        [muted, [5, 5], "spatial infinity  (𝐴 = 1)"],
        [ink, null, "puncture"],
      ];
      ctx.font = `${small}px ${ftext}`; ctx.textAlign = "left";
      let lx = 8, ly = wide ? H - 34 : P.y + P.h + small + 4;
      for (const [c, dash, t] of items) {
        const tw = ctx.measureText(t).width, need = 30 + tw + 22;
        if (lx + need > W) { lx = 8; ly += small + 8; }
        if (dash) { ctx.strokeStyle = c; ctx.lineWidth = 3; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(lx, ly - small * 0.33); ctx.lineTo(lx + 22, ly - small * 0.33); ctx.stroke(); ctx.setLineDash([]); }
        else { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(lx + 11, ly - small * 0.33, 5, 0, 2 * Math.PI); ctx.fill(); }
        ctx.fillStyle = ink; ctx.fillText(t, lx + 30, ly); lx += need;
      }
      ctx.fillStyle = muted;
      const c1 = `${S.K} × ${S.N} = ${S.K * S.N} nodes in the meridian plane`, c2 = "production grid: 45 × 40 × 8 = 14 400 unknowns";
      if (wide) ctx.fillText(`${c1}   ·   ${c2}`, 8, ly + small + 10);
      else { ctx.fillText(c1, 8, ly + small + 10); ctx.fillText(c2, 8, ly + 2 * small + 16); }

      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function init() { document.querySelectorAll(".lm-abt").forEach(mount); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
