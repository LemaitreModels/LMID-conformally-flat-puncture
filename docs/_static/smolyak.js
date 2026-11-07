/* Interactive Smolyak sparse grid in two parameters, the construction behind the sparse-grid models
   of the conformally-flat-puncture leaf. Mounts into every element with class "lm-smol". No
   dependencies; colours from the Lemaitre tokens.

   One-dimensional rule: nested Clenshaw–Curtis (Chebyshev–Gauss–Lobatto) nodes with m(0) = 1 node
   (the midpoint) and m(l) = 2^l + 1 nodes, cos(j pi / 2^l), for l >= 1. Level ell keeps the
   multi-indices with l1 + l2 <= ell and recombines them with the combination technique,
     I(ell) = sum_{l1 + l2 = ell} I_l1 x I_l2  -  sum_{l1 + l2 = ell - 1} I_l1 x I_l2,
   the d = 2 case of c_l = (-1)^(ell - |l|) binom(d - 1, ell - |l|) (Smolyak 1963).

   Three panels:
   - the square [-1, 1]^2 with the sparse grid at level ell (larger dots for coarser hierarchical
     level) over the faint full tensor grid of the same finest resolution; the subgrid of the
     highlighted multi-index is ringed, blue for coefficient +1, red for -1;
   - the multi-index plane (l1, l2) with the simplex l1 + l2 <= ell and the combination
     coefficients; the animation steps through the cells that carry a coefficient, and hovering or
     tapping a cell holds it;
   - the measured maximum relative error on a 40 x 40 test grid against the number of nodes, for
     the sparse grid at levels 0..7 and for full tensor grids of q x q Chebyshev points, q = 2..60.
   The test function blends a sum of one-parameter functions into a strongly coupled one,
     f = (1 - kappa) (g(x) + g(y)) / 2 + kappa h(x, y),
     g(t) = 1 / (1 + 4 (t - 0.15)^2),  h = 1 / (1 + 4 ((x - 0.15)^2 + (y - 0.15)^2)),
   with kappa the coupling slider (0 to 0.3, default 0). */
(function () {
  "use strict";

  function token(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  const LMAX = 7, QMAX = 60, NT = 40, FLOOR = 1e-16;
  const m = (l) => (l === 0 ? 1 : 2 ** l + 1);
  const cheb = (n) => (n === 1 ? [0] : Array.from({ length: n }, (_, j) => Math.cos((j * Math.PI) / (n - 1))));
  const TX = Array.from({ length: NT }, (_, i) => -1 + (2 * i + 1) / NT);

  // 1-D level of a coordinate: the first level whose node set contains it.
  function level1(x) {
    if (Math.abs(x) < 1e-12) return 0;
    for (let l = 1; l <= 12; l++) {
      const n = 2 ** l, j = (Math.acos(Math.max(-1, Math.min(1, x))) / Math.PI) * n;
      if (Math.abs(j - Math.round(j)) < 1e-7) return l;
    }
    return 99;
  }

  // Barycentric cardinal functions of the points xs, evaluated at the test points TX: NT x n.
  const cardCache = new Map();
  function cardinals(n) {
    if (cardCache.has(n)) return cardCache.get(n);
    const xs = cheb(n), M = TX.map(() => new Float64Array(n));
    if (n === 1) { M.forEach((r) => (r[0] = 1)); cardCache.set(n, M); return M; }
    const w = xs.map((_, j) => (j % 2 ? -1 : 1) * (j === 0 || j === n - 1 ? 0.5 : 1));
    TX.forEach((x, i) => {
      let s = 0; const r = M[i];
      for (let j = 0; j < n; j++) { const d = x - xs[j]; if (d === 0) { r.fill(0); r[j] = 1; return; } r[j] = w[j] / d; s += r[j]; }
      for (let j = 0; j < n; j++) r[j] /= s;
    });
    cardCache.set(n, M);
    return M;
  }

  // Tensor interpolant on nx x ny Chebyshev points, evaluated on the NT x NT test grid.
  function tensorOnTest(f, nx, ny) {
    const xs = cheb(nx), ys = cheb(ny), Lx = cardinals(nx), Ly = cardinals(ny);
    const T = Array.from({ length: nx }, () => new Float64Array(NT));        // F Ly^T
    for (let a = 0; a < nx; a++) {
      const Fa = ys.map((y) => f(xs[a], y));
      for (let k = 0; k < NT; k++) { let s = 0; const r = Ly[k]; for (let b = 0; b < ny; b++) s += Fa[b] * r[b]; T[a][k] = s; }
    }
    const out = new Float64Array(NT * NT);
    for (let i = 0; i < NT; i++) { const r = Lx[i]; for (let k = 0; k < NT; k++) { let s = 0; for (let a = 0; a < nx; a++) s += r[a] * T[a][k]; out[i * NT + k] = s; } }
    return out;
  }

  function combination(ell) {
    const cells = [];
    for (let l1 = 0; l1 <= ell; l1++) for (let l2 = 0; l1 + l2 <= ell; l2++) {
      const s = l1 + l2, c = s === ell ? 1 : s === ell - 1 ? -1 : 0;
      cells.push({ l1, l2, c });
    }
    return cells;
  }

  function sparsePoints(ell) {
    const seen = new Map();
    for (const { l1, l2 } of combination(ell)) for (const x of cheb(m(l1))) for (const y of cheb(m(l2))) {
      const k = x.toFixed(12) + "," + y.toFixed(12);
      if (!seen.has(k)) seen.set(k, [x, y, level1(x) + level1(y)]);
    }
    return [...seen.values()];
  }
  const COUNTS = Array.from({ length: LMAX + 1 }, (_, l) => sparsePoints(l).length);

  function errors(f) {
    const ft = new Float64Array(NT * NT);
    let scale = 0;
    TX.forEach((x, i) => TX.forEach((y, k) => { ft[i * NT + k] = f(x, y); scale = Math.max(scale, Math.abs(ft[i * NT + k])); }));
    const err = (u) => { let e = 0; for (let i = 0; i < u.length; i++) e = Math.max(e, Math.abs(u[i] - ft[i])); return Math.max(e / scale, FLOOR); };
    const sparse = [], tensor = [];
    for (let ell = 0; ell <= LMAX; ell++) {
      const u = new Float64Array(NT * NT);
      for (const { l1, l2, c } of combination(ell)) if (c) { const t = tensorOnTest(f, m(l1), m(l2)); for (let i = 0; i < u.length; i++) u[i] += c * t[i]; }
      sparse.push([COUNTS[ell], err(u)]);
    }
    for (let q = 2; q <= QMAX; q++) tensor.push([q * q, err(tensorOnTest(f, q, q))]);
    return { sparse, tensor };
  }

  function mount(root) {
    root.innerHTML = "";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "A Smolyak sparse grid on the square, the multi-indices it combines with their coefficients, and its interpolation error against the number of nodes compared with full tensor grids");
    const controls = document.createElement("div");
    controls.className = "lm-smol-controls";
    root.append(canvas, controls);

    const S = { ell: 4, kappa: 0 };
    let cache = null, pts = sparsePoints(S.ell);

    function slider(label, min, max, step, key, fmt) {
      const lab = document.createElement("label");
      const name = document.createElement("span"); name.innerHTML = label;
      const input = document.createElement("input");
      Object.assign(input, { type: "range", min, max, step, value: S[key] });
      const out = document.createElement("output"); out.textContent = fmt(S[key]);
      input.addEventListener("input", () => {
        S[key] = +input.value; out.textContent = fmt(S[key]);
        if (key === "ell") { pts = sparsePoints(S.ell); step0 = performance.now(); held = null; } else cache = null;
      });
      lab.append(name, input, out); controls.append(lab);
    }
    slider("level&nbsp;<i>ℓ</i>", 0, LMAX, 1, "ell", (v) => String(v));
    slider("coupling&nbsp;<i>κ</i>", 0, 0.3, "any", "kappa", (v) => v.toFixed(3));

    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1, A = {}, B = {}, C = {};
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = root.clientWidth;
      if (W >= 620) {
        const top = Math.round(Math.min(W * 0.46, 380)), bot = Math.round(Math.min(W * 0.36, 300));
        A = { x: 0, y: 0, w: Math.round(W / 2), h: top }; B = { x: Math.round(W / 2), y: 0, w: W - Math.round(W / 2), h: top };
        C = { x: 0, y: top, w: W, h: bot }; H = top + bot;
      } else {
        const ha = Math.round(W * 0.92), hb = Math.round(W * 0.85), hc = Math.round(W * 0.75);
        A = { x: 0, y: 0, w: W, h: ha }; B = { x: 0, y: ha, w: W, h: hb }; C = { x: 0, y: ha + hb, w: W, h: hc }; H = ha + hb + hc;
      }
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
    }
    new ResizeObserver(resize).observe(root);
    resize();

    // index-panel geometry, shared by drawing and hit-testing
    function indexGeom(fs) {
      const pad = fs * 2.2, size = Math.min(B.w - pad - 12, B.h - pad - 14 - fs * 1.2), cell = size / (LMAX + 1);
      const x0 = B.x + (B.w - size + pad) / 2, y0 = B.y + 8 + fs * 1.2 + size;
      return { cell, x0, y0, size };
    }
    let held = null, hover = null;
    function cellAt(e) {
      const r = canvas.getBoundingClientRect(), fs = parseFloat(getComputedStyle(root).fontSize) || 16, g = indexGeom(fs);
      const l1 = Math.floor((e.clientX - r.left - g.x0) / g.cell), l2 = Math.floor((g.y0 - (e.clientY - r.top)) / g.cell);
      return l1 >= 0 && l2 >= 0 && l1 <= LMAX && l2 <= LMAX && l1 + l2 <= S.ell ? { l1, l2 } : null;
    }
    canvas.addEventListener("pointermove", (e) => { hover = cellAt(e); canvas.style.cursor = hover ? "pointer" : ""; });
    canvas.addEventListener("pointerleave", () => { hover = null; });
    canvas.addEventListener("pointerdown", (e) => { const c = cellAt(e); held = c && held && c.l1 === held.l1 && c.l2 === held.l2 ? null : c; });

    const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const STEP = 1300;
    let step0 = performance.now();

    function frame(now) {
      const ink = token(root, "--lm-ink", "#1f1e1c"), muted = token(root, "--lm-muted", "#6c6a63");
      const rule = token(root, "--lm-rule", "#dcd8cc"), soft = token(root, "--lm-soft", "#f0eee6");
      const blue = token(root, "--lm-accent", "#2c5a85"), red = token(root, "--lm-accent2", "#b04632");
      const ftext = token(root, "--lm-font-text", "system-ui, sans-serif"), fmath = token(root, "--lm-font-math", "serif");
      const fs = parseFloat(getComputedStyle(root).fontSize) || 16, small = Math.round(fs * 0.82);
      const g = (t) => 1 / (1 + 4 * (t - 0.15) ** 2);
      const f = (x, y) => (1 - S.kappa) * (g(x) + g(y)) / 2 + S.kappa / (1 + 4 * ((x - 0.15) ** 2 + (y - 0.15) ** 2));
      if (!cache) cache = errors(f);

      const combo = combination(S.ell).filter((c) => c.c);
      let act = held || hover;
      if (!act && !still) act = combo[Math.floor((now - step0) / STEP) % combo.length];
      const actC = act ? (act.l1 + act.l2 === S.ell ? 1 : act.l1 + act.l2 === S.ell - 1 ? -1 : 0) : 0;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // --- the sparse grid on the square ------------------------------------------------------------
      const side = Math.min(A.w - fs * 2.4, A.h - fs * 2.6), sx0 = A.x + (A.w - side) / 2 + fs * 0.4, sy0 = A.y + fs * 1.9;
      const PX = (x) => sx0 + ((x + 1) / 2) * side, PY = (y) => sy0 + ((1 - y) / 2) * side;
      ctx.strokeStyle = rule; ctx.lineWidth = 1; ctx.strokeRect(sx0, sy0, side, side);
      const fine = cheb(m(S.ell));
      ctx.fillStyle = rule;
      if (fine.length <= 65) for (const x of fine) for (const y of fine) { ctx.beginPath(); ctx.arc(PX(x), PY(y), 1.1, 0, 2 * Math.PI); ctx.fill(); }
      ctx.fillStyle = ink;
      for (const [x, y, h] of pts) { ctx.beginPath(); ctx.arc(PX(x), PY(y), Math.max(1.4, 4.2 - 0.45 * h), 0, 2 * Math.PI); ctx.fill(); }
      if (act) {
        ctx.strokeStyle = actC > 0 ? blue : actC < 0 ? red : muted; ctx.lineWidth = 1.6;
        for (const x of cheb(m(act.l1))) for (const y of cheb(m(act.l2))) { ctx.beginPath(); ctx.arc(PX(x), PY(y), 5.5, 0, 2 * Math.PI); ctx.stroke(); }
      }
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`; ctx.textAlign = "left";
      const nT = m(S.ell) ** 2;
      ctx.fillText(`${pts.length} sparse nodes   ·   full tensor grid ${m(S.ell)}² = ${nT}`, sx0, A.y + fs * 1.1);
      ctx.font = `${fs}px ${fmath}`; ctx.textAlign = "center";
      ctx.fillText("𝑥", sx0 + side / 2, sy0 + side + fs * 1.15);
      ctx.save(); ctx.translate(sx0 - fs * 0.7, sy0 + side / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("𝑦", 0, 0); ctx.restore();

      // --- the multi-index plane ---------------------------------------------------------------------
      const G = indexGeom(fs);
      for (let l1 = 0; l1 <= LMAX; l1++) for (let l2 = 0; l2 <= LMAX; l2++) {
        const x = G.x0 + l1 * G.cell, y = G.y0 - (l2 + 1) * G.cell, s = l1 + l2;
        const inS = s <= S.ell, c = s === S.ell ? 1 : s === S.ell - 1 ? -1 : 0;
        ctx.fillStyle = inS ? soft : "transparent"; if (inS) ctx.fillRect(x + 1, y + 1, G.cell - 2, G.cell - 2);
        ctx.strokeStyle = inS && c ? (c > 0 ? blue : red) : rule; ctx.lineWidth = inS && c ? 1.6 : 0.8;
        ctx.strokeRect(x + 1.5, y + 1.5, G.cell - 3, G.cell - 3);
        if (act && act.l1 === l1 && act.l2 === l2) { ctx.globalAlpha = 0.18; ctx.fillStyle = c > 0 ? blue : c < 0 ? red : muted; ctx.fillRect(x + 1, y + 1, G.cell - 2, G.cell - 2); ctx.globalAlpha = 1; }
        if (inS && c) { ctx.fillStyle = c > 0 ? blue : red; ctx.font = `${Math.round(Math.min(small, G.cell * 0.42))}px ${ftext}`; ctx.textAlign = "center"; ctx.fillText(c > 0 ? "+1" : "−1", x + G.cell / 2, y + G.cell / 2 + small * 0.33); }
      }
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`; ctx.textAlign = "center";
      for (let l = 0; l <= LMAX; l++) { ctx.fillText(String(l), G.x0 + (l + 0.5) * G.cell, G.y0 + small + 3); ctx.fillText(String(l), G.x0 - small * 0.8, G.y0 - (l + 0.5) * G.cell + small * 0.33); }
      ctx.font = `${fs}px ${fmath}`; ctx.fillStyle = muted;
      ctx.textAlign = "right"; ctx.fillText("𝑙₁", G.x0 + G.size, G.y0 + small + fs + 4);
      ctx.textAlign = "center"; ctx.fillText("𝑙₂", G.x0 - small * 0.8, G.y0 - G.size - 6);
      if (act) {
        ctx.textAlign = "right"; ctx.fillStyle = actC > 0 ? blue : actC < 0 ? red : muted; ctx.font = `${small}px ${ftext}`;
        ctx.fillText(`subgrid ${m(act.l1)} × ${m(act.l2)}, coefficient ${actC > 0 ? "+1" : actC < 0 ? "−1" : "0"}`, G.x0 + G.size, G.y0 - G.size - 6);
      }

      // --- error against nodes ---------------------------------------------------------------------
      const em = { l: C.x + fs * 3.4, r: C.x + C.w - 12, t: C.y + fs * 1.4, b: C.y + C.h - fs * 2.6 };
      const NX = (n) => em.l + (Math.log10(n) / 4) * (em.r - em.l), LY = (lg) => em.t + (-lg / 16) * (em.b - em.t);
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(em.l, em.t); ctx.lineTo(em.l, em.b); ctx.lineTo(em.r, em.b); ctx.stroke();
      ctx.fillStyle = muted;
      const YT = { 0: "1", "-4": "10⁻⁴", "-8": "10⁻⁸", "-12": "10⁻¹²", "-16": "10⁻¹⁶" };
      ctx.font = `${small}px ${ftext}`; ctx.textAlign = "right";
      for (const lg of [0, -4, -8, -12, -16]) {
        ctx.fillText(YT[lg], em.l - 6, LY(lg) + small * 0.35);
        ctx.strokeStyle = rule; ctx.beginPath(); ctx.moveTo(em.l - 3, LY(lg)); ctx.lineTo(em.l, LY(lg)); ctx.stroke();
      }
      ctx.textAlign = "center";
      ["1", "10", "10²", "10³", "10⁴"].forEach((t, k) => ctx.fillText(t, NX(10 ** k), em.b + small + 4));
      ctx.font = `${fs}px ${ftext}`; ctx.textAlign = "center";
      ctx.fillText("nodes (elliptic solves)", (em.l + em.r) / 2, em.b + small + fs + 8);
      ctx.save(); ctx.translate(C.x + fs * 0.9, (em.t + em.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("relative error", 0, 0); ctx.restore();
      function curve(data, color, width, dots) {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
        data.forEach(([n, e], i) => { const x = NX(n), y = LY(Math.log10(e)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
        if (dots) { ctx.fillStyle = color; for (const [n, e] of data) { ctx.beginPath(); ctx.arc(NX(n), LY(Math.log10(e)), 3, 0, 2 * Math.PI); ctx.fill(); } }
      }
      curve(cache.tensor, ink, 1.3, false);
      curve(cache.sparse, blue, 1.6, true);
      const cur = cache.sparse[S.ell];
      ctx.strokeStyle = red; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(NX(cur[0]), LY(Math.log10(cur[1])), 6, 0, 2 * Math.PI); ctx.stroke();
      // legend, top right
      ctx.font = `${small}px ${ftext}`; ctx.textAlign = "left";
      const lx = em.r - small * 13, ly = em.t + small * 0.6;
      ctx.strokeStyle = blue; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 22, ly); ctx.stroke();
      ctx.fillStyle = blue; ctx.beginPath(); ctx.arc(lx + 11, ly, 3, 0, 2 * Math.PI); ctx.fill();
      ctx.fillText("sparse grid, ℓ = 0 … 7", lx + 30, ly + small * 0.35);
      ctx.strokeStyle = ink; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(lx, ly + small * 1.5); ctx.lineTo(lx + 22, ly + small * 1.5); ctx.stroke();
      ctx.fillStyle = ink; ctx.fillText("tensor grid q × q", lx + 30, ly + small * 1.85);

      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function init() { document.querySelectorAll(".lm-smol").forEach(mount); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
