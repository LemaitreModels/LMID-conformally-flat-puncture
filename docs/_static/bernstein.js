/* Interactive Bernstein ellipse: how far a singularity sits from [-1, 1] in the complex plane sets
   how fast Chebyshev interpolation on [-1, 1] converges. Mounts into every element with class
   "lm-bern". No dependencies; colours from the Lemaitre tokens.

   The test function is f(xi) = 1 / ((xi - a)^2 + b^2), analytic everywhere except a pair of poles at
   xi* = a +- i b (a double pole on the real axis when b = 0 and |a| > 1). With a = 0, b = 0.2 it is
   Runge's function 1 / (1 + 25 xi^2).

   Three panels:
   - the complex xi-plane: the interval [-1, 1] (the foci), the Q Chebyshev nodes on it, faint
     Bernstein ellipses E_rho, and in blue the largest one free of singularities, which passes
     through the poles (red crosses). A lighter ellipse grows out of the interval until it meets
     the poles, then fades, and starts again. Hollow circles mark where E_rho meets the real axis.
   - the measured maximum relative error of the Q-node interpolant against Q (dots), with the
     slope rho^-Q of the Bernstein bound (dashed; the prefactor is matched to the dots).
   - f and its Q-node interpolant on [-1, 1].

   Interpolation is at the Q Chebyshev points of the second kind, cos(j pi / (Q - 1)), evaluated
   with the barycentric formula (Berrut & Trefethen 2004, SIAM Rev. 46, 501). The bound is
   Theorem 8.2 of Trefethen, Approximation Theory and Approximation Practice (SIAM 2019). */
(function () {
  "use strict";

  function token(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  const QMAX = 60, GRID = 600, FLOOR = 1e-16;
  const MATH = { xi: "\u{1D709}", rho: "\u{1D70C}", Q: "\u{1D444}", f: "\u{1D453}", p: "\u{1D45D}", E: "\u{1D438}" };

  // Bernstein parameter of the ellipse through xi = a + i b: the larger |z| of z = xi +- sqrt(xi^2 - 1).
  function rhoOf(a, b) {
    const wr = a * a - b * b - 1, wi = 2 * a * b, m = Math.hypot(wr, wi);
    const sr = Math.sqrt(Math.max(0, (m + wr) / 2)), si = (wi < 0 ? -1 : 1) * Math.sqrt(Math.max(0, (m - wr) / 2));
    return Math.max(Math.hypot(a + sr, b + si), Math.hypot(a - sr, b - si));
  }

  const GX = Array.from({ length: GRID }, (_, g) => -1 + (2 * g + 1) / GRID);   // error grid, off the nodes

  function nodes(Q) { return Array.from({ length: Q }, (_, j) => Math.cos((j * Math.PI) / (Q - 1))); }

  // Barycentric interpolant through (x_j, f_j) at the Chebyshev points of the second kind.
  function interpolant(xs, fs) {
    const Q = xs.length, w = xs.map((_, j) => (j % 2 ? -1 : 1) * (j === 0 || j === Q - 1 ? 0.5 : 1));
    return (x) => {
      let num = 0, den = 0;
      for (let j = 0; j < Q; j++) {
        const d = x - xs[j];
        if (d === 0) return fs[j];
        const c = w[j] / d; num += c * fs[j]; den += c;
      }
      return num / den;
    };
  }

  function errors(f) {
    const fg = GX.map(f), scale = Math.max(...fg.map(Math.abs)), out = [];
    for (let Q = 2; Q <= QMAX; Q++) {
      const xs = nodes(Q), p = interpolant(xs, xs.map(f));
      let e = 0;
      for (let g = 0; g < GRID; g++) e = Math.max(e, Math.abs(p(GX[g]) - fg[g]));
      out.push([Q, Math.max(e / scale, FLOOR)]);
    }
    return out;
  }

  function mount(root) {
    root.innerHTML = "";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "The complex plane with the interval from -1 to 1, a pair of singularities and the Bernstein ellipse through them; the interpolation error against the number of nodes; and the function with its interpolant");
    const controls = document.createElement("div");
    controls.className = "lm-bern-controls";
    root.append(canvas, controls);

    const S = { a: 0, b: 0.2, Q: 12 };
    let cache = null;

    function slider(label, min, max, step, key, fmt) {
      const lab = document.createElement("label");
      const name = document.createElement("span"); name.innerHTML = label;
      const input = document.createElement("input");
      Object.assign(input, { type: "range", min, max, step, value: S[key] });
      const out = document.createElement("output");
      const show = () => { out.textContent = fmt(S[key]); };
      input.addEventListener("input", () => { S[key] = +input.value; if (key !== "Q") clamp(); sync(); });
      lab.append(name, input, out); controls.append(lab);
      return { input, show };
    }
    const sl = [
      slider("Re&nbsp;<i>ξ</i><sub>∗</sub>", -2, 2, "any", "a", (v) => v.toFixed(2)),
      slider("Im&nbsp;<i>ξ</i><sub>∗</sub>", 0, 1.2, "any", "b", (v) => v.toFixed(2)),
      slider("nodes&nbsp;<i>Q</i>", 2, QMAX, 1, "Q", (v) => String(v)),
    ];
    const keys = ["a", "b", "Q"];
    function clamp() {
      S.a = Math.max(-2.2, Math.min(2.2, S.a));
      S.b = Math.max(0, Math.min(1.3, S.b));
      if (Math.abs(S.a) <= 1.03) S.b = Math.max(S.b, 0.03);       // never on the interval itself
      cache = null; sweepStart = performance.now();
    }
    function sync() { sl.forEach((s, i) => { if (document.activeElement !== s.input) s.input.value = S[keys[i]]; s.show(); }); }
    sync();

    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1, wide = true, P = {}, Ep = {}, F = {};
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = root.clientWidth; wide = W >= 620;
      if (wide) {
        const top = Math.round(Math.min(W * 0.36, 340)), bot = Math.round(Math.min(W * 0.24, 210));
        const wp = Math.round(W * 0.56);
        P = { x: 0, y: 0, w: wp, h: top }; Ep = { x: wp, y: 0, w: W - wp, h: top }; F = { x: 0, y: top, w: W, h: bot };
        H = top + bot;
      } else {
        const hp = Math.round(W * 0.62), he = Math.round(W * 0.6), hf = Math.round(W * 0.55);
        P = { x: 0, y: 0, w: W, h: hp }; Ep = { x: 0, y: hp, w: W, h: he }; F = { x: 0, y: hp + he, w: W, h: hf };
        H = hp + he + hf;
      }
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
    }
    new ResizeObserver(resize).observe(root);
    resize();

    // The complex plane: equal scales, Re in [-XR, XR], Im in [-YR, YR].
    const XR = 2.25, YR = 1.35;
    function plane() {
      const pad = 10, s = Math.min((P.w - 2 * pad) / (2 * XR), (P.h - 2 * pad) / (2 * YR));
      const cx = P.x + P.w / 2, cy = P.y + P.h / 2;
      return { s, cx, cy, X: (x) => cx + x * s, Y: (y) => cy - y * s, inv: (px, py) => [(px - cx) / s, (cy - py) / s] };
    }

    // Drag the singularity anywhere in the plane panel.
    let dragging = false;
    function inPlane(e) { const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; return [x, y, x >= P.x && x <= P.x + P.w && y >= P.y && y <= P.y + P.h]; }
    function moveTo(x, y) { const [u, v] = plane().inv(x, y); S.a = u; S.b = Math.abs(v); clamp(); sync(); }
    canvas.addEventListener("pointerdown", (e) => { const [x, y, ok] = inPlane(e); if (!ok) return; dragging = true; canvas.setPointerCapture(e.pointerId); moveTo(x, y); });
    canvas.addEventListener("pointermove", (e) => { const [x, y, ok] = inPlane(e); canvas.style.cursor = ok ? "crosshair" : ""; if (dragging) moveTo(x, y); });
    canvas.addEventListener("pointerup", () => { dragging = false; });
    canvas.addEventListener("pointercancel", () => { dragging = false; });

    const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const GROW = 2600, HOLD = 1400, FADE = 700, REST = 900;
    let sweepStart = performance.now();

    function ellipse(pl, rho, n = 160) {
      const A = (rho + 1 / rho) / 2, B = (rho - 1 / rho) / 2;
      ctx.beginPath();
      for (let i = 0; i <= n; i++) { const t = (2 * Math.PI * i) / n; const px = pl.X(A * Math.cos(t)), py = pl.Y(B * Math.sin(t)); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    }
    // base^exp, right-aligned at x; with measure set, only returns the width
    function power(x, y, base, exp, fBase, fExp, measure) {
      ctx.font = fExp; const we = ctx.measureText(exp).width;
      ctx.font = fBase; const wb = ctx.measureText(base).width;
      if (measure) return wb + we;
      ctx.textAlign = "left"; ctx.fillText(base, x - wb - we, y);
      ctx.font = fExp; ctx.fillText(exp, x - we, y - parseFloat(fBase) * 0.4);
      ctx.font = fBase; ctx.textAlign = "right";
    }
    function cross(x, y, r) { ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke(); }

    function frame(now) {
      const ink = token(root, "--lm-ink", "#1f1e1c"), muted = token(root, "--lm-muted", "#6c6a63");
      const rule = token(root, "--lm-rule", "#dcd8cc"), blue = token(root, "--lm-accent", "#2c5a85");
      const red = token(root, "--lm-accent2", "#b04632");
      const ftext = token(root, "--lm-font-text", "system-ui, sans-serif"), fmath = token(root, "--lm-font-math", "serif");
      const fs = parseFloat(getComputedStyle(root).fontSize) || 16, small = Math.round(fs * 0.82);
      const f = (x) => 1 / ((x - S.a) * (x - S.a) + S.b * S.b);
      const rho = rhoOf(S.a, S.b);
      if (!cache) cache = errors(f);

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = "round"; ctx.lineJoin = "round";

      // --- complex plane ---------------------------------------------------------------------
      const pl = plane();
      ctx.save(); ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pl.X(-XR), pl.Y(0)); ctx.lineTo(pl.X(XR), pl.Y(0)); ctx.moveTo(pl.X(0), pl.Y(-YR)); ctx.lineTo(pl.X(0), pl.Y(YR)); ctx.stroke();
      for (const r of [1.1, 1.25, 1.5, 2, 3]) { ellipse(pl, r); ctx.strokeStyle = rule; ctx.lineWidth = 0.8; ctx.stroke(); }

      // the growing ellipse
      if (!still) {
        const c = (now - sweepStart) % (GROW + HOLD + FADE + REST);
        let r = rho, alpha = 0;
        if (c < GROW) { const s = c / GROW, e = 1 - (1 - s) ** 3; r = 1 + (rho - 1) * e; alpha = 0.55; }
        else if (c < GROW + HOLD) alpha = 0.55;
        else if (c < GROW + HOLD + FADE) alpha = 0.55 * (1 - (c - GROW - HOLD) / FADE);
        if (alpha > 0) { ellipse(pl, r); ctx.globalAlpha = alpha; ctx.strokeStyle = blue; ctx.lineWidth = 3.5; ctx.stroke(); ctx.globalAlpha = 1; }
      }
      ellipse(pl, rho); ctx.strokeStyle = blue; ctx.lineWidth = 1.6; ctx.stroke();

      // where E_rho meets the real axis: the real point a one-parameter fit would report
      const A = (rho + 1 / rho) / 2;
      ctx.fillStyle = token(root, "--lm-paper", "#faf9f5"); ctx.strokeStyle = blue; ctx.lineWidth = 1.4;
      for (const sx of [-A, A]) { ctx.beginPath(); ctx.arc(pl.X(sx), pl.Y(0), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }

      // the interval and its nodes
      ctx.strokeStyle = ink; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(pl.X(-1), pl.Y(0)); ctx.lineTo(pl.X(1), pl.Y(0)); ctx.stroke();
      ctx.fillStyle = ink;
      for (const x of nodes(S.Q)) { ctx.beginPath(); ctx.arc(pl.X(x), pl.Y(0), S.Q > 30 ? 1.6 : 2.3, 0, 2 * Math.PI); ctx.fill(); }

      // the singularities
      ctx.strokeStyle = red; ctx.lineWidth = 2.2;
      cross(pl.X(S.a), pl.Y(S.b), 6); if (S.b > 0) cross(pl.X(S.a), pl.Y(-S.b), 6);
      ctx.restore();

      // labels
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`; ctx.textAlign = "center";
      ctx.fillText("−1", pl.X(-1), pl.Y(0) + small + 4); ctx.fillText("1", pl.X(1), pl.Y(0) + small + 4);
      ctx.textAlign = "right"; ctx.font = `${fs}px ${fmath}`;
      ctx.fillText("Re " + MATH.xi, P.x + P.w - 6, pl.Y(0) - 6);
      ctx.textAlign = "left"; ctx.fillText("Im " + MATH.xi, pl.X(0) + 6, P.y + fs + 2);
      ctx.fillStyle = red; ctx.fillText(MATH.xi, pl.X(S.a) + 9, pl.Y(S.b) - 7); { const wx = ctx.measureText(MATH.xi).width; ctx.font = `${Math.round(fs * 0.7)}px ${fmath}`; ctx.fillText("∗", pl.X(S.a) + 9 + wx, pl.Y(S.b) - 7 + fs * 0.25); ctx.font = `${fs}px ${fmath}`; }
      const tE = 1.05, bE = (rho - 1 / rho) / 2;
      const ex = pl.X(A * Math.cos(tE)) + 6, ey = pl.Y(bE * Math.sin(tE)) - 6;
      if (ey > P.y + fs * 2.4 && ex < P.x + P.w - 30) { ctx.fillStyle = blue; ctx.textAlign = "left"; ctx.fillText(MATH.E, ex, ey); const wE = ctx.measureText(MATH.E).width; ctx.font = `${Math.round(fs * 0.7)}px ${fmath}`; ctx.fillText(MATH.rho, ex + wE, ey + fs * 0.25); ctx.font = `${fs}px ${fmath}`; }
      // readout
      ctx.textAlign = "left"; ctx.fillStyle = blue; ctx.font = `${fs}px ${fmath}`;
      const rate = Math.log10(rho);
      ctx.fillText(`${MATH.rho} = ${rho.toFixed(3)}`, P.x + 6, P.y + fs + 2);
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`;
      ctx.fillText(`${rate.toFixed(3)} decades per node`, P.x + 6, P.y + fs + small + 8);

      // --- error against Q ---------------------------------------------------------------------
      const em = { l: Ep.x + fs * 3.4, r: Ep.x + Ep.w - 12, t: Ep.y + fs * 0.9, b: Ep.y + Ep.h - fs * 2.6 };
      const QX = (q) => em.l + ((q - 0) / QMAX) * (em.r - em.l), LY = (lg) => em.t + (-lg / 16) * (em.b - em.t);
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(em.l, em.t); ctx.lineTo(em.l, em.b); ctx.lineTo(em.r, em.b); ctx.stroke();
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`;
      ctx.textAlign = "right";
      for (const lg of [0, -4, -8, -12, -16]) {
        if (lg === 0) { ctx.textAlign = "right"; ctx.font = `${small}px ${ftext}`; ctx.fillText("1", em.l - 6, LY(lg) + small * 0.35); }
        else power(em.l - 6, LY(lg) + small * 0.35, "10", String(lg).replace("-", "−"), `${small}px ${ftext}`, `${Math.round(small * 0.72)}px ${ftext}`);
        ctx.strokeStyle = rule; ctx.beginPath(); ctx.moveTo(em.l - 3, LY(lg)); ctx.lineTo(em.l, LY(lg)); ctx.stroke();
      }
      ctx.textAlign = "center";
      for (const q of [0, 20, 40, 60]) ctx.fillText(String(q), QX(q), em.b + small + 4);
      ctx.font = `${fs}px ${ftext}`; ctx.fillStyle = muted;
      ctx.fillText("nodes Q", (em.l + em.r) / 2, em.b + small + fs + 8);
      ctx.save(); ctx.translate(Ep.x + fs * 0.9, (em.t + em.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("relative error", 0, 0); ctx.restore();

      // the Bernstein slope, its prefactor matched to the measured points above roundoff
      const usable = cache.filter(([q, e]) => q >= 6 && e > 1e-13).map(([q, e]) => Math.log10(e) + q * rate).sort((x, y) => x - y);
      if (usable.length) {
        const c0 = usable[Math.floor(usable.length / 2)];
        const qEnd = Math.min(QMAX, (c0 + 16) / rate);
        ctx.save(); ctx.beginPath(); ctx.rect(em.l, em.t - 2, em.r - em.l, em.b - em.t + 4); ctx.clip();
        ctx.setLineDash([6, 5]); ctx.strokeStyle = blue; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(QX(2), LY(c0 - 2 * rate)); ctx.lineTo(QX(qEnd), LY(c0 - qEnd * rate)); ctx.stroke();
        ctx.setLineDash([]); ctx.restore();
        const qL = Math.min(qEnd * 0.55, QMAX * 0.6), yL = LY(Math.min(0, c0 - qL * rate));
        ctx.fillStyle = blue;
        const fB = `${fs}px ${fmath}`, fE = `${Math.round(fs * 0.7)}px ${fmath}`;
        const wL = power(0, 0, MATH.rho, "−" + MATH.Q, fB, fE, true);
        power(QX(qL) + 8 + wL, Math.max(em.t + fs, yL - 8), MATH.rho, "−" + MATH.Q, fB, fE);
      }
      ctx.fillStyle = ink;
      for (const [q, e] of cache) { ctx.beginPath(); ctx.arc(QX(q), LY(Math.log10(e)), 2.1, 0, 2 * Math.PI); ctx.fill(); }
      const cur = cache[S.Q - 2];
      ctx.strokeStyle = red; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(QX(cur[0]), LY(Math.log10(cur[1])), 5, 0, 2 * Math.PI); ctx.stroke();

      // --- f and its interpolant -----------------------------------------------------------------
      const fm = { l: F.x + fs * 3.4, r: F.x + F.w - 12, t: F.y + fs * 1.6, b: F.y + F.h - fs * 1.8 };
      const xs = nodes(S.Q), fsx = xs.map(f), p = interpolant(xs, fsx);
      const N = 400, xx = Array.from({ length: N + 1 }, (_, i) => -1 + (2 * i) / N);
      const fv = xx.map(f), pv = xx.map(p), fmax = Math.max(...fv);
      const lo = Math.max(-0.35 * fmax, Math.min(0, ...pv)), hi = Math.min(1.35 * fmax, Math.max(fmax, ...pv));
      const FX = (x) => fm.l + ((x + 1) / 2) * (fm.r - fm.l), FY = (y) => fm.b - ((y - lo) / (hi - lo || 1)) * (fm.b - fm.t);
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(fm.l, FY(0)); ctx.lineTo(fm.r, FY(0)); ctx.stroke();
      ctx.save(); ctx.beginPath(); ctx.rect(fm.l - 4, fm.t - 6, fm.r - fm.l + 8, fm.b - fm.t + 12); ctx.clip();
      ctx.strokeStyle = ink; ctx.lineWidth = 2;
      ctx.beginPath(); xx.forEach((x, i) => (i ? ctx.lineTo(FX(x), FY(fv[i])) : ctx.moveTo(FX(x), FY(fv[i])))); ctx.stroke();
      ctx.strokeStyle = blue; ctx.lineWidth = 1.4;
      ctx.beginPath(); xx.forEach((x, i) => (i ? ctx.lineTo(FX(x), FY(pv[i])) : ctx.moveTo(FX(x), FY(pv[i])))); ctx.stroke();
      ctx.fillStyle = blue;
      xs.forEach((x, j) => { ctx.beginPath(); ctx.arc(FX(x), FY(fsx[j]), S.Q > 30 ? 1.8 : 2.6, 0, 2 * Math.PI); ctx.fill(); });
      ctx.restore();
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`; ctx.textAlign = "center";
      ctx.fillText("−1", FX(-1), fm.b + small + 4); ctx.fillText("0", FX(0), fm.b + small + 4); ctx.fillText("1", FX(1), fm.b + small + 4);
      ctx.textAlign = "left"; ctx.font = `${fs}px ${fmath}`;
      ctx.fillStyle = ink; ctx.fillText(`${MATH.f}(${MATH.xi})`, F.x + 6, F.y + fs * 0.9);
      ctx.fillStyle = blue; ctx.textAlign = "right"; const wq = ` interpolant on ${S.Q} nodes`;
      ctx.font = `${fs}px ${ftext}`; ctx.fillText(wq, F.x + F.w - 8, F.y + fs * 0.9);
      const ww = ctx.measureText(wq).width; ctx.font = `${fs}px ${fmath}`; ctx.fillText(`${MATH.p}(${MATH.xi}),`, F.x + F.w - 8 - ww, F.y + fs * 0.9);

      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function init() { document.querySelectorAll(".lm-bern").forEach(mount); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
