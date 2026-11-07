/* Interactive parameter sensitivities and Hermite interpolation, for one parameter. Mounts into
   every element with class "lm-herm". No dependencies; colours from the Lemaitre tokens.

   The "solve" is a real nonlinear equation, a one-number stand-in for the constraint solve:
     R(u, theta) = u^3 + u - g(theta) = 0,   g(theta) = G0 (theta - a) / beta,   G0 = 2 / (3 sqrt 3).
   For real theta it has one real root, found by Newton's method. Its sensitivity comes from the
   tangent equation, differentiating R(u(theta), theta) = 0:
     R_u du/dtheta = -R_theta,   i.e.   (3 u^2 + 1) du/dtheta = g'(theta).
   R_u vanishes at u = +-i/sqrt 3, where g = +-i G0, so u(theta) has square-root branch points at
   theta* = a +- i beta: the sliders move them, as on the Bernstein-ellipse figure.

   Q nodes are the Chebyshev points of the second kind. Lagrange interpolates the Q values; Hermite
   interpolates the Q values and the Q tangents, with the osculatory cardinals
     H_i = [1 - 2 L_i'(theta_i)(theta - theta_i)] L_i^2,   Hhat_i = (theta - theta_i) L_i^2,
   L_i the barycentric Lagrange cardinal and L_i'(theta_i) = sum_{j != i} 1 / (theta_i - theta_j).

   Three panels: u(theta) with the two interpolants, the stored tangents drawn at the nodes and a
   moving probe with the tangent lines of the truth and of both interpolants; du/dtheta with the
   derivatives of both interpolants; and the maximum relative errors of value and derivative
   against Q, measured on 500 points. The probe sweeps by itself; hovering over the curves moves it
   by hand, and it resumes after a pause. */
(function () {
  "use strict";

  function token(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  const G0 = 2 / (3 * Math.sqrt(3)), QMAX = 32, NG = 500, FLOOR = 1e-16;
  const GX = Array.from({ length: NG }, (_, i) => -1 + (2 * i + 1) / NG);

  function problem(a, beta) {
    const g = (t) => (G0 * (t - a)) / beta, gp = G0 / beta;
    function u(t) {                                        // Newton on u^3 + u = g
      const G = g(t); let x = G / 2;
      for (let k = 0; k < 50; k++) { const dx = (x * x * x + x - G) / (3 * x * x + 1); x -= dx; if (Math.abs(dx) < 1e-16 * (1 + Math.abs(x))) break; }
      return x;
    }
    const du = (t) => { const x = u(t); return gp / (3 * x * x + 1); };   // the tangent equation
    return { u, du };
  }

  const cheb = (Q) => Array.from({ length: Q }, (_, j) => Math.cos((j * Math.PI) / (Q - 1)));

  // Lagrange and Hermite interpolants of (values fv, tangents fd) on xs: value and derivative at t.
  function interpolants(xs, fv, fd) {
    const Q = xs.length;
    const w = xs.map((_, j) => (j % 2 ? -1 : 1) * (j === 0 || j === Q - 1 ? 0.5 : 1));
    const D = xs.map((xi, i) => xs.reduce((s, xj, j) => (j === i ? s : s + 1 / (xi - xj)), 0));
    return (t) => {
      if (xs.some((x) => x === t)) t += 1e-12;                 // the formulas below are singular on a node
      let S = 0, Sp = 0;
      const A = new Float64Array(Q), Ap = new Float64Array(Q);
      for (let j = 0; j < Q; j++) { const d = t - xs[j]; A[j] = w[j] / d; Ap[j] = -w[j] / (d * d); S += A[j]; Sp += Ap[j]; }
      let lv = 0, ld = 0, hv = 0, hd = 0;
      for (let i = 0; i < Q; i++) {
        const L = A[i] / S, Lp = (Ap[i] * S - A[i] * Sp) / (S * S), dx = t - xs[i];
        const H = (1 - 2 * D[i] * dx) * L * L, Hh = dx * L * L;
        const Hp = -2 * D[i] * L * L + (1 - 2 * D[i] * dx) * 2 * L * Lp, Hhp = L * L + 2 * dx * L * Lp;
        lv += L * fv[i]; ld += Lp * fv[i];
        hv += H * fv[i] + Hh * fd[i]; hd += Hp * fv[i] + Hhp * fd[i];
      }
      return [lv, ld, hv, hd];
    };
  }

  function rhoOf(a, b) {
    const wr = a * a - b * b - 1, wi = 2 * a * b, m = Math.hypot(wr, wi);
    const sr = Math.sqrt(Math.max(0, (m + wr) / 2)), si = (wi < 0 ? -1 : 1) * Math.sqrt(Math.max(0, (m - wr) / 2));
    return Math.max(Math.hypot(a + sr, b + si), Math.hypot(a - sr, b - si));
  }

  function mount(root) {
    root.innerHTML = "";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "A solution and its parameter derivative with their Lagrange and Hermite interpolants, and the interpolation errors of value and derivative against the number of nodes");
    const controls = document.createElement("div");
    controls.className = "lm-herm-controls";
    root.append(canvas, controls);

    const S = { a: 0, b: 0.3, Q: 8 };
    let P = problem(S.a, S.b), cache = null, cur = null;

    function slider(label, min, max, step, key, fmt) {
      const lab = document.createElement("label");
      const name = document.createElement("span"); name.innerHTML = label;
      const input = document.createElement("input");
      Object.assign(input, { type: "range", min, max, step, value: S[key] });
      const out = document.createElement("output"); out.textContent = fmt(S[key]);
      input.addEventListener("input", () => {
        S[key] = +input.value; out.textContent = fmt(S[key]);
        if (key !== "Q") { P = problem(S.a, S.b); cache = null; }
        cur = null;
      });
      lab.append(name, input, out); controls.append(lab);
    }
    slider("Re&nbsp;<i>θ</i><sub>∗</sub>", -1.5, 1.5, "any", "a", (v) => v.toFixed(2));
    slider("Im&nbsp;<i>θ</i><sub>∗</sub>", 0.05, 1.2, "any", "b", (v) => v.toFixed(2));
    slider("nodes&nbsp;<i>Q</i>", 2, QMAX, 1, "Q", (v) => String(v));

    function errors() {
      const U = GX.map(P.u), DU = GX.map(P.du);
      const su = Math.max(...U.map(Math.abs)), sd = Math.max(...DU.map(Math.abs));
      const out = { lv: [], ld: [], hv: [], hd: [] };
      for (let Q = 2; Q <= QMAX; Q++) {
        const xs = cheb(Q), f = interpolants(xs, xs.map(P.u), xs.map(P.du));
        let e = [0, 0, 0, 0];
        GX.forEach((t, k) => { const r = f(t); e = [Math.max(e[0], Math.abs(r[0] - U[k])), Math.max(e[1], Math.abs(r[1] - DU[k])), Math.max(e[2], Math.abs(r[2] - U[k])), Math.max(e[3], Math.abs(r[3] - DU[k]))]; });
        out.lv.push([Q, Math.max(e[0] / su, FLOOR)]); out.ld.push([Q, Math.max(e[1] / sd, FLOOR)]);
        out.hv.push([Q, Math.max(e[2] / su, FLOOR)]); out.hd.push([Q, Math.max(e[3] / sd, FLOOR)]);
      }
      return out;
    }

    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1, A = {}, B = {}, C = {};
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = root.clientWidth;
      if (W >= 620) {
        const wl = Math.round(W * 0.58), h1 = Math.round(Math.min(W * 0.27, 230)), h2 = Math.round(Math.min(W * 0.24, 200));
        A = { x: 0, y: 0, w: wl, h: h1 }; B = { x: 0, y: h1, w: wl, h: h2 }; C = { x: wl, y: 0, w: W - wl, h: h1 + h2 }; H = h1 + h2;
      } else {
        const h1 = Math.round(W * 0.6), h2 = Math.round(W * 0.5), h3 = Math.round(W * 0.75);
        A = { x: 0, y: 0, w: W, h: h1 }; B = { x: 0, y: h1, w: W, h: h2 }; C = { x: 0, y: h1 + h2, w: W, h: h3 }; H = h1 + h2 + h3;
      }
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
    }
    new ResizeObserver(resize).observe(root);
    resize();

    // the probe: sweeps by itself, follows the pointer over the two curve panels, resumes after a pause
    const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let probe = 0.35, manualUntil = 0, phase = 0, last = performance.now();
    const PAUSE = 2500, PERIOD = 14000;
    function curveX(panel, fs) { return { l: panel.x + fs * 3.2, r: panel.x + panel.w - 14 }; }
    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const fs = parseFloat(getComputedStyle(root).fontSize) || 16, cx = curveX(A, fs);
      if (x >= cx.l && x <= cx.r && y >= A.y && y <= B.y + B.h && x <= A.x + A.w) {
        probe = -1 + (2 * (x - cx.l)) / (cx.r - cx.l); manualUntil = performance.now() + PAUSE;
        phase = Math.asin(Math.max(-1, Math.min(1, probe / 0.92))) / (2 * Math.PI);
      }
    });

    function frame(now) {
      const ink = token(root, "--lm-ink", "#1f1e1c"), muted = token(root, "--lm-muted", "#6c6a63");
      const rule = token(root, "--lm-rule", "#dcd8cc"), blue = token(root, "--lm-accent", "#2c5a85");
      const red = token(root, "--lm-accent2", "#b04632");
      const ftext = token(root, "--lm-font-text", "system-ui, sans-serif"), fmath = token(root, "--lm-font-math", "serif");
      const fs = parseFloat(getComputedStyle(root).fontSize) || 16, small = Math.round(fs * 0.82);
      if (!cache) cache = errors();
      if (!cur) {
        const xs = cheb(S.Q), fv = xs.map(P.u), fd = xs.map(P.du), f = interpolants(xs, fv, fd);
        const N = 301, tt = Array.from({ length: N + 1 }, (_, i) => -1 + (2 * i + 1) / (N + 1));   // off the nodes
        cur = { xs, fv, fd, f, tt, U: tt.map(P.u), DU: tt.map(P.du), R: tt.map(f) };
      }
      const dt = now - last; last = now;
      if (!still && now > manualUntil) { phase += dt / PERIOD; probe = 0.92 * Math.sin(2 * Math.PI * phase); }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = "round"; ctx.lineJoin = "round";

      function axes(panel, lo, hi, label) {
        const cx = curveX(panel, fs), t = panel.y + fs * 1.5, b = panel.y + panel.h - fs * 1.7;
        const X = (x) => cx.l + ((x + 1) / 2) * (cx.r - cx.l), Y = (y) => b - ((y - lo) / (hi - lo)) * (b - t);
        ctx.strokeStyle = rule; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx.l, t - 4); ctx.lineTo(cx.l, b); ctx.lineTo(cx.r, b); ctx.stroke();
        if (lo < 0 && hi > 0) { ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(cx.l, Y(0)); ctx.lineTo(cx.r, Y(0)); ctx.stroke(); ctx.setLineDash([]); }
        ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`; ctx.textAlign = "center";
        ctx.fillText("−1", X(-1), b + small + 3); ctx.fillText("0", X(0), b + small + 3); ctx.fillText("1", X(1), b + small + 3);
        ctx.font = `${fs}px ${fmath}`; ctx.textAlign = "left"; ctx.fillStyle = ink; ctx.fillText(label, panel.x + 6, panel.y + fs * 0.95);
        return { X, Y, t, b, l: cx.l, r: cx.r };
      }
      function range(arrs) { let lo = Infinity, hi = -Infinity; for (const a of arrs) for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); } const pad = 0.08 * (hi - lo || 1); return [lo - pad, hi + pad]; }
      function line(ax, ys, color, width, dash) {
        ctx.save(); ctx.beginPath(); ctx.rect(ax.l, ax.t - 6, ax.r - ax.l, ax.b - ax.t + 6); ctx.clip();
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath();
        cur.tt.forEach((x, i) => (i ? ctx.lineTo(ax.X(x), ax.Y(ys[i])) : ctx.moveTo(ax.X(x), ax.Y(ys[i])))); ctx.stroke(); ctx.restore();
      }

      // --- u(theta) -------------------------------------------------------------------------------
      const LV = cur.R.map((r) => r[0]), LD = cur.R.map((r) => r[1]), HV = cur.R.map((r) => r[2]), HD = cur.R.map((r) => r[3]);
      let [lo, hi] = range([cur.U]); const span = hi - lo; lo -= 0.25 * span; hi += 0.25 * span;
      const a1 = axes(A, lo, hi, "𝑢(𝜃)");
      line(a1, cur.U, ink, 2.2); line(a1, LV, blue, 1.4); line(a1, HV, red, 1.4, [6, 4]);
      // stored tangents at the nodes
      const seg = 0.07;
      ctx.strokeStyle = red; ctx.lineWidth = 1.6;
      cur.xs.forEach((x, i) => { ctx.beginPath(); ctx.moveTo(a1.X(x - seg), a1.Y(cur.fv[i] - seg * cur.fd[i])); ctx.lineTo(a1.X(x + seg), a1.Y(cur.fv[i] + seg * cur.fd[i])); ctx.stroke(); });
      ctx.fillStyle = ink; cur.xs.forEach((x, i) => { ctx.beginPath(); ctx.arc(a1.X(x), a1.Y(cur.fv[i]), 2.8, 0, 2 * Math.PI); ctx.fill(); });
      // probe and tangent lines
      const pr = cur.f(probe), pu = P.u(probe), pd = P.du(probe), half = 0.28;
      ctx.save(); ctx.beginPath(); ctx.rect(a1.l, a1.t - 6, a1.r - a1.l, a1.b - a1.t + 6); ctx.clip();
      ctx.strokeStyle = rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a1.X(probe), a1.t); ctx.lineTo(a1.X(probe), a1.b); ctx.stroke();
      for (const [v, d, c, w] of [[pu, pd, ink, 1.4], [pr[0], pr[1], blue, 1.4], [pr[2], pr[3], red, 1.4]]) {
        ctx.strokeStyle = c; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(a1.X(probe - half), a1.Y(v - half * d)); ctx.lineTo(a1.X(probe + half), a1.Y(v + half * d)); ctx.stroke();
      }
      ctx.restore();
      // legend
      ctx.font = `${small}px ${ftext}`; ctx.textAlign = "right";
      const lgx = A.x + A.w - 14, lgy = A.y + fs * 0.95;
      const leg = [["exact solve", ink, []], ["Lagrange", blue, []], ["Hermite", red, [6, 4]]];
      let xx = lgx;
      for (let i = leg.length - 1; i >= 0; i--) {
        const [t, c, d] = leg[i]; ctx.fillStyle = c; ctx.fillText(t, xx, lgy); const tw = ctx.measureText(t).width;
        ctx.strokeStyle = c; ctx.lineWidth = 1.6; ctx.setLineDash(d); ctx.beginPath(); ctx.moveTo(xx - tw - 24, lgy - small * 0.33); ctx.lineTo(xx - tw - 6, lgy - small * 0.33); ctx.stroke(); ctx.setLineDash([]);
        xx -= tw + 36;
      }

      // --- du/dtheta ------------------------------------------------------------------------------
      [lo, hi] = range([cur.DU]); const sp2 = hi - lo; lo -= 0.3 * sp2; hi += 0.3 * sp2;
      const a2 = axes(B, lo, hi, "∂𝑢/∂𝜃");
      line(a2, cur.DU, ink, 2.2); line(a2, LD, blue, 1.4); line(a2, HD, red, 1.4, [6, 4]);
      ctx.fillStyle = red; cur.xs.forEach((x, i) => { const y = a2.Y(cur.fd[i]); if (y >= a2.t && y <= a2.b) { ctx.beginPath(); ctx.arc(a2.X(x), y, 2.8, 0, 2 * Math.PI); ctx.fill(); } });
      ctx.strokeStyle = rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a2.X(probe), a2.t); ctx.lineTo(a2.X(probe), a2.b); ctx.stroke();
      ctx.font = `${fs}px ${fmath}`; ctx.textAlign = "right"; ctx.fillStyle = muted; ctx.fillText("𝜃", a2.r, a2.b - 6);

      // --- errors against Q -------------------------------------------------------------------------
      const em = { l: C.x + fs * 3.6, r: C.x + C.w - 10, t: C.y + fs * 3.4, b: C.y + C.h - fs * 2.6 };
      const QX = (q) => em.l + (q / QMAX) * (em.r - em.l), LY = (lg) => em.t + (-lg / 16) * (em.b - em.t);
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(em.l, em.t); ctx.lineTo(em.l, em.b); ctx.lineTo(em.r, em.b); ctx.stroke();
      ctx.fillStyle = muted; ctx.font = `${small}px ${ftext}`; ctx.textAlign = "right";
      const YT = { 0: "1", "-4": "10⁻⁴", "-8": "10⁻⁸", "-12": "10⁻¹²", "-16": "10⁻¹⁶" };
      for (const lg of [0, -4, -8, -12, -16]) { ctx.fillText(YT[lg], em.l - 6, LY(lg) + small * 0.35); }
      ctx.textAlign = "center";
      for (const q of [0, 8, 16, 24, 32]) ctx.fillText(String(q), QX(q), em.b + small + 4);
      ctx.font = `${fs}px ${ftext}`;
      ctx.fillText("nodes Q (nonlinear solves)", (em.l + em.r) / 2, em.b + small + fs + 8);
      ctx.save(); ctx.translate(C.x + fs * 0.9, (em.t + em.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("relative error", 0, 0); ctx.restore();
      function series(data, color, filled) {
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.2;
        for (const [q, e] of data) { ctx.beginPath(); ctx.arc(QX(q), LY(Math.log10(e)), 2.6, 0, 2 * Math.PI); filled ? ctx.fill() : ctx.stroke(); }
      }
      series(cache.ld, blue, false); series(cache.lv, blue, true); series(cache.hd, red, false); series(cache.hv, red, true);
      for (const [s, c] of [[cache.lv, blue], [cache.hv, red]]) { const p = s[S.Q - 2]; ctx.strokeStyle = c; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(QX(p[0]), LY(Math.log10(p[1])), 6.5, 0, 2 * Math.PI); ctx.stroke(); }
      // legend for the error panel, above the axes
      ctx.font = `${small}px ${ftext}`; ctx.textAlign = "left";
      const ly0 = C.y + fs * 0.9, lx0 = em.l;
      [["Lagrange", blue, 0], ["Hermite", red, 1]].forEach(([t, c, r]) => {
        const y = ly0 + r * (small + 6);
        ctx.fillStyle = c; ctx.beginPath(); ctx.arc(lx0 + 4, y - small * 0.33, 2.8, 0, 2 * Math.PI); ctx.fill();
        ctx.fillText("value", lx0 + 12, y);
        const vw = ctx.measureText("value").width;
        ctx.strokeStyle = c; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(lx0 + 24 + vw, y - small * 0.33, 2.8, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillText("derivative, " + t, lx0 + 32 + vw, y);
      });
      ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.font = `${small}px ${fmath}`;
      ctx.fillText(`𝜌 = ${rhoOf(S.a, S.b).toFixed(3)}`, em.r, em.t + small);

      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function init() { document.querySelectorAll(".lm-herm").forEach(mount); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
