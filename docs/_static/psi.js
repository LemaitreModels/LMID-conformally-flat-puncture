/* The conformal factor psi of conformally flat binary-black-hole initial data on the orbital
   plane, as delivered by the parametric model: psi = psi_BL + u, with the Brill–Lindquist part
   computed exactly here and the regular part u interpolated from a table of the model's output
   (psi-data.js, window.LM_PSI_DATA). Drawn as a colour map with contours on a flat, tilted disc.
   It has no sliders of its own: figure 10 (spin.js) hosts it below its sheet and shares its
   sliders, through window.LMPsi.attach. No dependencies. */
(function () {
  "use strict";

  function token(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }
  function rgb(hex) { const n = parseInt(hex.replace("#", ""), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }

  // Colour maps: stops from flat space (t = 0) to deep in a throat (t = 1).
  const MAPS = { "Prussian ink": { stops: ["#faf9f5", "#c9d6e3", "#2c5a85", "#14263a"], lines: "ink" } };
  const LEVELS = [1.06, 1.08, 1.1, 1.125, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.65, 1.8, 2, 2.3, 2.6, 3, 3.5, 4];  // contours of psi
  const VMIN = Math.log(0.04), VMAX = Math.log(0.8);  // colour range of log(psi - 1)

  function decode(D) {           // float32 little-endian: mean, Phi (rank x ng^2), coeffs
    const bin = atob(D.data), buf = new ArrayBuffer(bin.length), b8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) b8[i] = bin.charCodeAt(i);
    const f = new Float32Array(buf), F = D.meta.ng * D.meta.ng, R = D.meta.rank;
    return { mean: f.subarray(0, F), Phi: f.subarray(F, F + R * F), C: f.subarray(F + R * F) };
  }

  // Draws into a canvas placed in `root` before `before`, for the parameters in the shared object
  // P = {q, d, chiA, chiB} (d the coordinate separation, holes placed about the centre of mass as
  // in figure 10). Returns the redraw function; the host calls it when P changes.
  function attach(root, P, before) {
    const D = window.LM_PSI_DATA;
    if (!D) return () => {};
    const M = D.meta, NG = M.ng, S = M.span, Bn = M.b, Qn = M.q, Cn = M.chi, FR = NG * NG, R = M.rank, NC = Cn.length;
    const T = decode(D);
    const canvas = document.createElement("canvas");
    canvas.className = "lm-psi-canvas";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Colour map of the conformal factor of the two black holes on their orbital plane, from the parametric model");
    root.insertBefore(canvas, before);

    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1;
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = root.clientWidth;
      H = Math.round(Math.min(W * 0.42, 400));
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
      draw();
    }

    const L = 12;          // radius of the disc in units of M: as figure 10, so the panels align
    const TILT = 0.4;      // elevation of the viewer, radians: as figures 09 and 10
    const N = 220;         // colour-map resolution across the disc
    const off = document.createElement("canvas"); off.width = off.height = N;
    const octx = off.getContext("2d");
    const img = octx.createImageData(N, N);

    // u for the current parameters on the model's (s, t) = (x/b, z/b) grid: the POD coefficients
    // interpolated linearly in log b and q, and by 3-point Lagrange on the nearest spin nodes
    // (the scheme the table's error was measured for), then decoded.
    function lin(x, n) {
      let i = 0; while (i < n.length - 2 && x > n[i + 1]) i++;
      const f = Math.min(1, Math.max(0, (x - n[i]) / (n[i + 1] - n[i])));
      return [[i, 1 - f], [i + 1, f]];
    }
    function quad(x, n) {
      let i = 0, best = Infinity;
      n.forEach((v, k) => { if (Math.abs(v - x) < best) { best = Math.abs(v - x); i = k; } });
      i = Math.min(n.length - 3, Math.max(0, i - 1));
      const m = n.slice(i, i + 3);
      return [0, 1, 2].map((a) => {
        const [o, p] = [0, 1, 2].filter((c) => c !== a);
        return [i + a, (x - m[o]) * (x - m[p]) / ((m[a] - m[o]) * (m[a] - m[p]))];
      });
    }
    const G = new Float32Array(FR), c = new Float64Array(R), lnB = Bn.map(Math.log);
    function field(b) {
      c.fill(0);
      for (const [i, wi] of lin(Math.log(b), lnB)) for (const [j, wj] of lin(P.q, Qn))
        for (const [k, wk] of quad(P.chiA, Cn)) for (const [l, wl] of quad(P.chiB, Cn)) {
          const w = wi * wj * wk * wl, base = (((i * Qn.length + j) * NC + k) * NC + l) * R;
          for (let r = 0; r < R; r++) c[r] += w * T.C[base + r];
        }
      G.set(T.mean);
      for (let r = 0; r < R; r++) { const w = c[r], o = r * FR; for (let p = 0; p < FR; p++) G[p] += w * T.Phi[o + p]; }
    }
    // Bilinear on the table's non-uniform (s, t) nodes (dense at the punctures); outside the
    // table, u falls off as 1/r.
    const SN = M.s_nodes, TN = M.t_nodes;
    function cell(v, n) {
      let lo = 0, hi = n.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (n[m] <= v) lo = m; else hi = m; }
      return [lo, Math.min(1, Math.max(0, (v - n[lo]) / (n[lo + 1] - n[lo])))];
    }
    function uAt(x, z, b) {
      let s = x / b, t = z / b, f = 1;
      const r = Math.max(Math.abs(s), Math.abs(t));
      if (r > S) { f = S / r; s *= f; t *= f; }
      const [i, a] = cell(s, SN), [j, c] = cell(t, TN), p = j * NG + i;
      return f * ((1 - a) * (1 - c) * G[p] + a * (1 - c) * G[p + 1] + (1 - a) * c * G[p + NG] + a * c * G[p + NG + 1]);
    }

    function draw() {
      if (!W) return;
      const b = Math.min(Bn[Bn.length - 1], Math.max(Bn[0], P.d / 2)), mA = P.q / (1 + P.q), mB = 1 / (1 + P.q);
      const zc = b * (mA - mB);       // the screen is centred on the centre of mass, as figure 10
      field(b);
      const ink = token(root, "--lm-ink", "#1f1e1c"), paper = token(root, "--lm-paper", "#faf9f5");
      const dark = rgb(ink).reduce((s, v) => s + v, 0) > rgb(paper).reduce((s, v) => s + v, 0);
      const map = MAPS["Prussian ink"];
      const stops = map.stops ? map.stops.map(rgb) : null;
      if (stops && dark) { stops.reverse(); stops[0] = rgb(paper); }   // dark: flat space is the paper, throats glow
      const ramp = (t, st) => {
        t = Math.min(1, Math.max(0, t)) * (st.length - 1);
        const i = Math.min(st.length - 2, Math.floor(t)), f = t - i;
        return [0, 1, 2].map((c) => st[i][c] + f * (st[i + 1][c] - st[i][c]));
      };
      const red = rgb(token(root, "--lm-accent2", "#b04632"));

      // psi on the disc. Horizontal on screen is the line through the holes (z), depth is x.
      const psi = new Float32Array(N * N);
      let umax = 1e-12;
      const ug = new Float32Array(N * N);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const z = (i + 0.5) / N * 2 * L - L + zc, x = (j + 0.5) / N * 2 * L - L;
        const rA = Math.hypot(x, z - b), rB = Math.hypot(x, z + b);
        const u = uAt(x, z, b);
        ug[j * N + i] = u; umax = Math.max(umax, Math.abs(u));
        psi[j * N + i] = 1 + mA / (2 * rA) + mB / (2 * rB) + u;
      }
      const px = img.data;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = j * N + i, o = 4 * k;
        const r = Math.hypot((i + 0.5) / N * 2 - 1, (j + 0.5) / N * 2 - 1);
        const fade = r >= 1 ? 0 : r < 0.8 ? 1 : 1 - Math.pow((r - 0.8) / 0.2, 2) * (3 - 2 * (r - 0.8) / 0.2);
        if (!stops) { px[o + 3] = 0; continue; }
        let c = ramp((Math.log(Math.max(psi[k] - 1, 1e-9)) - VMIN) / (VMAX - VMIN), stops);
        if (map.diverge) {             // tint by the model's correction u, relative to its peak here
          const a = 0.75 * Math.min(1, Math.abs(ug[k]) / umax);
          c = c.map((v, ch) => v + a * (red[ch] - v));
        }
        px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255 * fade;
      }
      octx.putImageData(img, 0, 0);

      const ct = Math.sin(TILT), scale = W / (2 * L) * 0.97, cx = W / 2, cy = H * 0.5;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.save();
      ctx.translate(cx, cy); ctx.scale(scale * 2 * L / N, scale * 2 * L / N * ct); ctx.translate(-N / 2, -N / 2);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      ctx.drawImage(off, 0, 0);
      ctx.restore();

      // Contours by marching squares, drawn in screen space so the lines keep their weight.
      const scr = (gi, gj) => [cx + ((gi + 0.5) / N * 2 * L - L) * scale, cy + ((gj + 0.5) / N * 2 * L - L) * scale * ct];
      ctx.lineWidth = 0.7; ctx.lineJoin = "round";
      LEVELS.forEach((lv, n) => {
        const path = new Path2D();
        for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
          const ri = Math.hypot((i + 1) / N * 2 - 1, (j + 1) / N * 2 - 1);
          if (ri > 0.97) continue;
          const v = [psi[j * N + i], psi[j * N + i + 1], psi[(j + 1) * N + i + 1], psi[(j + 1) * N + i]];
          const P0 = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], pts = [];
          for (let e = 0; e < 4; e++) {
            const a = v[e] - lv, c = v[(e + 1) % 4] - lv;
            if ((a < 0) !== (c < 0)) {
              const f = a / (a - c), p = P0[e], q = P0[(e + 1) % 4];
              pts.push(scr(p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])));
            }
          }
          for (let e = 0; e + 1 < pts.length; e += 2) { path.moveTo(pts[e][0], pts[e][1]); path.lineTo(pts[e + 1][0], pts[e + 1][1]); }
        }
        if (map.lines === "ramp") {
          const c = ramp(n / (LEVELS.length - 1), [rgb(token(root, "--lm-accent", "#2c5a85")), red]);
          ctx.strokeStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.globalAlpha = 0.9; ctx.lineWidth = 1;
        } else {
          ctx.strokeStyle = ink; ctx.globalAlpha = 0.35; ctx.lineWidth = 0.6;
        }
        ctx.stroke(path);
      });
      ctx.globalAlpha = 1;

      // Label, top right above the disc's rim, in the accent blue: "LM's psi".
      const x0 = W * 0.93, y0 = cy - 0.12 * W;
      ctx.fillStyle = token(root, "--lm-accent", "#2c5a85"); ctx.textBaseline = "alphabetic"; ctx.textAlign = "right";
      ctx.font = "italic " + 1.15 * parseFloat(getComputedStyle(root).fontSize) + "px " + token(root, "--lm-font-math", "'Latin Modern Math', serif");
      ctx.fillText("\u03C8", x0, y0);
      const wpsi = ctx.measureText("\u03C8").width;
      ctx.font = "italic " + getComputedStyle(root).fontSize + " " + token(root, "--lm-font-text", "system-ui, sans-serif");
      ctx.fillText("LM\u2019s ", x0 - wpsi, y0);

    }

    new ResizeObserver(resize).observe(root);
    resize();
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
    return draw;
  }

  window.LMPsi = { attach: attach, range: () => window.LM_PSI_DATA && window.LM_PSI_DATA.meta };
})();
