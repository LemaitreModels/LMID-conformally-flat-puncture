/* Interactive spinning binary black hole on one time slice: the see-through sheet of figure 08,
   with each hole's spin shown as frame dragging, a twist of the grid lines about that hole.
   Sliders: mass ratio, separation and the two aligned spins. With psi.js (figure 11) loaded first,
   the parametric model's conformal factor is drawn below the sheet, on the same sliders. Mounts into every element with
   class "lm-spin". No dependencies. Colours come from the
   Lemaitre style tokens (--lm-ink, --lm-rule) when present. */
(function () {
  "use strict";

  function token(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function mount(root) {
    root.innerHTML = "";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Two spinning black holes on a curved grid of space, the grid twisted around each by its spin");
    const controls = document.createElement("div");
    controls.className = "lm-spin-controls";
    root.append(canvas, controls);

    const P = { q: 2.0, d: 8.0, chiA: 0.6, chiB: -0.4 };
    // The sliders drift on their own around a centre: the default, or wherever the visitor last
    // left that slider. Each wanders by up to AMP of its range as a sum of two slow sines with
    // incommensurate periods, so the motion is smooth and never repeats. Touching a slider makes it
    // the new centre and stops the drift for PAUSE_MS; then it eases back in over RAMP_MS, starting
    // exactly at the centre. Visitors who ask for reduced motion get no drift.
    const PERIOD = { q: [14, 37], d: [17, 43], chiA: [11, 27], chiB: [12, 31] };   // seconds
    const AMP = 0.12, PAUSE_MS = 2500, RAMP_MS = 3000;
    const auto = { theta: {}, centre: {}, ui: {}, pauseUntil: 0, last: performance.now(), psiAt: 0, psiCost: 20 };
    const stillMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const sliders = [
      ["q", "mass ratio", 1, 3, 0.05, (v) => "q = " + v.toFixed(2)],
      ["d", "separation", 6, 20, 0.1, (v) => "d = " + v.toFixed(1) + " M"],
      ["chiA", "spin A", -0.9, 0.9, 0.05, (v) => "\u03C7A = " + (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(2)],
      ["chiB", "spin B", -0.9, 0.9, 0.05, (v) => "\u03C7B = " + (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(2)],
    ];
    for (const [key, label, lo, hi, step, fmt] of sliders) {
      const lab = document.createElement("label");
      const name = document.createElement("span");
      name.textContent = label;
      const input = document.createElement("input");
      // step "any": the drift moves the thumb continuously, not in slider steps.
      Object.assign(input, { type: "range", min: lo, max: hi, step: "any", value: P[key] });
      const out = document.createElement("output");
      out.textContent = fmt(P[key]);
      input.addEventListener("input", () => {
        P[key] = +input.value; out.textContent = fmt(P[key]); if (redrawPsi) redrawPsi();
        auto.pauseUntil = performance.now() + PAUSE_MS;
        for (const k in auto.ui) { auto.centre[k] = P[k]; auto.theta[k] = [0, 0]; }   // resume from here
      });
      lab.append(name, input, out);
      controls.append(lab);
      auto.ui[key] = { input, out, fmt, lo, hi };
    }
    // Figure 11, when loaded: the model's conformal factor on the orbital plane, drawn below the
    // sheet and driven by the same sliders. The ranges above are inside its model's box.
    let redrawPsi = null;
    if (window.LMPsi) redrawPsi = window.LMPsi.attach(root, P, controls);
    for (const k in auto.ui) { auto.centre[k] = P[k]; auto.theta[k] = [0, 0]; }
    function drift(now) {
      const dt = Math.min(0.1, (now - auto.last) / 1000); auto.last = now;
      if (stillMotion || now < auto.pauseUntil) return;
      const ease = Math.min(1, (now - auto.pauseUntil) / RAMP_MS), speed = ease * ease * (3 - 2 * ease);
      for (const k in auto.ui) {
        const u = auto.ui[k];
        const th = auto.theta[k];
        th[0] += speed * dt * 2 * Math.PI / PERIOD[k][0];
        th[1] += speed * dt * 2 * Math.PI / PERIOD[k][1];
        const off = AMP * (u.hi - u.lo) * (0.65 * Math.sin(th[0]) + 0.35 * Math.sin(th[1]));
        P[k] = Math.min(u.hi, Math.max(u.lo, auto.centre[k] + off));
        u.input.value = P[k]; u.out.textContent = u.fmt(P[k]);
      }
      // The psi panel costs a few ms to tens of ms: redraw it as often as it keeps up, leaving
      // at least as much time again for the sheet.
      if (redrawPsi && now - auto.psiAt > 2 * auto.psiCost) {
        const t0 = performance.now(); redrawPsi();
        auto.psiCost = 0.8 * auto.psiCost + 0.2 * (performance.now() - t0); auto.psiAt = now;
      }
    }


    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1;
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = root.clientWidth;
      H = Math.round(Math.min(W * 0.56, 540));
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
    }
    new ResizeObserver(resize).observe(root);
    resize();

    const L = 12;                  // radius of the sheet in units of M
    const DX = 0.33;                // grid spacing in units of M
    const TILT = 0.4;              // elevation of the viewer, radians: same as figure 09
    const ZMIN = -5.4;             // depth at which the funnels are cut off
    const TWIST = 6.0;             // strength of the drawn frame dragging (radians, schematic)
    const phase = 0;               // initial data: the holes sit still, side by side on the x axis

    function frame(now) {
      drift(now);
      const m1 = P.q / (1 + P.q), m2 = 1 / (1 + P.q);
      const r1 = P.d * m2, r2 = P.d * m1;                    // distances from the centre of mass
      const c = Math.cos(phase), sn = Math.sin(phase);
      const x1 = r1 * c, y1 = r1 * sn, x2 = -r2 * c, y2 = -r2 * sn;

      // Height: -(psi - 1) of the Brill–Lindquist conformal factor, which diverges at each
      // puncture (cut off at ZMIN). One time slice of initial data, so no ripples.
      function z(x, y) {
        const a = Math.hypot(x - x1, y - y1, 0.05), b = Math.hypot(x - x2, y - y2, 0.05);
        const psi = 1 + m1 / (2 * a) + m2 / (2 * b);
        return Math.max(ZMIN, -4.5 * (psi - 1));
      }

      // Frame dragging: each grid point is turned about each hole by an angle that grows as the
      // spin J = chi m^2 and falls off as 1/rho^2 (schematic; the Lense–Thirring rate falls as
      // 1/rho^3). Hole A is the heavier one, m1. The sense of the twist follows the spin's sign.
      const holesSpin = [[x1, y1, m1, P.chiA], [x2, y2, m2, P.chiB]];
      function drag(x, y) {
        for (const [hx, hy, hm, chi] of holesSpin) {
          const dx = x - hx, dy = y - hy, rho2 = dx * dx + dy * dy;
          const ang = TWIST * chi * hm * hm / (rho2 + 0.25 * hm * hm);
          const ca = Math.cos(ang), sa = Math.sin(ang);
          x = hx + dx * ca - dy * sa; y = hy + dx * sa + dy * ca;
        }
        return [x, y];
      }

      const ink = token(root, "--lm-ink", "#1f1e1c");
      const rule = token(root, "--lm-rule", "#dcd8cc");

      const ct = Math.cos(TILT), st = Math.sin(TILT);
      const scale = W / (2 * L) * 0.97;
      const cam = 60;                                         // perspective distance, units of M
      function proj(x, y, zz) {
        const up = y * st + zz * ct, depth = -y * ct + zz * st;   // depth > 0 is nearer the viewer
        const f = cam / (cam - depth);
        return [W / 2 + x * scale * f, H * 0.46 - up * scale * f, depth];
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);                             // the page shows through
      ctx.lineJoin = "round";
      ctx.strokeStyle = ink;
      ctx.lineWidth = 0.6;

      // A see-through sheet cut to a disc: grid lines only, finely sampled so the funnels stay
      // sharp. Each line is the chord of the disc at its offset, fading towards the rim. The grid
      // is turned 45 degrees about the vertical.
      const r45 = Math.SQRT1_2;
      // Each segment goes into one of NB alpha bands by its distance from the centre, so the sheet
      // fades out towards its rim (and the funnels, near the centre, stay fully drawn).
      const step = 0.08, NB = 48, bands = [];
      // ...and into one of NZ bands by depth, so the lines also fade down each throat.
      const NZ = 16;
      for (let k = 0; k < NB * NZ; k++) bands.push(new Path2D());
      const zBand = (zz) => Math.min(NZ - 1, Math.max(0, Math.floor((-zz / -ZMIN) * NZ)));
      const zAlpha = (j) => { const s = Math.min(1, Math.max(0, ((j + 0.5) / NZ - 0.15) / 0.85)); return 1 - s * s * (3 - 2 * s); };
      const alphaOf = (k) => {                                // flat to 55 % of the radius, then a smoothstep to 0
        const t = Math.min(1, Math.max(0, ((k + 0.5) / NB - 0.55) / 0.45));
        return 0.6 * (1 - t * t * (3 - 2 * t));
      };
      for (let pass = 0; pass < 2; pass++) {
        for (let u = -L; u <= L + 1e-9; u += DX) {
          const half = Math.sqrt(Math.max(0, L * L - u * u));
          if (half < step) continue;
          let prev = null;
          for (let v = -half; v <= half + 1e-9; v += step) {
            const vv = Math.min(v, half), a = pass ? u : vv, b = pass ? vv : u, x = (a - b) * r45, y = (a + b) * r45;
            const zz = z(x, y), [xd, yd] = drag(x, y), q = proj(xd, yd, zz);
            const r = Math.hypot(a, b);
            if (prev) {
              const k = Math.min(NB - 1, Math.floor(((r + prev[2]) / 2 / L) * NB));
              const kb = k * NZ + zBand((zz + prev[3]) / 2);
              bands[kb].moveTo(prev[0], prev[1]); bands[kb].lineTo(q[0], q[1]);
            }
            prev = [q[0], q[1], r, zz];
          }
        }
      }
      for (let k = 0; k < NB; k++) for (let j = 0; j < NZ; j++) {
        ctx.globalAlpha = alphaOf(k) * zAlpha(j); ctx.stroke(bands[k * NZ + j]);
      }
      ctx.globalAlpha = 1;

      // Label, top right above the sheet's rim.
      ctx.font = "italic " + getComputedStyle(root).fontSize + " " + token(root, "--lm-font-text", "system-ui, sans-serif");
      ctx.fillStyle = ink; ctx.textAlign = "right"; ctx.textBaseline = "alphabetic";
      ctx.fillText("AI\u2019s impression", W * 0.93, H * 0.46 - 0.12 * W);

      // Horizons: black balls of the isolated isotropic radius m_i/2, sitting in their funnels,
      // drawn far one first. The rim keeps them visible in dark mode.
      const holes = [[x1, y1, m1 / 2], [x2, y2, m2 / 2]].sort((u, v) => v[1] - u[1]);
      for (const [hx, hy, hm] of holes) {
        const zc = z(hx + hm * 2.1, hy);
        const p = proj(hx, hy, zc);
        const R = hm * scale * 1.4;
        ctx.beginPath();
        ctx.arc(p[0], p[1], R, 0, 2 * Math.PI);
        ctx.fillStyle = "#000000"; ctx.fill();
        ctx.strokeStyle = rule; ctx.lineWidth = 0.8; ctx.stroke();
      }

      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function init() { document.querySelectorAll(".lm-spin").forEach(mount); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
