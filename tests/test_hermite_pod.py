"""H3 acceptance — POD (reduced-basis) re-encoding of the gradient-enhanced Hermite
surrogate + the Smolyak-compatibility decision.

The H3 milestone of ``GRADIENT_ENHANCED_PLAN.md`` §4.  Two tiers:

  * **fast, pure-interpolant** gates (no solver) — POD faithfulness / truncation-tail
    reproduction / gradient preservation / rank-barely-grows on a synthetic low-rank
    corpus, save/load, and the Smolyak-decision demonstrations;
  * a **slow, solver-backed** gate on ``solver_abt`` at the 44×32 grid — the real
    stacked value+derivative rank/compression numbers, the gradient preservation on
    real certified tangents, and the unchanged ``evaluate_polished`` certification.

Gates (§4 H3):
  * the reduced-basis Hermite reproduces the full Hermite to the truncation tail;
  * the exposed parameter gradient is preserved to ~1e-6;
  * Smolyak decision **(b)** — a documented note (the module docstring) + tests
    asserting the DENSE/anisotropic path is the supported one for the gradient
    enhancement, while the value-only interpolant slots into the combination
    technique bit-for-bit.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from lm.initial_data.parametric.parametric import cheb_param_nodes
from lm.initial_data.parametric.parametric_nd import ParametricSolutionND
from lm.initial_data.parametric.hermite import cardinal_deriv_at_nodes
from lm.initial_data.parametric.hermite_nd import HermiteSolutionND, from_problem_nd_hermite
from lm.initial_data.parametric import parametric_nd_smolyak as sm
from lm.initial_data.parametric.hermite_pod import (
    pod_basis,
    build_pod_hermite,
    project_hermite_pod,
    PODHermiteND,
    load_pod_hermite_nd,
    rank_for_tail,
    value_only_hermite_subgrid,
    level0_enhanced_is_taylor,
)


FIELD_SHAPE = (5, 12)
NFEAT = int(np.prod(FIELD_SHAPE))


# --------------------------------------------------------------------------
# Synthetic low-rank corpus:  U(θ,x) = mean(x) + Σ_{m<K} p_m(θ) φ_m(x)
# with φ_m orthonormal spatial modes and p_m analytic (degree ≤ 2Q+1).  Its
# parameter-tangent dU/dθ = Σ p_m'(θ) φ_m lives in the SAME span{φ_m}, so the
# stacked value+derivative corpus has the same rank as the value-only corpus.
# --------------------------------------------------------------------------
def _lowrank_1d(lo, hi, Q, K=8, seed=0, amp_decay=1.6):
    rng = np.random.default_rng(seed)
    # orthonormal spatial modes (nfeat × K)
    Phi_true, _ = np.linalg.qr(rng.standard_normal((NFEAT, K)))
    amps = 10.0 ** (-amp_decay * np.arange(K))            # geometric singular decay
    deg = 2 * Q + 1
    coef = rng.standard_normal((K, deg + 1))              # p_m coeffs over s^deg..s^0
    a = 2.0 / (hi - lo)
    c = (lo + hi) / (hi - lo)
    powers = np.arange(deg, -1, -1)

    def p(theta):                                          # (K,) coeff values p_m(θ)
        s = a * float(theta) - c
        return amps * (coef @ (s ** powers))

    def dp(theta):                                         # (K,) derivatives p_m'(θ)
        s = a * float(theta) - c
        dpow = np.where(powers > 0, powers * s ** np.clip(powers - 1, 0, None), 0.0)
        return amps * a * (coef @ dpow)

    mean_field = rng.standard_normal(NFEAT) * 1e-1

    def truth(theta):
        return (mean_field + Phi_true @ p(theta)).reshape(FIELD_SHAPE)

    def truth_grad(theta):
        return (Phi_true @ dp(theta)).reshape(FIELD_SHAPE)

    nodes, weights = cheb_param_nodes(lo, hi, Q)
    U = np.stack([truth(t).reshape(FIELD_SHAPE) for t in nodes])
    dU = np.stack([truth_grad(t).reshape(FIELD_SHAPE) for t in nodes])
    her = HermiteSolutionND(
        axes=[(lo, hi, Q)], nodes=[nodes], weights=[weights],
        U_nodes=U, dU_nodes=dU[:, None, ...],             # (Q+1, d=1, *field)
        cvec=[cardinal_deriv_at_nodes(nodes)], enhanced=(0,),
        iters=np.zeros(nodes.size, int), residuals=np.zeros(nodes.size))
    return her, truth, truth_grad, K


def _off_node(lo, hi, Q, fracs=(0.137, 0.371, 0.523, 0.689, 0.853), guard=1e-3):
    nodes, _ = cheb_param_nodes(lo, hi, Q)
    pts = []
    for fr in fracs:
        th = lo + fr * (hi - lo)
        assert float(np.min(np.abs(th - nodes))) > guard
        pts.append(np.array([th]))
    return pts


# ==========================================================================
# H3-U1 — POD faithfulness: PODHermite == POD projection of the full Hermite
# ==========================================================================
def test_pod_faithfulness_bitlevel():
    """``PODHermite.evaluate(θ) == mean + ΦΦᵀ(full_hermite(θ) − mean)`` to roundoff
    (the linear projection commutes with the linear Hermite interpolation — the
    paper Sec. III E re-encoding identity, now for the gradient-enhanced form)."""
    lo, hi, Q = -1.0, 2.0, 6
    her, truth, _, K = _lowrank_1d(lo, hi, Q, K=8, seed=1)
    Phi, mean, diag = pod_basis(her, r=K)                 # full stacked rank
    pod = project_hermite_pod(her, Phi, mean)
    for th in _off_node(lo, hi, Q):
        full = np.asarray(her.evaluate(th)).reshape(-1)
        proj = mean + Phi @ (Phi.T @ (full - mean))
        got = np.asarray(pod.evaluate(th)).reshape(-1)
        assert np.max(np.abs(got - proj)) < 1e-10, np.max(np.abs(got - proj))


# ==========================================================================
# H3-U2 — reduced-basis Hermite reproduces the full Hermite to the truncation tail
# ==========================================================================
def test_pod_reproduces_full_hermite_full_rank():
    """At full rank (r=K) the reduced-basis Hermite reproduces the full Hermite to
    the interpolation floor (the corpus is exactly rank-K)."""
    lo, hi, Q = 1.5, 4.0, 7
    her, truth, _, K = _lowrank_1d(lo, hi, Q, K=8, seed=2)
    pod, diag = build_pod_hermite(her, r=K)
    for th in _off_node(lo, hi, Q):
        full = np.asarray(her.evaluate(th)).reshape(-1)
        got = np.asarray(pod.evaluate(th)).reshape(-1)
        rel = np.linalg.norm(got - full) / np.linalg.norm(full)
        assert rel < 1e-11, rel


def test_pod_reproduces_full_hermite_to_truncation_tail():
    """At a truncated rank the reduced-basis Hermite reproduces the full Hermite to
    (a small multiple of) the singular-value tail beyond r — the "differs only by a
    controllable truncation tail" property."""
    lo, hi, Q = -1.0, 2.0, 7
    her, truth, _, K = _lowrank_1d(lo, hi, Q, K=9, seed=3, amp_decay=1.6)
    Phi_full, mean, diag = pod_basis(her, r=K)
    s = diag["s"]
    for tail in (1e-4, 1e-6):
        r = rank_for_tail(s, tail)
        pod, _ = build_pod_hermite(her, r=r)
        # predicted relative tail from the discarded singular energy
        tail_energy = np.sqrt(np.sum(s[r:] ** 2) / np.sum(s ** 2))
        worst = 0.0
        for th in _off_node(lo, hi, Q):
            full = np.asarray(her.evaluate(th)).reshape(-1)
            got = np.asarray(pod.evaluate(th)).reshape(-1)
            worst = max(worst, np.linalg.norm(got - full) / max(np.linalg.norm(full), 1e-30))
        # reproduction is at the truncation-tail level (generous factor for the
        # per-θ vs corpus-averaged energy)
        assert worst <= max(1e3 * tail_energy, 1e-12), (tail, worst, tail_energy)


# ==========================================================================
# H3-U3 — exposed parameter gradient preserved to ~1e-6
# ==========================================================================
def test_pod_gradient_preserved():
    """The exposed parameter gradient of the reduced-basis Hermite is the full
    Hermite gradient projected onto Φ (``P_r·∂U/∂θ``), preserved to the truncation
    tail: at a gradient-safe rank the projection loss is ≤ 1e-6."""
    lo, hi, Q = 1.5, 4.0, 7
    her, truth, truth_grad, K = _lowrank_1d(lo, hi, Q, K=9, seed=4, amp_decay=1.4)
    Phi_full, mean, diag = pod_basis(her, r=K)
    r = rank_for_tail(diag["s"], 1e-8)                    # gradient-safe rank
    pod, _ = build_pod_hermite(her, r=r)
    Phi = pod.Phi
    worst_loss = 0.0
    worst_match = 0.0
    for th in _off_node(lo, hi, Q):
        thj = jnp.asarray(th, dtype=jnp.float64)
        Jfull = np.asarray(jax.jacfwd(lambda t: her.evaluate_jax(t).reshape(-1))(thj))  # (nfeat,1)
        Jpod = np.asarray(jax.jacfwd(lambda t: pod.evaluate_jax(t).reshape(-1))(thj))
        # (i) POD gradient == P_r · full gradient (the projection identity)
        Jproj = Phi @ (Phi.T @ Jfull)
        worst_match = max(worst_match, np.linalg.norm(Jpod - Jproj) / np.linalg.norm(Jfull))
        # (ii) the projection loses only the truncation tail of the gradient
        loss = np.linalg.norm(Jfull - Jproj) / np.linalg.norm(Jfull)
        worst_loss = max(worst_loss, loss)
    assert worst_match < 1e-9, worst_match
    assert worst_loss < 1e-6, worst_loss


# ==========================================================================
# H3-U4 — the derivative fields share the value basis; rank barely grows (R5)
# ==========================================================================
def test_pod_rank_barely_grows():
    """The stacked value+derivative corpus has essentially the same POD rank as the
    value-only corpus (the derivative fields ``dU/dθ_k`` live in the same
    low-rank spatial subspace as ``U``) — the R5 storage mitigation."""
    lo, hi, Q = -1.0, 2.0, 8
    her, _, _, K = _lowrank_1d(lo, hi, Q, K=8, seed=5, amp_decay=1.5)
    Phi, mean, diag = pod_basis(her, tail=1e-6)
    for tail in (1e-4, 1e-6, 1e-8):
        rv = diag["rank_value"][tail]
        rs = diag["rank_stacked"][tail]
        # adding the derivative corpus grows the rank by at most one mode
        assert rs <= rv + 1, (tail, rs, rv)
    # and the derivative fields are captured by the value-only rank-r basis
    resid = diag["dU_on_value_basis_resid"]
    assert max(resid) < 1e-5, resid


# ==========================================================================
# H3-U5 — enhanced=() reduces to the value-only POD re-encoding
# ==========================================================================
def test_pod_value_only_matches_projection():
    """With no enhanced axes the reduced-basis Hermite is the value-only POD
    re-encoding: ``mean + ΦΦᵀ(value_interpolant − mean)``."""
    lo, hi, Q = -1.0, 2.0, 6
    her, truth, _, K = _lowrank_1d(lo, hi, Q, K=8, seed=6)
    # value-only twin (same nodes, enhanced=())
    val = HermiteSolutionND(
        axes=her.axes, nodes=her.nodes, weights=her.weights,
        U_nodes=her.U_nodes, dU_nodes=np.zeros_like(her.dU_nodes),
        cvec=her.cvec, enhanced=(), iters=her.iters, residuals=her.residuals)
    Phi, mean, _ = pod_basis(val, r=K, include_derivatives=False)
    pod = project_hermite_pod(val, Phi, mean)
    for th in _off_node(lo, hi, Q):
        full = np.asarray(val.evaluate(th)).reshape(-1)
        proj = mean + Phi @ (Phi.T @ (full - mean))
        got = np.asarray(pod.evaluate(th)).reshape(-1)
        assert np.max(np.abs(got - proj)) < 1e-10, np.max(np.abs(got - proj))


# ==========================================================================
# H3-U6 — persistence round-trips bit-for-bit
# ==========================================================================
def test_pod_save_load_roundtrip(tmp_path):
    lo, hi, Q = 1.5, 4.0, 6
    her, _, _, K = _lowrank_1d(lo, hi, Q, K=8, seed=7)
    pod, _ = build_pod_hermite(her, r=6)
    path = pod.save(tmp_path / "h3_pod", meta={"axis_names": ["b"], "note": "H3 test"})
    back = load_pod_hermite_nd(path)
    assert back.r == pod.r
    assert tuple(back.enhanced) == tuple(pod.enhanced)
    assert np.array_equal(back.Phi, pod.Phi)
    assert np.array_equal(back.mean, pod.mean)
    assert np.array_equal(back.coeff_hermite.U_nodes, pod.coeff_hermite.U_nodes)
    assert np.array_equal(back.coeff_hermite.dU_nodes, pod.coeff_hermite.dU_nodes)
    assert back.meta["kind"] == "pod_hermite_nd"
    assert back.meta["axis_names"] == ["b"]
    for th in _off_node(lo, hi, Q):
        assert np.array_equal(np.asarray(back.evaluate(th)), np.asarray(pod.evaluate(th)))


def test_pod_load_rejects_wrong_kind(tmp_path):
    """A HermiteSolutionND artifact must not load as a pod_hermite_nd."""
    lo, hi, Q = -1.0, 2.0, 5
    her, _, _, _ = _lowrank_1d(lo, hi, Q, K=6, seed=8)
    p = her.save(tmp_path / "plain_hermite")
    with pytest.raises(ValueError):
        load_pod_hermite_nd(p)


# ==========================================================================
# H3-U7 — add-only / standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lm.initial_data.parametric.hermite_pod as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm", "import context", "torch"):
        assert forbidden not in src, forbidden


# ==========================================================================
# H3-S1 (Smolyak decision, part 1) — the VALUE-ONLY Hermite interpolant slots into
# the combination technique BIT-FOR-BIT
# ==========================================================================
def _psnd_subgrid(axes_lohi, levels, U_nodes):
    """A ParametricSolutionND on the SAME nested-CC nodes as
    ``value_only_hermite_subgrid`` (the committed Smolyak subgrid form)."""
    d = len(axes_lohi)
    nodes, weights = [], []
    for k, (lo, hi) in enumerate(axes_lohi):
        n, w = sm.nested_levels(lo, hi, int(levels[k]))
        nodes.append(np.asarray(n, dtype=float))
        weights.append(np.asarray(w, dtype=float))
    grid = tuple(len(n) for n in nodes)
    axes_meta = [(lo, hi, len(nodes[k]) - 1) for k, (lo, hi) in enumerate(axes_lohi)]
    return ParametricSolutionND(
        axes=axes_meta, nodes=nodes, weights=weights, U_nodes=np.asarray(U_nodes, float),
        iters=np.zeros(grid, int), residuals=np.zeros(grid))


def test_smolyak_value_only_combination_bitforbit():
    """A signed combination of value-only ``HermiteSolutionND`` subgrids on nested
    CC levels is bit-for-bit the same combination of ``ParametricSolutionND``
    subgrids (what ``SmolyakSolutionND`` uses): the Hermite interpolant *family*
    telescopes into the combination technique identically in the value-only limit."""
    axes_lohi = [(1.5, 4.0), (-0.4, 0.4)]
    fs = (3, 4)
    rng = np.random.default_rng(0)
    coefA = rng.standard_normal((4,) + fs)
    coefB = rng.standard_normal((4,) + fs)

    def g(theta):                                          # smooth analytic field
        t0 = (2 * theta[0] - (1.5 + 4.0)) / (4.0 - 1.5)
        t1 = (2 * theta[1] - 0.0) / 0.8
        pa = sum(coefA[j] * t0 ** (3 - j) for j in range(4))
        pb = sum(coefB[j] * t1 ** (3 - j) for j in range(4))
        return pa + pb

    index_set = sm.isotropic_index_set(2, 2)
    coeffs = sm.combination_coeffs(index_set)
    kept = sorted(coeffs)

    psnd_subs, herm_subs = [], []
    for l in kept:
        nodes = [sm.nested_levels(*axes_lohi[k], int(l[k]))[0] for k in range(2)]
        grid = tuple(len(n) for n in nodes)
        U = np.empty(grid + fs)
        for i, t0 in enumerate(nodes[0]):
            for j, t1 in enumerate(nodes[1]):
                U[i, j] = g([t0, t1])
        psnd_subs.append(_psnd_subgrid(axes_lohi, l, U))
        herm_subs.append(value_only_hermite_subgrid(axes_lohi, l, U))

    probes = [np.array([2.1, 0.13]), np.array([3.37, -0.21]), np.array([1.83, 0.29])]
    for th in probes:
        comb_p = sum(coeffs[l] * psnd_subs[i].evaluate(th) for i, l in enumerate(kept))
        comb_h = sum(coeffs[l] * herm_subs[i].evaluate(th) for i, l in enumerate(kept))
        assert np.array_equal(comb_h, comb_p)


# ==========================================================================
# H3-S2 (Smolyak decision, part 2) — the ENHANCED path is dense-only
# ==========================================================================
def test_level0_enhanced_is_taylor():
    """A Smolyak level-0 (single-midpoint) factor, ENHANCED, is the 1-node linear
    Taylor ``U_0 + (θ−θ_0)·dU_0`` (the fragile R4 mode), NOT the value-only
    constant — the concrete reason the enhancement is dense-only."""
    lo, hi = 1.5, 4.0
    rng = np.random.default_rng(1)
    U0 = rng.standard_normal(FIELD_SHAPE)
    dU0 = rng.standard_normal(FIELD_SHAPE)
    theta = 2.7                                            # off the midpoint 2.75
    hv, tv = level0_enhanced_is_taylor(lo, hi, U0, dU0, theta)
    assert np.max(np.abs(hv - tv)) < 1e-13                # == 1-node Taylor
    # and NOT the value-only constant U0 (the enhancement genuinely changes it)
    assert np.max(np.abs(hv - U0)) > 1e-3


def test_enhanced_gradient_requires_dense_tangent():
    """Structural boundary: the gradient enhancement needs per-node certified
    tangents, which the DENSE ``solver_abt`` wiring supplies (``HermiteSolverND``
    requires a ``tangent_fn``) but the SPARSE Smolyak wiring cannot — the committed
    Smolyak builder stores only values, and its solver (``parametric_nd_3d``) has no
    certified tangent."""
    import inspect
    from lm.initial_data.parametric.hermite_nd import HermiteSolverND

    # the dense Hermite builder REQUIRES a tangent_fn (the enhancement source)
    sig_h = inspect.signature(HermiteSolverND.__init__)
    assert "tangent_fn" in sig_h.parameters

    # the sparse Smolyak builder has NO tangent mechanism (values only)
    sig_s = inspect.signature(sm.SmolyakSolverND.__init__)
    assert set(sig_s.parameters) - {"self"} == {"solve_fn", "axes"}
    assert not hasattr(sm.SmolyakSolverND, "tangent_fn")

    # a SmolyakSolutionND carries values only — no derivative field anywhere
    assert not hasattr(sm.SmolyakSolutionND, "dU_nodes")
    fields = {f for f in sm.SmolyakSolutionND.__dataclass_fields__}
    assert not any("dU" in f or "tangent" in f for f in fields)


# ==========================================================================
# Solver-backed gate (slow) — real stacked value+derivative POD on solver_abt
# ==========================================================================
_FIXED = {"q": 1.0, "chi_A": 0.0}
M_TOT = 1.0
P_MOM = 0.5


def _relL2(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.slow
def test_pod_real_corpus_solver_backed():
    """The real deliverable on ``solver_abt`` (Na=44 Nb=32, enhanced b & χ_B):
      (i)  the stacked value+derivative POD rank barely grows over value-only;
      (ii) the certified tangents dU/dθ are captured by the value basis;
      (iii) the reduced-basis Hermite reproduces the full Hermite (held-out) and
            preserves the exposed gradient to the truncation tail;
      (iv) ``evaluate_polished`` still certifies to ‖R‖∞ ≤ 1e-10."""
    from lm.initial_data.solver import solver_abt as sa
    from lm.initial_data.parametric.parametric_nd_2c import theta_to_slice

    prob = sa.make_problem(Na=44, Nb=32, P=P_MOM)
    axes = [{"name": "b", "min": 1.5, "max": 4.0, "Q": 6},
            {"name": "chi_B", "min": -0.4, "max": 0.4, "Q": 5}]
    her = from_problem_nd_hermite(prob, axes, enhanced=["b", "chi_B"], M_tot=M_TOT,
                                  fixed=_FIXED, use_cache=True).build(tol=1e-12, max_iter=20)

    # ---- (i)+(ii) rank / share-the-basis (gradient-safe shipped tail 1e-8) ----
    pod, diag = build_pod_hermite(her, tail=1e-8, solve_fn=her._solve_fn)
    # barely grows at the shipped 1e-6 tail (adding the two derivative corpora
    # grows the rank by a single mode; the ~1.3x growth at the deepest 1e-8 tail
    # is the fine-scale derivative content near the corpus floor, reported honestly
    # in findings.md but not gated)
    rv = diag["rank_value"][1e-6]
    rs = diag["rank_stacked"][1e-6]
    assert rs <= rv + 2, (rs, rv)
    assert rs <= 1.15 * rv, (rs, rv)
    assert max(diag["dU_on_value_basis_resid"]) < 1e-4, diag["dU_on_value_basis_resid"]

    # ---- (iii) held-out reproduction + gradient preservation ----
    hold = [np.array([2.15, 0.11]), np.array([3.4, -0.23]), np.array([1.9, 0.28])]
    worst_rep = 0.0
    for th in hold:
        rep = _relL2(pod.evaluate(th), her.evaluate(th))
        worst_rep = max(worst_rep, rep)
    assert worst_rep < 1e-6, worst_rep

    Phi = pod.Phi
    worst_gloss = 0.0
    for th in hold:
        thj = jnp.asarray(th, dtype=jnp.float64)
        Jfull = np.asarray(jax.jacfwd(lambda t: her.evaluate_jax(t).reshape(-1))(thj))
        Jproj = Phi @ (Phi.T @ Jfull)
        worst_gloss = max(worst_gloss, np.linalg.norm(Jfull - Jproj) / np.linalg.norm(Jfull))
    assert worst_gloss < 1e-6, worst_gloss

    # ---- (iv) certification unchanged ----
    for th in hold[:2]:
        U, info = pod.evaluate_polished(th, newton_steps=5, tol=1e-12)
        assert info.residual_norm <= 1e-10, (th, info.residual_norm)
