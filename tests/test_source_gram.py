"""Acceptance tests — the Â² Gram-tensor route (the OPT-IN twin of the source).

``source_3d.A2_at_nodes_3d`` rebuilds the summed Bowen–York tensor at every
node for every parameter point, although the tensor is exactly linear in the
twelve momentum/spin components — so Â² is an exact quadratic form
``qᵀ G(x) q`` with ``G`` depending only on (grid, b).  What is pinned here:

  * **The quadratic-form identity.**  The Gram route equals the tensor route to
    the last ulps for arbitrary vector momenta and spins — the same float64
    products regrouped, so agreement is a few 1e-16 relative, not exact.  That
    regrouping is also why the route is OPT-IN: no default may move even one
    ulp without a cross-leaf decision (the curved sibling pins this package's
    arithmetic bit-for-bit).
  * **The default route is untouched.**  ``assemble`` without ``a2_gram``
    produces the tensor route's Â² bit-for-bit.
  * **Masking, caching, immutability.**  Non-finite (BC-edge) nodes give
    exactly 0 for every θ, as in the tensor route; the tensor is built once per
    (grid, b) and shared read-only.
  * **The exact tangent.**  ``G`` is symmetric, so the quadratic identity
    ``Â²(q+dq) − Â²(q) − Â²(dq) = 2 (G q)·dq`` holds to roundoff — the hook
    that replaces per-direction tensor rebuilds in the sensitivity modules.
  * **End to end.**  A certified Newton–Krylov solve on a Gram-route assembly
    certifies under the existing monitor and lands on the same field.
"""

import numpy as np

from lemaitre.initial_data.conformally_flat_puncture.solver import operators_3d as ops3
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d_nk as nk
from lemaitre.initial_data.conformally_flat_puncture.solver import source_3d
from lemaitre.initial_data.conformally_flat_puncture.solver.solver_3d import Slice3D


GRID = dict(Na=16, Nb=12, Nphi=6)


def _slice(b=2.5):
    """A genuinely non-axisymmetric slice: off-axis momenta + misaligned spins."""
    return Slice3D(b=b, m_A=0.6, m_B=0.4,
                   P_A_vec=(0.04, 0.11, -0.02), P_B_vec=(-0.04, -0.11, 0.02),
                   S_A_vec=(0.05, 0.0, 0.03), S_B_vec=(0.0, 0.04, -0.01))


def _cloud(b):
    prob = s3.make_problem(**GRID)
    rho, z, _, _, _ = ops3.meridian_geometry(prob.A, prob.B, b)
    return prob, rho, z


def test_gram_matches_the_tensor_route():
    """The identity itself: qᵀGq = Â² to a few ulps, for random loadings and b.

    Judged against the tensor route in the norm that matters downstream — the
    worst deviation relative to max|Â²| — because entries near the masked edge
    are tiny and an elementwise relative there would measure roundoff noise,
    not the identity.
    """
    rng = np.random.default_rng(5)
    for b in (1.5, 2.5, 7.0):
        _, rho, z = _cloud(b)
        phi = ops3.phi_grid(GRID["Nphi"])
        for _ in range(5):
            vecs = rng.standard_normal((4, 3)) * 0.3
            ref = source_3d.A2_at_nodes_3d(rho, z, phi, b, *vecs)
            got = source_3d.A2_gram_at_nodes_3d(rho, z, phi, b, *vecs)
            rel = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
            print(f"\n[a2-gram] b={b}: |gram-tensor|/max|A2| = {rel:.2e}")
            assert rel < 1e-13, f"Gram route off by {rel:.2e} (b={b})"


def test_masked_nodes_are_exactly_zero():
    """BC-edge (non-finite) nodes carry Â² = 0 for EVERY θ, as in the tensor route."""
    b = 2.5
    _, rho, z = _cloud(b)
    phi = ops3.phi_grid(GRID["Nphi"])
    finite = np.isfinite(rho) & np.isfinite(z)
    assert not finite.all(), "test needs the A=1 edge in the cloud"
    sl = _slice(b)
    got = source_3d.A2_gram_at_nodes_3d(rho, z, phi, b, sl.P_A_vec, sl.P_B_vec,
                                        sl.S_A_vec, sl.S_B_vec)
    assert np.all(got[~finite] == 0.0)


def test_gram_tensor_is_cached_and_read_only():
    """One build per (grid, b), shared read-only; a new b is a new tensor."""
    b = 2.5
    _, rho, z = _cloud(b)
    phi = ops3.phi_grid(GRID["Nphi"])
    source_3d.clear_a2_gram_cache()
    G1 = source_3d.a2_gram_tensor(rho, z, phi, b)
    G2 = source_3d.a2_gram_tensor(rho, z, phi, b)
    assert G1 is G2
    assert G1 is not source_3d.a2_gram_tensor(rho, z, phi, 2.0 * b)
    try:
        G1[0, 0, 0, 0] = 1.0
        raised = False
    except ValueError:
        raised = True
    assert raised, "the shared Gram tensor must be read-only"
    # symmetric in the loading indices — what makes 2(Gq)·dq the exact tangent
    assert np.array_equal(G1, np.swapaxes(G1, 2, 3))


def test_assemble_default_route_is_untouched():
    """``assemble`` without ``a2_gram`` is the tensor route BIT-FOR-BIT; with it,
    the Gram route bit-for-bit.  The opt-in must not perturb the default."""
    prob = s3.make_problem(**GRID)
    sl = _slice()
    rho, z, _, _, _ = ops3.meridian_geometry(prob.A, prob.B, sl.b)
    args = (rho, z, prob.phi, sl.b, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec)
    asm = s3.assemble(prob, sl)
    assert np.array_equal(asm.A2, source_3d.A2_at_nodes_3d(*args))
    asm_g = s3.assemble(prob, sl, a2_gram=True)
    assert np.array_equal(asm_g.A2, source_3d.A2_gram_at_nodes_3d(*args))


def test_exact_tangent_identity():
    """``Â²(q+dq) − Â²(q) − Â²(dq) = 2 (G q)·dq`` to roundoff — no finite
    differences involved, because the form is exactly quadratic."""
    b = 2.5
    _, rho, z = _cloud(b)
    phi = ops3.phi_grid(GRID["Nphi"])
    rng = np.random.default_rng(6)
    q = rng.standard_normal(12) * 0.3
    dq = rng.standard_normal(12) * 0.1
    G = source_3d.a2_gram_tensor(rho, z, phi, b)

    def a2(v):
        return (G @ v) @ v

    tangent = 2.0 * ((G @ q) @ dq)
    ref = a2(q + dq) - a2(q) - a2(dq)
    scale = np.max(np.abs(a2(q)))
    rel = np.max(np.abs(tangent - ref)) / scale
    print(f"\n[a2-gram-tangent] |2(Gq)·dq − exact expansion|/max|A2| = {rel:.2e}")
    assert rel < 1e-13


def test_certified_solve_on_the_gram_route():
    """End to end: a Gram-route assembly certifies under the existing monitor and
    lands on the default route's field (the ulp-level Â² shift moves the solve
    well below the certification scale)."""
    prob = s3.make_problem(**GRID)
    sl = _slice()
    U_ref, i_ref = nk.newton_solve_nk(prob, sl, tol=1e-11, max_iter=15)
    asm = s3.assemble(prob, sl, separable=True, a2_gram=True)
    U_g, i_g = nk.newton_solve_nk(prob, sl, tol=1e-11, max_iter=15, asm=asm)
    assert i_g.converged
    d = np.max(np.abs(np.asarray(U_g) - np.asarray(U_ref))) / np.max(np.abs(U_ref))
    print(f"\n[a2-gram-solve] certified {i_g.residual_norm:.2e} "
          f"(ref {i_ref.residual_norm:.2e}), |U_gram−U_ref|/|U| = {d:.2e}")
    assert d < 1e-9
