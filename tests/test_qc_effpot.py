"""``qc_effpot.eccentricity`` must never evaluate the surrogate ON a box edge.

Both edges of the production ``b`` box are *endpoint* Chebyshev--Lobatto nodes of
the model's ``b`` axis (``production_box.B_MIN, B_MAX = 3.0, 10.0``, and the
shipped ``surrogate_bpt_ecc.npz`` carries nodes at exactly 3.0 and 10.0).  There
the barycentric quotient of ``parametric_nd.evaluate_jax`` is 0/0, so ``V``
returns NaN -- which is why every other evaluator in ``qc_effpot`` nudges through
``_off_node`` before it evaluates.

``eccentricity`` did not.  It handed ``box_b[0]`` straight to ``brentq`` as a
bracket endpoint, so on the retargeted box every rung of the caller's ``b0``
ladder (``run_qc_effpot.py``) came back unmeasurable -- silently, because the
routine's documented contract is to return NaN for a turning point that leaves
the box, and an all-NaN ladder is indistinguishable from "not measurable here".
It never fired on the historical (2.6, 6.4) window, whose edges sat strictly
inside the model box.

These tests stub the potential rather than solving anything: ``_effpot_jit``
caches ``(V, dV, d2V)`` on the model keyed by ``id(prob)``, so seeding that cache
installs an analytic parabola whose second turning point is known in closed form.
For ``V(b) = (b - b_circ)^2`` the other root of ``V(b) = V(b0)`` is
``b' = 2 b_circ - b0``, giving ``e = |b0 - b'|/(b0 + b') = |b0 - b_circ|/b_circ``.
"""
import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.applications import qc_effpot as E

BOX_B = (3.0, 10.0)
B_CIRC = 6.0
# a b axis whose ENDPOINT nodes are the box edges, as a Chebyshev-Lobatto axis is
B_NODES = np.array([3.0, 3.6, 4.9, 6.5, 8.1, 9.4, 10.0])
PT_NODES = np.array([0.0, 0.4, 0.8])      # J/(2b) stays in ~[0.05, 0.17] here
_NUDGE = E._NODE_SHIFT_FRAC * (BOX_B[1] - BOX_B[0])   # 0.014 in b


class _StubModel:
    """Carries the two things ``_off_node`` and ``_effpot_jit`` read off a model."""

    def __init__(self):
        self.nodes = [B_NODES, PT_NODES]


def _install(calls, b_circ=B_CIRC):
    """A model+prob whose ``V`` is a parabola, NaN exactly on a b node.

    NaN *only* at exact coincidence, which is what the real 0/0 barycentric
    quotient does -- a point merely near a node evaluates fine.
    """
    model, prob = _StubModel(), object()

    def V(b, J):
        b = float(b)
        calls.append(b)
        if np.any(B_NODES == b):
            return np.nan
        return (b - b_circ) ** 2

    model.__dict__["_effpot_jit_cache"] = {id(prob): (prob, (V, None, None))}
    return model, prob


@pytest.mark.parametrize("b0", [8.0, 8.9, 7.25])
def test_outer_apsis_is_measurable_though_the_lower_edge_is_a_node(b0):
    """``b0 > b_circ`` brackets on ``[box_b[0], b_circ]`` -- the failing branch."""
    calls = []
    model, prob = _install(calls)
    e, bp = E.eccentricity(model, prob, b0, J=1.0, b_circ=B_CIRC, box_b=BOX_B)
    assert np.isfinite(e), "the lower box edge being a node made the rung unmeasurable"
    assert bp == pytest.approx(2 * B_CIRC - b0, abs=1e-9)
    assert e == pytest.approx(abs(b0 - B_CIRC) / B_CIRC, abs=1e-9)


def test_inner_apsis_is_measurable_though_the_upper_edge_is_a_node():
    """``b0 < b_circ`` brackets on ``[b_circ, box_b[1]]``.

    The same defect on the other branch: ``box_b[1] = 10.0`` is the upper
    endpoint node.  The shipped caller only ever sweeps ``b0 >= b_circ``, so this
    half was latent -- it would have fired the first time anyone swept inward.
    """
    calls = []
    model, prob = _install(calls)
    b0 = 4.0
    e, bp = E.eccentricity(model, prob, b0, J=1.0, b_circ=B_CIRC, box_b=BOX_B)
    assert np.isfinite(e)
    assert bp == pytest.approx(2 * B_CIRC - b0, abs=1e-9)
    assert e == pytest.approx(abs(b0 - B_CIRC) / B_CIRC, abs=1e-9)


def test_the_potential_is_never_evaluated_on_a_node():
    """The property behind both fixes, stated directly.

    This is the assertion that would have caught the bug as a bug rather than as
    a puzzling all-NaN column: it fails on the unguarded code even if the NaN
    happened to be swallowed downstream.
    """
    for b0 in (8.0, 4.0):
        calls = []
        model, prob = _install(calls)
        E.eccentricity(model, prob, b0, J=1.0, b_circ=B_CIRC, box_b=BOX_B)
        assert calls, "V was never evaluated"
        on_node = [b for b in calls if np.any(B_NODES == b)]
        assert not on_node, f"V evaluated on the node(s) {on_node}"


def test_unmeasurable_turning_point_still_returns_nan():
    """The guard must not turn an honest NaN into a fabricated number.

    With ``b_circ`` close to the lower edge the second turning point falls
    outside the box; the contract is ``(nan, nan)``, never a silent ``b' = b0``
    (which would report a perfectly circular orbit -- the bug 201d7e7 removed).
    """
    calls = []
    model, prob = _install(calls, b_circ=3.5)
    e, bp = E.eccentricity(model, prob, 9.0, J=1.0, b_circ=3.5, box_b=BOX_B)
    assert np.isnan(e) and np.isnan(bp)


def test_the_nudge_costs_a_sliver_at_the_very_edge():
    """The documented price of the guard, pinned so it cannot drift silently.

    ``_NODE_SHIFT_FRAC`` is sized for *derivative* accuracy (2e-3 of the span --
    here 0.014 in ``b``), so the bracket starts at 3.014 rather than 3.0 and a
    second turning point inside that sliver is reported unmeasurable rather than
    located.  That is the conservative direction, but it is a real blind spot:
    ``b' = 3.007`` is inside the box and outside the search.
    """
    bp_target = 3.007
    b0 = 2 * B_CIRC - bp_target                     # 8.993, comfortably in box
    assert BOX_B[0] < bp_target < BOX_B[0] + _NUDGE
    calls = []
    model, prob = _install(calls)
    e, bp = E.eccentricity(model, prob, b0, J=1.0, b_circ=B_CIRC, box_b=BOX_B)
    assert np.isnan(e) and np.isnan(bp)
