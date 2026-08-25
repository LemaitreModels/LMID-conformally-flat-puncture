"""Acceptance tests for the certification gate — the residual check that stands
between a solve and a returned datum.

The paper states that every queried parameter is refined "until the discrete
constraint residual satisfies ``‖R‖∞ ≤ 1e-10``, a gate checked before the datum
is returned".  Before this module that sentence was half true: every
``evaluate_polished`` *measured* the residual of the field it returned and
handed it back as ``info.residual_norm``, but nothing compared it to anything.
Worse, the default ``tol`` was ``1e-12`` — below the equilibrated residual's
roundoff floor — so ``info.converged`` was False on every default-tol query, the
solver's ``if rn < tol: break`` was unreachable, and the one status flag a caller
might have checked reported nothing.

What must hold:

* the gate threshold is single-sourced (:data:`certification.CERT_TOL_U`, the
  live ``u``-norm gate, beside the historical ``v``-norm :data:`certification.CERT_TOL`) and is
  the default ``tol`` of **every** ``evaluate_polished`` — a regression here is
  how the ``1e-12`` default would creep back;
* ``strict=True`` refuses to return an uncertified datum, on every model class;
* ``strict=False`` (the default) still returns it, because the polish-history
  producers query *below* the gate deliberately and a sweep wants a bad point
  flagged rather than the sweep lost;
* a NaN residual — a diverged solve, not an under-converged one — fails the gate
  rather than passing it;
* the shipped-model query path wires the damped-Newton globalization on;
* **the gate is actually closed at each exit** — ``qc_targeting._certified_solve``,
  ``qc_targeting.gauss_newton_target`` and ``qc_effpot.Eb_certified`` each refuse
  to hand back an uncertified datum. This is the half that makes the paper's
  sentence true of the package rather than only of its bookkeeping, and it is
  otherwise uncovered: ``test_qc_targeting.py`` exercises only the observable and
  Jacobian algebra, and there is no ``test_qc_effpot.py``.

These stub the *solver* but run the real gate (``certified_return``) and the real
application control flow, so the whole module runs in about a second: the thing
under test is the gate, not the elliptic solve.
"""
import inspect
import math

import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.parametric import (
    certification, hermite, hermite_nd, hermite_pod, hermite_smolyak,
    hermite_smolyak_pod, hermite_smolyak_pod_cross, parametric, parametric_nd,
    parametric_nd_smolyak,
)
from lemaitre.initial_data.conformally_flat_puncture.parametric.certification import (
    CERT_TOL, CERT_TOL_U, CertificationError, certified_return, gate_for,
)

# Every class that exposes the `evaluate_polished` contract.  `hermite_smolyak_cross
# .HermiteCrossSolutionND` inherits it verbatim from `HermiteSmolyakSolutionND` and so
# is covered by that entry.
POLISHED_CLASSES = [
    parametric.ParametricSolution,
    parametric_nd.ParametricSolutionND,
    parametric_nd_smolyak.SmolyakSolutionND,
    hermite.HermiteSolution1D,
    hermite_nd.HermiteSolutionND,
    hermite_pod.PODHermiteND,
    hermite_smolyak.HermiteSmolyakSolutionND,
    hermite_smolyak_pod.PODHermiteSmolyak,
    hermite_smolyak_pod_cross.PODHermiteSmolyakCross,
]

_FIELD = np.zeros((3, 3))


class _StubInfo:
    """The three attributes the gate reads off a solver's info object."""

    def __init__(self, residual_norm, iters=2):
        self.residual_norm = residual_norm
        self.iters = iters
        self.converged = residual_norm < CERT_TOL   # stub: untagged solve_fn -> v gate


def _stub_model(cls, residual_norm):
    """An instance of ``cls`` whose solver always achieves ``residual_norm``.

    Bypasses ``__init__`` (these are heavy dataclasses carrying node pools and POD
    bases) and shadows the two attributes ``evaluate_polished`` actually touches:
    its own ``evaluate`` and the injected ``_solve_fn``.  So the code under test is
    the real method body, including the real gate wiring.
    """
    obj = object.__new__(cls)
    obj.evaluate = lambda theta: _FIELD
    obj._solve_fn = lambda theta, guess, tol, steps: (_FIELD, _StubInfo(residual_norm))
    return obj


def _theta_for(cls):
    """The 1-D layers take a scalar parameter; the N-D layers take a vector."""
    if cls in (parametric.ParametricSolution, hermite.HermiteSolution1D):
        return 5.0
    return np.array([5.0, 1.0])


# ==========================================================================
# C1 — the threshold is single-sourced and is everyone's default
# ==========================================================================
def test_cert_tol_is_the_papers_gate():
    """Both thresholds, and the fact that they are different numbers in different norms.

    :data:`CERT_TOL_U` is the live gate — ``1e-11`` on the equilibrated residual in
    the ``u`` norm, Frederik's value of 2026-08-25.  :data:`CERT_TOL` is its
    ``v``-norm predecessor at its published ``1e-10``, kept under a distinct name
    because the **submitted** flat paper's eleven residual numbers were measured
    against it and a reversibility constraint says no published number is orphaned.

    **They are not comparable and no single factor converts them** — the measured
    ``v/u`` ratio on field-converged solves runs ``1×`` at ``Nφ = 1`` to ``6.6e+08``
    at ``36×24×20``.  And ``CERT_TOL_U`` is **not a tightening**: at the production
    grid it is ``≈1.7e-08`` in ``v``-equivalent terms, ~170× looser.
    """
    assert CERT_TOL_U == 1e-11, "the live u-norm gate moved"
    assert CERT_TOL == 1e-10, (
        "the historical v-norm gate moved; it is the submitted paper's number and "
        "must not be re-tuned")
    assert CERT_TOL_U != CERT_TOL, (
        "the two thresholds collapsed onto one value; they are in different norms")


@pytest.mark.parametrize("cls", POLISHED_CLASSES, ids=lambda c: c.__name__)
def test_evaluate_polished_defaults_to_the_gate(cls):
    """`tol` defaults to *the attached solver's* gate, and the gate is opt-in.

    **The default is `None`, and that is the fix rather than a regression.**  It used
    to be the one threshold, because there was one norm.  Since 2026-08-25 there are
    two and a container is generic — the same class carries a 2-D ABT solve_fn or a
    3-D Newton–Krylov one, and those report residuals in different norms — so the
    threshold cannot live in this signature.  `None` resolves through
    :func:`certification.gate_for`, which reads it off the solver.  A default below
    the equilibrated roundoff floor (the pre-2026-08-13 `1e-12`) would still make
    `info.converged` meaningless, and `gate_for` cannot return one.
    """
    sig = inspect.signature(cls.evaluate_polished)
    assert sig.parameters["tol"].default is None, (
        f"{cls.__name__}.evaluate_polished pins tol to "
        f"{sig.parameters['tol'].default!r} in its signature; it must defer to "
        f"gate_for(solve_fn), because which norm the residual is in is a property "
        f"of the attached solver and not of the container")
    assert sig.parameters["strict"].default is False


def test_the_gate_follows_the_solver_not_the_container():
    """`gate_for` gives the 3-D NK path `CERT_TOL_U` and everything else `CERT_TOL`.

    **The axisymmetric half is the subtle one and it cost a test.** `solver_abt`'s
    2-D path has no `Nφ`, hence no `(1 − B²)^{−m/2}` factor: it *is* the `Nφ = 1`
    case, where the two norms are bit-identical. The norm flip is a no-op there, so
    the threshold must not move either — applying `1e-11` is not a change of units
    but a genuine `10×` tightening, and it showed up as `test_certified_polish_4d`
    spending a third Newton step against an internal target a decade below where it
    was calibrated.

    An undeclared solver is assumed axisymmetric because that is the **conservative**
    direction: a `u`-norm solver that forgets to declare itself gets a gate `170×`
    too tight in `v`-equivalent terms and fails loudly, rather than one four orders
    too loose that passes silently.
    """
    from lemaitre.initial_data.conformally_flat_puncture.parametric import (
        parametric_nd_3d as p3d)
    from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3

    undeclared = lambda *a: None
    assert gate_for(undeclared) == CERT_TOL, (
        "an undeclared solve_fn must fall back to the v-norm gate, which is the "
        "norm-invariant axisymmetric case and the conservative choice")

    sf, _ = p3d.make_solve_fn(s3.make_problem(Na=20, Nb=16, Nphi=4), ["b"])
    got = gate_for(sf)
    print(f"\n[gate-for] 3-D NK solve_fn -> {got:.1e} (u)   "
          f"undeclared -> {gate_for(undeclared):.1e} (v)")
    assert got == CERT_TOL_U, (
        f"the 3-D Newton-Krylov solve_fn resolves to {got:.1e}, not the u-norm gate "
        f"{CERT_TOL_U:.1e} — its monitor is EQUIL_NORM_DEFAULT and a v-calibrated "
        f"threshold read against a u residual is four orders too loose")


# ==========================================================================
# C2 — strict refuses to return an uncertified datum
# ==========================================================================
@pytest.mark.parametrize("cls", POLISHED_CLASSES, ids=lambda c: c.__name__)
def test_strict_returns_a_certified_datum(cls):
    U, info = _stub_model(cls, _CERTIFIED).evaluate_polished(_theta_for(cls), strict=True)
    assert np.shape(U) == _FIELD.shape
    assert info.residual_norm <= CERT_TOL      # the stub's solve_fn is untagged


@pytest.mark.parametrize("cls", POLISHED_CLASSES, ids=lambda c: c.__name__)
def test_strict_raises_on_an_uncertified_datum(cls):
    """The known failure mode: a global-convergence stall at an extreme corner
    leaves the residual on a plateau far above the gate."""
    theta = _theta_for(cls)
    with pytest.raises(CertificationError) as exc:
        _stub_model(cls, 3.2e-7).evaluate_polished(theta, strict=True)
    assert exc.value.residual_norm == 3.2e-7
    assert exc.value.tol == CERT_TOL, (
        "the stub attaches an UNTAGGED solve_fn, so gate_for resolves the v-norm "
        "gate -- the 3-D u-norm path is covered by "
        "test_the_gate_follows_the_solver_not_the_container")
    # the message must localise the failure, or a sweep failure is undiagnosable
    assert "3.200e-07" in str(exc.value)
    assert str(theta) in str(exc.value)


@pytest.mark.parametrize("cls", POLISHED_CLASSES, ids=lambda c: c.__name__)
def test_non_strict_still_returns_it(cls):
    """The deliberate default: measure and report, do not refuse.

    The polish-history producers (`run_polish_table.py`, `run_polish_cold.py`)
    sweep tol=1e-12 to watch the residual fall *through* 1e-10; a library-level
    raise would make that measurement impossible to take.
    """
    U, info = _stub_model(cls, 3.2e-7).evaluate_polished(_theta_for(cls))
    assert np.shape(U) == _FIELD.shape
    assert info.residual_norm == 3.2e-7


# ==========================================================================
# C3 — a diverged solve fails the gate rather than slipping through it
# ==========================================================================
def test_nan_residual_fails_the_gate():
    """`not (r <= tol)` rather than `r > tol`: every comparison against NaN is
    False, so the naive form would return a diverged field as certified."""
    with pytest.raises(CertificationError):
        certified_return(_FIELD, _StubInfo(math.nan), 5.0, CERT_TOL, strict=True)


def test_exactly_at_the_gate_passes():
    U, info = certified_return(_FIELD, _StubInfo(CERT_TOL), 5.0, CERT_TOL, strict=True)
    assert info.residual_norm == CERT_TOL


# ==========================================================================
# C4 — the shipped-model query path carries the globalization
# ==========================================================================
def test_attach_solve_fn_3d_defaults_the_globalization_on():
    """`make_solve_fn` defaults retry_tol=None so build-time behaviour is
    bit-for-bit unchanged; the *query* path of a loaded model must not inherit
    that, since it is the path the certification claim is made about."""
    from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd_3d import (
        make_solve_fn,
    )
    assert inspect.signature(make_solve_fn).parameters["retry_tol"].default is None
    attach = inspect.signature(parametric_nd.attach_solve_fn_3d)
    assert attach.parameters["retry_tol"].default == CERT_TOL_U


# ==========================================================================
# C5 — the gate is closed at every point where a datum leaves the package
# ==========================================================================
# A solve is stubbed out, but `prob` is real: the observables (M_ADM, the binding
# energy) differentiate the field against the actual spectral operators, so a
# shape or convention slip still shows up here.
_SMALL_GRID = dict(Na=8, Nb=6, Nphi=4)

# The 4-D χ box of `qc_targeting` (b, q, chi_Ay, chi_By) — the shipped axis order.
_BOX_QC = (np.array([6.0, 1.0, -0.6, -0.6]), np.array([12.0, 3.0, 0.6, 0.6]))
_THETA_QC = np.array([8.0, 1.5, 0.1, 0.1])

# RE-MEASURED for the u-norm gate, 2026-08-25.  `4e-11` was "comfortably inside" a
# 1e-10 gate and is 4x OUTSIDE the 1e-11 one -- a stub value calibrated to a
# threshold silently inverts when the threshold moves, which is why these are named
# and commented rather than written inline at each call site.
_CERTIFIED = 4e-12        # comfortably inside the u-norm gate (2.5x under)
_STALLED = 3.2e-7         # the observed high-residual plateau of a global stall


@pytest.fixture(scope="module")
def prob():
    from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3
    return s3.make_problem(**_SMALL_GRID)


class _StubModel:
    """A surrogate whose solver always achieves ``residual_norm``.

    ``evaluate_polished`` routes through the **real** :func:`certified_return`, so
    these tests exercise the production gate; only the elliptic solve is faked.
    Records each call so the wiring (did the caller actually pass ``strict``?) is
    checkable, not merely inferable from the fact that it raised.
    """

    def __init__(self, residual_norm, field):
        self.residual_norm = residual_norm
        self._field = np.asarray(field)
        self.calls = []

    def evaluate(self, theta):
        return self._field

    def evaluate_jax(self, theta):
        import jax.numpy as jnp
        return jnp.asarray(self._field)

    def evaluate_polished(self, theta, newton_steps=2, tol=None, strict=False):
        tol = CERT_TOL_U if tol is None else tol      # this stub stands in for a 3-D model
        self.calls.append({"tol": tol, "strict": strict, "newton_steps": newton_steps})
        return certified_return(self._field, _StubInfo(self.residual_norm),
                                theta, tol, strict)


def test_eb_certified_returns_a_certified_binding_energy(prob):
    from lemaitre.initial_data.conformally_flat_puncture.applications import qc_effpot

    model = _StubModel(_CERTIFIED, np.zeros(prob.shape))
    E_b, resid = qc_effpot.Eb_certified(model, prob, b=8.0, P_t=0.07)
    assert np.isfinite(E_b)
    assert resid == _CERTIFIED
    assert model.calls[-1]["strict"] is True, "Eb_certified must close the gate"
    assert model.calls[-1]["tol"] == CERT_TOL_U


def test_eb_certified_refuses_an_uncertified_binding_energy(prob):
    """E_b is differenced across neighbouring separations to locate the ISCO, so an
    uncertified U contaminates a difference of two nearly equal numbers."""
    from lemaitre.initial_data.conformally_flat_puncture.applications import qc_effpot

    model = _StubModel(_STALLED, np.zeros(prob.shape))
    with pytest.raises(CertificationError):
        qc_effpot.Eb_certified(model, prob, b=8.0, P_t=0.07)


def test_certified_solve_refuses_an_uncertified_field(prob, monkeypatch):
    """``_certified_solve`` calls the solver directly rather than through
    ``evaluate_polished``, so it needs its own gate — and its own test."""
    from lemaitre.initial_data.conformally_flat_puncture.applications import qc_targeting as T

    field = np.zeros(prob.shape)

    def _fake_nk(prob_, sl, U0=None, tol=CERT_TOL_U, max_iter=25, **kw):   # 3-D NK
        return field, _StubInfo(_fake_nk.residual)

    monkeypatch.setattr(T.nk, "newton_solve_nk", _fake_nk)

    _fake_nk.residual = _CERTIFIED
    U, rn = T._certified_solve(None, prob, _THETA_QC, None, CERT_TOL_U, 25)
    assert np.shape(U) == prob.shape and rn == _CERTIFIED

    _fake_nk.residual = _STALLED
    with pytest.raises(CertificationError):
        T._certified_solve(None, prob, _THETA_QC, None, CERT_TOL_U, 25)


def test_gauss_newton_target_refuses_to_emit_an_uncertified_configuration(prob):
    """The targeting loop reports ``certified_residual`` as a max over its solves.
    Before the gate, a stalled solve merely inflated that number in a Result the
    caller still received; now it stops the emission."""
    from lemaitre.initial_data.conformally_flat_puncture.applications import qc_targeting as T

    target = np.array([1.0, 0.5])
    kw = dict(target_names=("M_ADM", "J"), box=_BOX_QC, max_steps=1,
              correction_steps=0)

    good = _StubModel(_CERTIFIED, np.zeros(prob.shape))
    res = T.gauss_newton_target(good, prob, target, _THETA_QC, **kw)
    assert res.n_certified_solves == 1
    assert res.certified_residual == _CERTIFIED
    assert good.calls[-1]["strict"] is True, "the certified last mile must gate"

    bad = _StubModel(_STALLED, np.zeros(prob.shape))
    with pytest.raises(CertificationError):
        T.gauss_newton_target(bad, prob, target, _THETA_QC, **kw)


def test_polish_tol_defaults_are_the_gate():
    """The application defaults must not drift from the gate **of their own solver**.

    That drift is what made the paper's threshold and the code's threshold differ
    before.  Since 2026-08-25 the two 3-D targeting entry points take the ``u``-norm
    ``CERT_TOL_U`` while ``sensitivity`` -- the axisymmetric ABT cousin, where the
    norms coincide -- keeps ``CERT_TOL``.  Checking them against one constant is what
    the norm flip made wrong.
    """
    from lemaitre.initial_data.conformally_flat_puncture.applications import qc_targeting as T

    assert inspect.signature(T.gauss_newton_target).parameters["polish_tol"].default == CERT_TOL_U
    assert inspect.signature(T.broyden_target).parameters["tol_inner"].default == CERT_TOL_U

    # `sensitivity` is the axisymmetric ABT cousin of the same loop. Its default was
    # 1e-12 — below `solver_abt.newton_solve`'s ~1e-11 residual floor, so the
    # tolerance its own docstring claimed (1e-10) was not the one it used.
    from lemaitre.initial_data.conformally_flat_puncture.applications import sensitivity as S

    assert inspect.signature(S.gauss_newton_target).parameters["polish_tol"].default == CERT_TOL, (
        "sensitivity is the AXISYMMETRIC ABT cousin -- solver_abt has no Nphi, so "
        "the two norms coincide there and its gate must NOT have moved with the "
        "2026-08-25 flip; CERT_TOL_U would be a 10x tightening, not a re-norming")
