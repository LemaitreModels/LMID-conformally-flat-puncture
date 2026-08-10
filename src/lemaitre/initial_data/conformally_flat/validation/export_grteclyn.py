"""Export a converged solve to a plain file an evolution code can read.

Written for the GRTeclyn constraint check (``docs/GRTECLYN_CONSTRAINTS_PLAN.md``):
hand the initial data to an independent numerical-relativity code, let it compute
the Hamiltonian and momentum constraints with its own fourth-order stencils in
its own evolution variables, and report the convergence.  This module is the
only part of that plan that touches this repo, and it stays standalone — nothing
here imports the evolution code, and the reference evaluator below is
**numpy-only**, so it is an independent check of the file format rather than a
re-run of the solver.

WHAT IS EXPORTED.  Everything an evaluator needs to reconstruct

    psi(x)  =  psi_BL(x) + u(x),        psi_BL = 1 + m_A/2 r_A + m_B/2 r_B
    Ahat^{ij}(x)                        (closed-form Bowen-York, momenta+spins)

at an arbitrary Cartesian point, namely the geometry ``(b, m_A, m_B, P_A, P_B,
S_A, S_B)`` and the nodal correction ``u`` on the frozen ABT grid.  ``psi_BL``
and ``Ahat`` are closed forms, so only ``u`` is data.

THE PHI REPRESENTATION.  ``solver_3d.evaluate_field`` interpolates ``u`` with a
tensor-product barycentric rule in ``(A, B)`` followed by *trigonometric*
interpolation of the equispaced phi samples.  Rather than ship the nodal samples
and make the consumer redo an FFT, we ship the equivalent real cosine/sine
coefficients

    u(A, B, phi) = a_0(A,B) + sum_k [ a_k(A,B) cos(k phi) + b_k(A,B) sin(k phi) ]

so the consumer only sums cos/sin.  This is exact, not an approximation: the
transform in :func:`phi_modes` is the algebraic rewrite of
``solver_3d._fourier_interp`` for real samples, and the round-trip is gated to
machine precision in ``tests/test_export_grteclyn.py``.  Because the ``(A, B)``
interpolation is linear, interpolating the coefficients and then summing the
series gives the same number as interpolating per phi-plane and then
interpolating in phi.

FILE FORMAT (``format 1``).  A plain ASCII token stream: ``#`` comments, then
``<key> <values...>`` records.  Chosen over JSON/HDF5 so that the consumer needs
no parser library — ``ifstream >> token`` is enough — and so the format is
readable in a diff.  Keys:

    format b m_A m_B  P_A P_B S_A S_B  Na Nb Nphi  A B  cos_m sin_m  C S

``Na`` is the Chebyshev-Lobatto order, so ``A`` has ``Na+1`` nodes
(``A[0] = 1`` is spatial infinity, ``A[Na] = 0`` the inner axis edge); ``B`` has
``Nb`` Gauss-Legendre interior nodes.  ``C`` and ``S`` are flattened
``C[i][j][t]``, i.e. ``t`` fastest then ``j`` then ``i``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

FORMAT_VERSION = 1


# --------------------------------------------------------------------------
# phi: nodal samples <-> real cosine/sine coefficients
# --------------------------------------------------------------------------
def phi_mode_layout(nphi: int) -> Tuple[np.ndarray, np.ndarray]:
    """Wavenumbers of the cosine and sine terms of the trig interpolant.

    For ``Nphi`` equispaced real samples the interpolant that
    ``solver_3d._fourier_interp`` evaluates is

        a_0 + sum_{k=1}^{K} a_k cos k phi + sum_{k=1}^{K'} b_k sin k phi

    with ``K = Nphi//2`` and ``K' = (Nphi-1)//2``: for even ``Nphi`` the Nyquist
    wavenumber ``Nphi/2`` carries a cosine term but no sine term (its sine
    vanishes at every node and is not determined by the samples).
    """
    cos_m = np.arange(0, nphi // 2 + 1, dtype=int)
    sin_m = np.arange(1, (nphi - 1) // 2 + 1, dtype=int)
    return cos_m, sin_m


def phi_modes(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Real cos/sin coefficients of the trig interpolant of ``vals``.

    ``vals`` has shape ``(..., Nphi)`` (phi last, equispaced on ``[0, 2pi)``).
    Returns ``(C, S)`` with shapes ``(..., len(cos_m))`` and
    ``(..., len(sin_m))``.

    Derivation.  ``_fourier_interp`` returns ``Re sum_k c_k e^{i m_k phi} / N``
    with ``c = fft(vals)`` and ``m_k = fftfreq(N)*N``.  For real ``vals``,
    ``c_{N-k} = conj(c_k)`` and ``m_{N-k} = -k``, so the terms ``k`` and ``N-k``
    combine into ``2 Re(c_k e^{i k phi}) / N = 2[Re c_k cos k phi -
    Im c_k sin k phi]/N``.  Hence ``a_k = 2 Re c_k / N``, ``b_k = -2 Im c_k / N``
    for ``0 < k < N/2``; ``a_0 = c_0 / N``; and for even ``N`` the unpaired
    Nyquist term contributes ``a_{N/2} = Re c_{N/2} / N``.
    """
    vals = np.asarray(vals, dtype=float)
    nphi = vals.shape[-1]
    cos_m, sin_m = phi_mode_layout(nphi)
    c = np.fft.fft(vals, axis=-1)

    cos_c = np.empty(vals.shape[:-1] + (cos_m.size,), dtype=float)
    sin_c = np.empty(vals.shape[:-1] + (sin_m.size,), dtype=float)
    for t, k in enumerate(cos_m):
        if k == 0:
            cos_c[..., t] = c[..., 0].real / nphi
        elif 2 * k == nphi:                       # Nyquist (even Nphi only)
            cos_c[..., t] = c[..., k].real / nphi
        else:
            cos_c[..., t] = 2.0 * c[..., k].real / nphi
    for t, k in enumerate(sin_m):
        sin_c[..., t] = -2.0 * c[..., k].imag / nphi
    return cos_c, sin_c


def phi_eval(cos_c: np.ndarray, sin_c: np.ndarray, cos_m: np.ndarray,
             sin_m: np.ndarray, phi) -> np.ndarray:
    """Evaluate the cos/sin series at ``phi`` (broadcast over the leading axes)."""
    phi = np.asarray(phi, dtype=float)
    out = np.zeros(np.broadcast(cos_c[..., 0], phi).shape, dtype=float)
    for t, k in enumerate(cos_m):
        out = out + cos_c[..., t] * np.cos(k * phi)
    for t, k in enumerate(sin_m):
        out = out + sin_c[..., t] * np.sin(k * phi)
    return out


# --------------------------------------------------------------------------
# The exported record
# --------------------------------------------------------------------------
@dataclass
class Export:
    """One exported slice: geometry, the ABT grid, and the phi-modal ``u``."""
    b: float
    m_A: float
    m_B: float
    P_A: Tuple[float, float, float]
    P_B: Tuple[float, float, float]
    S_A: Tuple[float, float, float]
    S_B: Tuple[float, float, float]
    Na: int
    Nb: int
    Nphi: int
    A: np.ndarray                     # (Na+1,)
    B: np.ndarray                     # (Nb,)
    cos_m: np.ndarray
    sin_m: np.ndarray
    C: np.ndarray                     # (Na+1, Nb, ncos)
    S: np.ndarray                     # (Na+1, Nb, nsin)
    provenance: Sequence[str] = field(default_factory=tuple)

    # -- writing ------------------------------------------------------------
    def write(self, path: str) -> str:
        """Write the ASCII record; returns ``path``."""
        def fmt(a):
            return " ".join(f"{v:.17e}" for v in np.asarray(a).reshape(-1))

        lines = [f"# LM-initial-data GRTeclyn export, format {FORMAT_VERSION}"]
        lines += [f"# {ln}" for ln in self.provenance]
        lines += [
            "# psi = psi_BL + u ;  psi_BL = 1 + m_A/(2 r_A) + m_B/(2 r_B),",
            "# punctures A at (0,0,+b) and B at (0,0,-b).",
            "# u(A,B,phi) = sum_t C[i][j][t] cos(cos_m[t] phi)",
            "#            + sum_t S[i][j][t] sin(sin_m[t] phi)",
            "# with (A,B) by tensor-product barycentric interpolation, and",
            "# (A,B) <- (rho,z) by the closed-form inverse ABT map.",
            f"format {FORMAT_VERSION}",
            f"b {self.b:.17e}",
            f"m_A {self.m_A:.17e}",
            f"m_B {self.m_B:.17e}",
            f"P_A {fmt(self.P_A)}",
            f"P_B {fmt(self.P_B)}",
            f"S_A {fmt(self.S_A)}",
            f"S_B {fmt(self.S_B)}",
            f"Na {self.Na}",
            f"Nb {self.Nb}",
            f"Nphi {self.Nphi}",
            f"ncos {self.cos_m.size}",
            f"nsin {self.sin_m.size}",
            "cos_m " + " ".join(str(int(v)) for v in self.cos_m),
            "sin_m " + (" ".join(str(int(v)) for v in self.sin_m)
                        if self.sin_m.size else ""),
            "A " + fmt(self.A),
            "B " + fmt(self.B),
            "C " + fmt(self.C),
            "S " + (fmt(self.S) if self.S.size else ""),
        ]
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    # -- reading ------------------------------------------------------------
    @staticmethod
    def read(path: str) -> "Export":
        """Read a record written by :meth:`write`."""
        prov = []
        rec: dict = {}
        with open(path) as f:
            for raw in f:
                if raw.startswith("#"):
                    prov.append(raw[1:].strip())
                    continue
                tok = raw.split()
                if not tok:
                    continue
                rec[tok[0]] = tok[1:]

        def one(key, cast=float):
            return cast(rec[key][0])

        def arr(key, cast=float):
            return np.array([cast(v) for v in rec.get(key, [])], dtype=(
                float if cast is float else int))

        fmt_v = one("format", int)
        if fmt_v != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format {fmt_v}, this reader handles "
                f"{FORMAT_VERSION}")
        Na, Nb, Nphi = one("Na", int), one("Nb", int), one("Nphi", int)
        ncos, nsin = one("ncos", int), one("nsin", int)
        cos_m, sin_m = arr("cos_m", int), arr("sin_m", int)
        if cos_m.size != ncos or sin_m.size != nsin:
            raise ValueError(f"{path}: mode-count mismatch")
        C = arr("C").reshape(Na + 1, Nb, ncos)
        S = (arr("S").reshape(Na + 1, Nb, nsin) if nsin
             else np.zeros((Na + 1, Nb, 0)))
        A, B = arr("A"), arr("B")
        if A.size != Na + 1 or B.size != Nb:
            raise ValueError(f"{path}: node-count mismatch")
        return Export(
            b=one("b"), m_A=one("m_A"), m_B=one("m_B"),
            P_A=tuple(arr("P_A")), P_B=tuple(arr("P_B")),
            S_A=tuple(arr("S_A")), S_B=tuple(arr("S_B")),
            Na=Na, Nb=Nb, Nphi=Nphi, A=A, B=B,
            cos_m=cos_m, sin_m=sin_m, C=C, S=S, provenance=tuple(prov))


# --------------------------------------------------------------------------
# Reference evaluator — numpy only, no solver import
# --------------------------------------------------------------------------
def _bary_weights(x: np.ndarray) -> np.ndarray:
    """Barycentric weights of the node set ``x`` (same rule as the solver)."""
    x = np.asarray(x, dtype=float)
    w = np.ones(x.size)
    for j in range(x.size):
        d = x[j] - x
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    return w


def _bary_matrix(xq, x, w) -> np.ndarray:
    """Interpolation matrix ``(M, n)`` with exact-node rows handled."""
    xq = np.atleast_1d(np.asarray(xq, dtype=float))
    d = xq[:, None] - np.asarray(x, dtype=float)[None, :]
    hit = np.isclose(d, 0.0, atol=1e-13)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(hit, 0.0, w[None, :] / d)
    row_hit = hit.any(axis=1)
    t[row_hit] = 0.0
    t[hit] = 1.0
    return t / t.sum(axis=1, keepdims=True)


def inverse_abt(rho, z, b):
    """``(rho, z) -> (A, B)``: the closed-form inverse ABT map.

    Independent transcription of ``solver.operators_abt.inverse_map``:
    ``r1, r2`` are the distances to ``(0,0,+b)`` and ``(0,0,-b)``,
    ``xi = (r1+r2)/2b``, ``B = (r2-r1)/2b``, ``A = sqrt((xi-1)/(xi+1))``.
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    r1 = np.hypot(rho, z - b)
    r2 = np.hypot(rho, z + b)
    xi = (r1 + r2) / (2.0 * b)
    B = (r2 - r1) / (2.0 * b)
    A = np.sqrt(np.clip((xi - 1.0) / (xi + 1.0), 0.0, 1.0))
    return A, np.clip(B, -1.0, 1.0)


def eval_u(exp: Export, x, y, z) -> np.ndarray:
    """The correction ``u`` at Cartesian points, from the exported record only."""
    x = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
    y = np.atleast_1d(np.asarray(y, dtype=float)).reshape(-1)
    z = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    A_q, B_q = inverse_abt(rho, z, exp.b)

    wA, wB = _bary_weights(exp.A), _bary_weights(exp.B)
    MA = _bary_matrix(A_q, exp.A, wA)              # (M, Na+1)
    MB = _bary_matrix(B_q, exp.B, wB)              # (M, Nb)

    # Interpolate the coefficients: B first then A, matching the solver's order.
    #   tmp[m, i, t] = sum_j MB[m, j] C[i, j, t]
    #   co [m, t]    = sum_i MA[m, i] tmp[m, i, t]
    def interp(coef):
        if coef.shape[-1] == 0:
            return np.zeros((A_q.size, 0))
        tmp = np.einsum("mj,ijt->mit", MB, coef)
        return np.einsum("mi,mit->mt", MA, tmp)

    return phi_eval(interp(exp.C), interp(exp.S), exp.cos_m, exp.sin_m, phi)


def eval_psi_BL(exp: Export, x, y, z) -> np.ndarray:
    """``psi_BL = 1 + m_A/(2 r_A) + m_B/(2 r_B)`` at Cartesian points."""
    x = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
    y = np.atleast_1d(np.asarray(y, dtype=float)).reshape(-1)
    z = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
    rA = np.sqrt(x ** 2 + y ** 2 + (z - exp.b) ** 2)
    rB = np.sqrt(x ** 2 + y ** 2 + (z + exp.b) ** 2)
    return 1.0 + exp.m_A / (2.0 * rA) + exp.m_B / (2.0 * rB)


def eval_psi(exp: Export, x, y, z) -> np.ndarray:
    """The full conformal factor ``psi = psi_BL + u``."""
    return eval_psi_BL(exp, x, y, z) + eval_u(exp, x, y, z)


def eval_Ahat(exp: Export, x, y, z) -> np.ndarray:
    """Closed-form Bowen-York ``Ahat^{ij}`` at Cartesian points, shape (M,3,3).

    Independent transcription of ``solver.source_3d.A_full_tensor_vec``:
    momentum ``(3/2r^2)[P^i n^j + n^i P^j - (delta^{ij} - n^i n^j)(P.n)]`` plus
    spin ``(3/r^3)(v^i n^j + n^i v^j)`` with ``v = S x n``, summed over the two
    punctures.  Indices are raised/lowered with the flat conformal metric, so
    the component array is the same for ``Ahat^{ij}`` and ``Ahat_ij``.
    """
    X = np.stack([np.atleast_1d(np.asarray(q, dtype=float)).reshape(-1)
                  for q in (x, y, z)], axis=1)
    out = np.zeros((X.shape[0], 3, 3))
    eye = np.eye(3)
    for x0, Pv, Sv in ((np.array([0.0, 0.0, exp.b]), np.asarray(exp.P_A,
                                                                dtype=float),
                        np.asarray(exp.S_A, dtype=float)),
                       (np.array([0.0, 0.0, -exp.b]), np.asarray(exp.P_B,
                                                                 dtype=float),
                        np.asarray(exp.S_B, dtype=float))):
        d = X - x0
        r = np.linalg.norm(d, axis=1)
        n = d / r[:, None]
        if np.any(Pv != 0.0):
            Pn = n @ Pv
            PnT = (Pv[None, :, None] * n[:, None, :]
                   + n[:, :, None] * Pv[None, None, :])
            proj = eye[None] - n[:, :, None] * n[:, None, :]
            out += (3.0 / (2.0 * r ** 2))[:, None, None] * (
                PnT - proj * Pn[:, None, None])
        if np.any(Sv != 0.0):
            v = np.cross(np.broadcast_to(Sv, n.shape), n)
            vn = (v[:, :, None] * n[:, None, :]
                  + n[:, :, None] * v[:, None, :])
            out += (3.0 / r ** 3)[:, None, None] * vn
    return out


# --------------------------------------------------------------------------
# Building an Export from a solve
# --------------------------------------------------------------------------
def _git_sha(path: Optional[str] = None) -> str:
    try:
        root = path or os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(["git", "-C", root, "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:                                    # pragma: no cover
        return "unknown"


def from_solution(prob, sl, U, *, residual: Optional[float] = None,
                  raw_residual: Optional[float] = None,
                  note: str = "") -> Export:
    """Package a converged nodal ``U`` on ``prob`` for slice ``sl``.

    ``prob`` is a ``solver_3d.Problem3D``, ``sl`` a ``solver_3d.Slice3D``, and
    ``U`` the nodal correction with ``prob.shape == (Na+1, Nb, Nphi)``.
    ``residual`` should be the **equilibrated** residual (``NKInfo.residual_norm``
    — the certified number the solve controls), ``raw_residual`` the raw nodal
    inf-norm; both are recorded so the provenance is unambiguous.
    """
    U = np.asarray(U, dtype=float).reshape(prob.shape)
    C, S = phi_modes(U)
    cos_m, sin_m = phi_mode_layout(prob.Nphi)
    prov = [f"git {_git_sha()}",
            f"grid Na={prob.Na} Nb={prob.Nb} Nphi={prob.Nphi}"]
    if residual is not None:
        prov.append(f"newton_residual_equilibrated_inf {residual:.6e}")
    if raw_residual is not None:
        prov.append(f"newton_residual_raw_inf {raw_residual:.6e}")
    if note:
        prov.append(note)
    return Export(
        b=float(sl.b), m_A=float(sl.m_A), m_B=float(sl.m_B),
        P_A=tuple(float(v) for v in sl.P_A_vec),
        P_B=tuple(float(v) for v in sl.P_B_vec),
        S_A=tuple(float(v) for v in sl.S_A_vec),
        S_B=tuple(float(v) for v in sl.S_B_vec),
        Na=prob.Na, Nb=prob.Nb, Nphi=prob.Nphi,
        A=np.asarray(prob.A, dtype=float), B=np.asarray(prob.B, dtype=float),
        cos_m=cos_m, sin_m=sin_m, C=C, S=S, provenance=tuple(prov))


def solve_and_export(b: float, m_A: float, m_B: float, *,
                     P_A=(0.0, 0.0, 0.0), P_B=(0.0, 0.0, 0.0),
                     S_A=(0.0, 0.0, 0.0), S_B=(0.0, 0.0, 0.0),
                     Na: int = 36, Nb: int = 24, Nphi: int = 8,
                     path: Optional[str] = None, note: str = "") -> Export:
    """Run the certified Newton-Krylov solve and export the result.

    Imported lazily so that reading/evaluating an exported file needs numpy
    only (the solver stack pulls in jax).
    """
    from ..solver import solver_3d as s3
    from ..solver import solver_3d_nk as nk

    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    sl = s3.Slice3D(b=b, m_A=m_A, m_B=m_B, P_A_vec=tuple(P_A),
                    P_B_vec=tuple(P_B), S_A_vec=tuple(S_A),
                    S_B_vec=tuple(S_B))
    U, info = nk.newton_solve_nk(prob, sl)
    if not info.converged:
        raise RuntimeError(
            f"solve did not converge: equilibrated residual "
            f"{info.residual_norm:.3e} after {info.iters} iterations")
    exp = from_solution(prob, sl, U, residual=info.residual_norm,
                        raw_residual=info.raw_residual_norm, note=note)
    if path:
        exp.write(path)
    return exp


def dump_reference_table(exp: Export, path: str, *, n: int = 64,
                         half_width: float = 9.0, seed: int = 0) -> str:
    """Write a table of reference ``psi`` / ``Ahat`` values for a C++ port test.

    Random points in the cube of the given half width, the punctures avoided to
    ``0.25 b``.  Columns: ``x y z psi Ahat_xx Ahat_xy Ahat_xz Ahat_yy Ahat_yz
    Ahat_zz``.
    """
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-half_width, half_width, size=3)
        rA = np.linalg.norm(p - np.array([0.0, 0.0, exp.b]))
        rB = np.linalg.norm(p - np.array([0.0, 0.0, -exp.b]))
        if min(rA, rB) > 0.25 * exp.b:
            pts.append(p)
    P = np.array(pts)
    psi = eval_psi(exp, P[:, 0], P[:, 1], P[:, 2])
    Ah = eval_Ahat(exp, P[:, 0], P[:, 1], P[:, 2])
    with open(path, "w") as f:
        f.write("# x y z psi Axx Axy Axz Ayy Ayz Azz\n")
        for k in range(P.shape[0]):
            vals = [P[k, 0], P[k, 1], P[k, 2], psi[k],
                    Ah[k, 0, 0], Ah[k, 0, 1], Ah[k, 0, 2],
                    Ah[k, 1, 1], Ah[k, 1, 2], Ah[k, 2, 2]]
            f.write(" ".join(f"{v:.17e}" for v in vals) + "\n")
    return path
