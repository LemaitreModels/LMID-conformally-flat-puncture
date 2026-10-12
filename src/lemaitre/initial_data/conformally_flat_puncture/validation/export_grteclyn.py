"""Export a converged solve to a plain file an evolution code can read.

Written for the GRTeclyn constraint check:
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
interpolation in phi.  Rather than ship the nodal samples and make the consumer
redo an FFT, we ship the equivalent real cosine/sine coefficients

    u(A, B, phi) = a_0(A,B) + sum_k [ a_k(A,B) cos(k phi) + b_k(A,B) sin(k phi) ]

so the consumer only sums cos/sin.  This is exact, not an approximation: the
transform in :func:`phi_modes` is the algebraic rewrite of
``solver_3d._fourier_interp`` for real samples, and the round-trip is gated to
machine precision in ``tests/test_export_grteclyn.py``.  Because the ``(A, B)``
interpolation is linear, interpolating the coefficients and then summing the
series gives the same number as interpolating per mode and then summing.

THE AXIS FACTOR (``format 2``).  What is shipped is the **smooth factor**, not
the physical field: the wavenumber-``k`` coefficients are divided by
``w_k(B) = (1-B^2)^{k/2}`` at the nodes, and the consumer multiplies them back
after interpolating.  This mirrors ``solver_3d.evaluate_field`` and is what makes
both evaluators the truncated spectral expansion rather than an approximation to
it: the solver's approximation space carries that factor analytically (see
``operators_3d``), so ``u_k`` itself is not a polynomial in B — for odd ``k`` not
even smooth at ``B=+-1`` — while ``u_k / w_k`` is, and only the latter is
recovered exactly by polynomial interpolation.  Format 1 shipped the physical
coefficients and interpolated them directly, which was exact only for ``k=0``;
a format-1 consumer must not read a format-2 file, and :meth:`Export.read`
refuses the mismatch.

FILE FORMAT (``format 2``).  A plain ASCII token stream: ``#`` comments, then
``<key> <values...>`` records.  Chosen over JSON/HDF5 so that the consumer needs
no parser library — ``ifstream >> token`` is enough — and so the format is
readable in a diff.  Keys:

    format b m_A m_B  P_A P_B S_A S_B  Na Nb Nphi  A B  cos_m sin_m  C S

``Na`` is the Chebyshev-Lobatto order, so ``A`` has ``Na+1`` nodes
(``A[0] = 1`` is spatial infinity, ``A[Na] = 0`` the inner axis edge); ``B`` has
``Nb`` Gauss-Legendre interior nodes.  ``C`` and ``S`` are flattened
``C[i][j][t]``, i.e. ``t`` fastest then ``j`` then ``i``.

FORMAT 4 — the ``"v2"`` grid (``solver.operators_abt.SCHEMES``).  The same token
stream with two differences, both forced by the grid:

* the angular nodes are shipped in the map's interpolation coordinate ``s``
  (key ``s`` instead of ``B``; ``B = 2s/(1+s^2)``, the ABT angle), because the
  field is a polynomial in ``s``, not in ``B``;
* the ``A`` nodes are Chebyshev-Gauss (no node at ``A = 0`` or ``A = 1``) and the
  field carries the radial factor ``A - 1`` of TwoPunctures' ``u = (A - 1) V``.

So what is shipped is ``V_k = u_k / (A - 1)`` — and NOT divided by ``w_k(B)``: on the
ABT angle ``w_k = sin^k(eta)`` is rational in ``s``, so ``u_k`` is itself analytic in
``s``, and dividing by ``w_k`` at nodes where it is ~``(1 - s)^k`` small would only
amplify roundoff (``solver_3d.evaluate_field`` makes the same choice on a mapped
grid, and this rule is that evaluator, exactly).  The consumer interpolates ``C, S``
in ``(A, s)`` and multiplies by ``A - 1`` at the query point.  The query coordinate
``s`` must be computed without cancellation next to the axis, where ``s`` is
ill-conditioned in ``B`` (:func:`inverse_abt_s`).  Two records name the
grid explicitly, ``angular_map abt`` and ``radial_map gauss``, and a reader checks
them.  Format numbers: 2 = ``"paper-v1"``, 3 = the curved sibling's channel, 4 =
``"v2"``; every reader checks the number by EQUALITY against the rules it
implements, because reading one format with another's rule is silently wrong.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

#: The format each discretisation (``solver.operators_abt.SCHEMES``) is exported in.
#: ``"paper-v1"`` (``"cos"`` + ``"lobatto"``: a polynomial in ``B`` on
#: Chebyshev–Gauss–Lobatto ``A``) is format 2; ``"v2"`` (``"abt"`` + ``"gauss"``: a
#: polynomial in ``s`` on Chebyshev–Gauss ``A``, times ``A - 1``) is format 4.  A mixed
#: grid has no format and is refused.  (Format 3 is the curved sibling's channel.)
FORMAT_OF_SCHEME = {"paper-v1": 2, "v2": 4}

#: The formats this module reads and writes.
FORMATS = tuple(sorted(FORMAT_OF_SCHEME.values()))

#: The current format: what an export of the solver's default grid (``"v2"``) is.
FORMAT_VERSION = FORMAT_OF_SCHEME["v2"]


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


def axis_factor(B, k) -> np.ndarray:
    """``w_k(B) = (1-B^2)^{k/2}`` — the associated-Legendre factor of the basis.

    Numpy-only transcription of ``solver_3d._w_factor`` (itself the factor of
    ``operators_3d.bc_factor``).  Finite at ``B=+-1``, where it vanishes for
    every ``k>0``: a regular wavenumber-``k`` mode has no amplitude on the axis.
    """
    B = np.asarray(B, dtype=float)
    if int(k) == 0:
        return np.ones_like(B)
    return (1.0 - B ** 2) ** (abs(int(k)) / 2.0)


def _factor_columns(B, modes) -> np.ndarray:
    """``(len(B), len(modes))`` stack of :func:`axis_factor`, one column per mode."""
    B = np.asarray(B, dtype=float)
    if len(modes) == 0:
        return np.zeros(B.shape + (0,))
    return np.stack([axis_factor(B, int(k)) for k in modes], axis=-1)


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
    """One exported slice: geometry, the ABT grid, and the phi-modal ``u``.

    ``B`` holds the angular nodes in the format's interpolation coordinate: ``B``
    itself in format 2, ``s`` in format 4 (written under the key ``s``).  ``fmt``
    defaults to 2 because a record built without it is the original grid.
    """
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
    C: np.ndarray                     # (Na+1, Nb, ncos), smooth factor
    S: np.ndarray                     # (Na+1, Nb, nsin), smooth factor
    provenance: Sequence[str] = field(default_factory=tuple)
    fmt: int = 2                      # 2 ("paper-v1") or 4 ("v2"); see FORMAT_OF_SCHEME

    # -- writing ------------------------------------------------------------
    def write(self, path: str) -> str:
        """Write the ASCII record; returns ``path``."""
        def fmt(a):
            return " ".join(f"{v:.17e}" for v in np.asarray(a).reshape(-1))

        if self.fmt not in FORMATS:
            raise ValueError(f"format {self.fmt}: this writer handles {FORMATS}")
        lines = [f"# LM-initial-data GRTeclyn export, format {self.fmt}"]
        lines += [f"# {ln}" for ln in self.provenance]
        if self.fmt == 2:
            lines += [
                "# psi = psi_BL + u ;  psi_BL = 1 + m_A/(2 r_A) + m_B/(2 r_B),",
                "# punctures A at (0,0,+b) and B at (0,0,-b).",
                "# u(A,B,phi) = sum_t w(cos_m[t],B) C[i][j][t] cos(cos_m[t] phi)",
                "#            + sum_t w(sin_m[t],B) S[i][j][t] sin(sin_m[t] phi)",
                "# with w(k,B) = (1-B^2)^(k/2) applied AFTER interpolating C,S",
                "# in (A,B) by tensor-product barycentric interpolation, and",
                "# (A,B) <- (rho,z) by the closed-form inverse ABT map.",
            ]
        else:
            lines += [
                "# psi = psi_BL + u ;  psi_BL = 1 + m_A/(2 r_A) + m_B/(2 r_B),",
                "# punctures A at (0,0,+b) and B at (0,0,-b).",
                "# u(A,s,phi) = (A-1) sum_t C[i][j][t] cos(cos_m[t] phi)",
                "#            + (A-1) sum_t S[i][j][t] sin(sin_m[t] phi)",
                "# with (A-1) applied AFTER interpolating C,S in (A,s) by",
                "# tensor-product barycentric interpolation (no axis factor:",
                "# u_k is analytic in s on this angle);",
                "# A on Chebyshev-Gauss nodes (no edge node), s the ABT angle",
                "# (B = 2s/(1+s^2)); (A,s) <- (rho,z) by the closed-form inverse map,",
                "# with 1-B and 1+B formed without cancellation (see inverse_abt_s).",
            ]
        lines += [
            f"format {self.fmt}",
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
            *(["angular_map abt", "radial_map gauss"] if self.fmt == 4 else []),
            "cos_m " + " ".join(str(int(v)) for v in self.cos_m),
            "sin_m " + (" ".join(str(int(v)) for v in self.sin_m)
                        if self.sin_m.size else ""),
            "A " + fmt(self.A),
            ("s " if self.fmt == 4 else "B ") + fmt(self.B),
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
        if fmt_v not in FORMATS:
            raise ValueError(
                f"{path}: format {fmt_v}, this reader handles {FORMATS}")
        if fmt_v == 4 and (rec.get("angular_map") != ["abt"]
                           or rec.get("radial_map") != ["gauss"]):
            raise ValueError(f"{path}: format 4 must declare 'angular_map abt' and "
                             "'radial_map gauss'")
        Na, Nb, Nphi = one("Na", int), one("Nb", int), one("Nphi", int)
        ncos, nsin = one("ncos", int), one("nsin", int)
        cos_m, sin_m = arr("cos_m", int), arr("sin_m", int)
        if cos_m.size != ncos or sin_m.size != nsin:
            raise ValueError(f"{path}: mode-count mismatch")
        C = arr("C").reshape(Na + 1, Nb, ncos)
        S = (arr("S").reshape(Na + 1, Nb, nsin) if nsin
             else np.zeros((Na + 1, Nb, 0)))
        A, B = arr("A"), arr("s" if fmt_v == 4 else "B")
        if A.size != Na + 1 or B.size != Nb:
            raise ValueError(f"{path}: node-count mismatch")
        return Export(
            b=one("b"), m_A=one("m_A"), m_B=one("m_B"),
            P_A=tuple(arr("P_A")), P_B=tuple(arr("P_B")),
            S_A=tuple(arr("S_A")), S_B=tuple(arr("S_B")),
            Na=Na, Nb=Nb, Nphi=Nphi, A=A, B=B,
            cos_m=cos_m, sin_m=sin_m, C=C, S=S, provenance=tuple(prov), fmt=fmt_v)


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


def inverse_abt_s(rho, z, b):
    """``(rho, z) -> (A, s, 1 - B^2)`` for format 4: the ABT angle ``s`` (with
    ``B = 2s/(1+s^2)``) and ``1 - B^2``, both without cancellation next to the axis
    (the evaluator needs only ``A`` and ``s``; ``1 - B^2`` is returned for checks).

    Independent transcription of ``solver.operators_abt.inverse_map_s``.  ``A`` as in
    :func:`inverse_abt`.  ``B = 2z/(r1+r2)``, and ``1 - B``, ``1 + B`` are sums of
    ``r - c`` / ``r + c`` terms (``c = z -/+ b``) rationalised as ``rho^2/(r +/- c)``
    where they would cancel, so a point a hair off the outer axis still gets ``s`` to
    ~eps (``s`` is ill-conditioned in ``B`` there: ``1 - s ~ sqrt(2(1 - B))``).  Then
    ``s = B / (1 + sqrt((1-B)(1+B)))``, finite at ``B = 0``.
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    A, _ = inverse_abt(rho, z, b)
    c1, c2 = z - b, z + b
    r1, r2 = np.hypot(rho, c1), np.hypot(rho, c2)
    rho2 = rho * rho

    def r_minus_c(r, c):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(c > 0.0, rho2 / (r + c), r - c)

    def r_plus_c(r, c):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(c < 0.0, rho2 / (r - c), r + c)

    S = r1 + r2
    B = np.clip(2.0 * z / S, -1.0, 1.0)
    om = (r_minus_c(r1, c1) + r_minus_c(r2, c2)) / S
    op = (r_plus_c(r1, c1) + r_plus_c(r2, c2)) / S
    omb2 = np.clip(om * op, 0.0, None)
    return A, B / (1.0 + np.sqrt(omb2)), omb2


def eval_u(exp: Export, x, y, z) -> np.ndarray:
    """The correction ``u`` at Cartesian points, from the exported record only."""
    x = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
    y = np.atleast_1d(np.asarray(y, dtype=float)).reshape(-1)
    z = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    if exp.fmt == 4:
        A_q, B_q, _ = inverse_abt_s(rho, z, exp.b)         # B_q holds s here
    else:
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

    # ...then restore the factor analytically at the query point: w_k (format 2);
    # the radial A - 1 (format 4, whose u_k is analytic in s — no w_k).
    if exp.fmt == 4:
        fq = (A_q - 1.0)[:, None]
        C_q = interp(exp.C) * fq
        S_q = interp(exp.S) * fq
    else:
        C_q = interp(exp.C) * _factor_columns(B_q, exp.cos_m)
        S_q = interp(exp.S) * _factor_columns(B_q, exp.sin_m)
    return phi_eval(C_q, S_q, exp.cos_m, exp.sin_m, phi)


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

    The grid decides the format (:data:`FORMAT_OF_SCHEME`): ``"paper-v1"`` is
    format 2 (a polynomial in ``B`` on Lobatto ``A``), ``"v2"`` is format 4 (a
    polynomial in ``s`` on Chebyshev–Gauss ``A``, with the radial factor ``A - 1``).
    A mixed grid (cos + gauss, abt + lobatto) has no format and is refused rather
    than written wrong.
    """
    bmap = getattr(prob, "bmap", "cos")
    radial = getattr(prob, "radial", "lobatto")
    fmt_v = {("cos", "lobatto"): 2, ("abt", "gauss"): 4}.get((bmap, radial))
    if fmt_v is None:
        raise NotImplementedError(
            f"no export format holds the mixed grid bmap={bmap!r}, radial={radial!r}: "
            f"format 2 is ('cos', 'lobatto') = 'paper-v1' and format 4 is "
            f"('abt', 'gauss') = 'v2'. Exporting another needs a new format on both "
            "the writer and the GRTeclyn reader.")
    U = np.asarray(U, dtype=float).reshape(prob.shape)
    C, S = phi_modes(U)
    cos_m, sin_m = phi_mode_layout(prob.Nphi)
    # Ship the SMOOTH factor.  Format 2: divide out w_k(B) at the (interior) B nodes,
    # so the consumer interpolates a polynomial in B and restores w_k.  Format 4:
    # divide out only the radial A - 1 (< 0 at every Chebyshev-Gauss node); u_k is
    # analytic in s on the ABT angle, so w_k stays in (solver_3d.evaluate_field's rule).
    if fmt_v == 2:
        C = C / _factor_columns(prob.B, cos_m)[None, :, :]
        if sin_m.size:
            S = S / _factor_columns(prob.B, sin_m)[None, :, :]
        nodes = np.asarray(prob.B, dtype=float)
    else:
        fA = (np.asarray(prob.A, dtype=float) - 1.0)[:, None, None]
        C = C / fA
        if sin_m.size:
            S = S / fA
        nodes = np.asarray(prob.Bs, dtype=float)       # s, as laid
    prov = [f"git {_git_sha()}",
            f"grid Na={prob.Na} Nb={prob.Nb} Nphi={prob.Nphi} scheme={prob.scheme}"]
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
        A=np.asarray(prob.A, dtype=float), B=nodes,
        cos_m=cos_m, sin_m=sin_m, C=C, S=S, provenance=tuple(prov), fmt=fmt_v)


def solve_and_export(b: float, m_A: float, m_B: float, *,
                     P_A=(0.0, 0.0, 0.0), P_B=(0.0, 0.0, 0.0),
                     S_A=(0.0, 0.0, 0.0), S_B=(0.0, 0.0, 0.0),
                     Na: int = 36, Nb: int = 24, Nphi: int = 8,
                     scheme: str = "v2",
                     path: Optional[str] = None, note: str = "") -> Export:
    """Run the certified Newton-Krylov solve and export the result.

    ``scheme`` names the grid (a key of :data:`FORMAT_OF_SCHEME`), and with it the
    format.  Imported lazily so that reading/evaluating an exported file needs numpy
    only (the solver stack pulls in jax).
    """
    from ..solver import solver_3d as s3
    from ..solver import solver_3d_nk as nk

    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi, scheme=scheme)
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
