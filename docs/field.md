# Use the field

Every route returns the same object: `U`, the regular part $u$ of the conformal
factor at the nodes of the spectral grid `prob`, for the binary `sl`. This page turns
it into the 3+1 initial data at any point, and exports it for an evolution code.

You need three things: `U`, the grid `prob`, and the binary `sl` (a
`solver_3d.Slice3D`). [Solve one binary](solve.md) builds all three. After
[evaluating](evaluate.md) or [refining](refine.md) a model, build the other two
from the model; neither step solves anything:

```python
from lemaitre.initial_data.conformally_flat_puncture import models
from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd_3d import theta_to_slice3d

prob = models.problem(model)
sl = theta_to_slice3d(theta, model.meta["axis_names"], M_tot=1.0, fixed={"qc": 1.0})
```

## The conformal factor

`evaluate_field` interpolates `U` spectrally to points given in cylindrical
coordinates $(\rho, z, \phi)$ around the $z$ axis that carries the punctures. It
returns $u$, so add the Brill–Lindquist part yourself:

```python
import numpy as np
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d

rho, z, phi = np.array([1.0]), np.array([2.0]), np.array([0.0])
u = solver_3d.evaluate_field(prob, U, rho, z, phi, sl.b)
r_A = np.hypot(rho, z - sl.b)                  # puncture A sits at z = +b
r_B = np.hypot(rho, z + sl.b)
psi = 1.0 + sl.m_A / (2 * r_A) + sl.m_B / (2 * r_B) + u
```

For the binary of [Solve one binary](solve.md) this gives $u = 0.0046691905$ and
$\psi = 1.1278217915$ at that point.

`evaluate_field` sums the same Fourier modes in $\phi$ and the same polynomials in
$A$ that the solver uses. In the angle it interpolates each Fourier mode as a
polynomial in the ABT coordinate $s$. That is exact for the axisymmetric mode; for
$m \ge 1$ the mode carries a factor $(1 - B^2)^{k/2}$ that is rational in $s$, so
there the evaluation is a spectrally accurate approximation rather than an exact
one.

## The metric and the extrinsic curvature

The data are conformally flat and maximally sliced, with Bowen–York extrinsic
curvature. In Cartesian coordinates $x = \rho\cos\phi$, $y = \rho\sin\phi$, $z$:

$$
\gamma_{ij} \ = \ \psi^4 \, \delta_{ij}, \qquad
K_{ij} \ = \ \psi^{-2} \, \hat A_{ij}, \qquad K \ = \ 0 ,
$$

where $\hat A_{ij}$ is the sum of the Bowen–York momentum and spin tensors of the two
punctures, known in closed form:

```python
from lemaitre.initial_data.conformally_flat_puncture.solver import source_3d
from lemaitre.initial_data.conformally_flat_puncture.validation.adm import physical_K_lower

X = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=-1)          # (n, 3)
A_hat = source_3d.A_full_tensor_vec(X, sl.b, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec)
K = physical_K_lower(psi, A_hat)                                             # (n, 3, 3)
gamma = psi[:, None, None] ** 4 * np.eye(3)
```

## Global quantities

```python
from lemaitre.initial_data.conformally_flat_puncture.solver.diagnostics_3d import (
    adm_mass_spectral_3d, adm_J_closed_form)

M_adm = adm_mass_spectral_3d(prob, U, sl)       # read off the spectral expansion at infinity
J_adm = adm_J_closed_form(sl.b, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec)
```

The ADM angular momentum of Bowen–York data does not depend on $u$, so it is exact
in closed form. Aiming a binary at a chosen ADM mass and angular momentum is the
job of `applications.qc_targeting`.

## Export to an evolution code

`validation.export_grteclyn` writes a slice in a plain-text format that the
GRTeclyn evolution code reads. The file carries the grid, the binary and the
spectral coefficients of $u$, so the reader rebuilds $\psi$ and $\hat A_{ij}$ itself
at its own points. The format follows the grid: format 4 for the production grid.

`from_solution` records the residual you hand it but does not check it, so export
only a certified field, with its `info` from a solve or from `evaluate_polished`
(a bare `model.evaluate` has no residual to record):

```python
from lemaitre.initial_data.conformally_flat_puncture.parametric.certification import (
    CERT_TOL_U, certified_return)
from lemaitre.initial_data.conformally_flat_puncture.validation import export_grteclyn

U, info = certified_return(U, info, theta, CERT_TOL_U, strict=True)   # refuse an uncertified field
exp = export_grteclyn.from_solution(prob, sl, U, residual=info.residual_norm,
                                    raw_residual=info.raw_residual_norm)
exp.write("binary.lmid")
```

`export_grteclyn.solve_and_export(b, m_A, m_B, P_A=..., S_A=..., ...)` solves and
writes in one call, and refuses to write an unconverged solve. [Reproduce the
paper](DATA.md) describes the evolution check the paper runs with these files.
