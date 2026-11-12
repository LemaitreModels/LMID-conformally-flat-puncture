# Solve one binary

The direct route: no model, no download, any Bowen–York binary. The solver starts
from $u = 0$ and runs Newton–Krylov iterations until the discrete constraint residual
is below the gate. [The constraint solver](solver.md) explains each step.

```python
import numpy as np
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d, solver_3d_nk
from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd_3d import theta_to_slice3d
from lemaitre.initial_data.conformally_flat_puncture.parametric.certification import (
    CERT_TOL_U, certified_return)

theta = np.array([5.0, 1.5, 0.3, -0.2])                   # (b, q, chi_Ay, chi_By)
sl = theta_to_slice3d(theta, ["b", "q", "chi_Ay", "chi_By"], M_tot=1.0,
                      fixed={"qc": 1.0})                  # quasi-circular momenta
prob = solver_3d.make_problem(44, 40, 8, scheme="v2")     # the production grid

U, info = solver_3d_nk.newton_solve_nk(prob, sl)
U, info = certified_return(U, info, theta, CERT_TOL_U, strict=True)   # raise unless certified
print(U.shape, info.converged, info.gmres_iters)          # (45, 40, 8) True [3, 3, 3]
print(["%.1e" % r for r in info.history])                # ['2.4e-03', '9.6e-06', '9.2e-11', '2.1e-16']
```

`theta_to_slice3d` turns the parameters into a `Slice3D`: the two bare masses, the
puncture positions $z = \pm b$, the spins, and the quasi-circular momenta. Here it
gives $m_A = 0.6$, $m_B = 0.4$, $\vec S_A = (0, 0.108, 0)$ and
$\vec P_A \approx (0.0918, 0, -0.0007)$: tangential momentum along $x$ and a small
radial part along the axis.

**Any other binary** is a `Slice3D` you write yourself, with free momenta and spins:

```python
my_sl = solver_3d.Slice3D(b=4.0, m_A=0.5, m_B=0.5,
                          P_A_vec=(0.1, 0.0, 0.0), P_B_vec=(-0.1, 0.0, 0.0),
                          S_A_vec=(0.0, 0.0, 0.1), S_B_vec=(0.0, 0.0, 0.0))
my_U, my_info = solver_3d_nk.newton_solve_nk(prob, my_sl)
```

What to check:

- **`info.converged`** says whether the residual met the gate. The solver returns
  either way, so a script that uses `U` should call `certified_return(...,
  strict=True)` as above, which raises `CertificationError` when it did not.
- **`info.residual_norm`** is the equilibrated discrete residual. Quadratic
  convergence usually carries the last step well below the gate of $10^{-11}$.
- **The grid** sets the spatial accuracy. `make_problem(44, 40, 8)` is the grid the
  shipped models use. A finer grid costs more and resolves more; the gate certifies
  the solve on whichever grid you choose, not its truncation error.

## Cost

On an Apple M2 laptop the first solve on a grid takes about a second, most of it
building the grid's operators; later solves on the same grid take a fraction of a
second (measured 2026-11-12: 1.3 s, then 0.2 s, on a busy machine). The
preconditioner is built once per grid and reused at every separation.

## Derivatives of the solution

The certified tangent $\partial U/\partial\theta_k$ is one more linear solve with the
same Jacobian, including the change of the quasi-circular momenta with $\theta_k$. It
needs the quasi-circular `sl` and `U` of the first example:

```python
from lemaitre.initial_data.conformally_flat_puncture.applications.sensitivity_3d_qc import (
    certified_tangent_3d_qc)

dU = certified_tangent_3d_qc(prob, U, sl, "chi_Ay", 1.0)   # dU/dchi_Ay, shape (45, 40, 8)
```

It is much slower than a solve at present: the call above took 25 minutes on a heavily
loaded laptop (2026-11-12), against a fraction of a second for the solve. Its cost has
not been profiled yet.

[Sensitivities](sensitivities.md) derives it.
