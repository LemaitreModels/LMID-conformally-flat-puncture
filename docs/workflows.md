# Choose a workflow

There are three ways to get initial data out of this package. They differ in what
you need to download, how long a query takes, and what is guaranteed about the
answer.

| | Evaluate a model | Solve one binary | Refine a model prediction |
|---|---|---|---|
| **What you call** | `model.evaluate(theta)` | `solver_3d_nk.newton_solve_nk(prob, sl)` | `model.evaluate_polished(theta)` |
| **Needs a model file** | yes (downloaded on first use) | no | yes |
| **Constraint solve** | none | full Newton–Krylov solve from zero | two Newton steps from the model's guess |
| **What is guaranteed** | nothing per query; the error is the model's measured interpolation error | the discrete constraint residual is below the gate when `info.converged` | the discrete constraint residual is below the gate, or an error is raised |
| **Derivatives** | `jax.jacfwd(model.evaluate_jax)`, the interpolant's | the certified tangent, one linear solve per parameter | either |
| **Parameters** | inside the model's box | any Bowen–York binary | inside the model's box |
| **Typical use** | many queries where an approximation is enough: scans, plots, gradients | one configuration, or one outside the box | certified initial data for an evolution, many times |

The **gate** is a bound of $10^{-11}$ on the discrete constraint residual (the
[constraint solver](solver.md) page defines it). It certifies the discrete solve on
the model's spatial grid. How close that discrete solution is to the continuum one is
a separate question, which [Accuracy and certification](accuracy.md) answers.

All three routes return the same thing: the regular part $u$ of the conformal
factor, $\psi = \psi_{\rm BL} + u$, as values at the nodes of one spectral grid.
[Use the field](field.md) turns that into $\psi$, the metric and the extrinsic
curvature at any point, and exports it to an evolution code.

The parameters, for every route:

- $b$ is **half** the coordinate separation; the punctures sit on the $z$ axis at
  $z = \pm b$, puncture A at $+b$.
- $q = m_A / m_B \ge 1$ is the ratio of the bare masses, and $m_A + m_B = 1$.
- $\chi_X = S_X / m_X^2$ is each spin divided by its own bare mass squared.
- The orbit lies in the $x$–$z$ plane with orbital angular momentum along $y$, so an
  *aligned* spin is the $y$ component, `chi_Ay` and `chi_By`.
- The momenta are not parameters. The models and `theta_to_slice3d(...,
  fixed={"qc": 1.0})` set them to the quasi-circular values of
  `quasicircular.QC_PRESCRIPTION`.

The pages that follow are short and runnable:

- [Installation](install.md)
- [Evaluate a model](evaluate.md)
- [Solve one binary](solve.md)
- [Refine a model prediction](refine.md)
- [Use the field](field.md)
