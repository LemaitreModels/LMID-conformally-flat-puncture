# Accuracy and certification

"How accurate is it?" has four answers, one per source of error
([Compression](compression.md) lists them). This page says what each one is, how it
is measured, and which of them the certification gate controls.

## What the gate certifies

A certified field is one whose **discrete** constraint residual, measured after the
last Newton step on the field actually returned, is at most $10^{-11}$:

$$
\| R(U) \|^{(u)}_\infty \ \le \ 10^{-11} .
$$

$R$ is the residual of the collocation equations, each row divided by the size of
its own operator row so that rows near the punctures and near infinity count alike
(the *equilibrated* residual, defined on [the constraint solver](solver.md) page).
The bound is `certification.CERT_TOL_U`, and it sits well above the residual that
converged solves reach (quadratic convergence usually ends far below it) and far
below the error the grid itself makes.

Certification is a statement about the **solve**: the returned field solves the
discrete equations on its grid, independently of how good the starting guess was. It
is not a statement about the distance to the continuum solution.

Where the gate is applied:

- `evaluate_polished` checks it by default and raises `CertificationError` on a miss
  (`strict=False` returns the field and its residual instead).
- The raw solver `solver_3d_nk.newton_solve_nk` reports it in `info.converged` and
  returns either way; `certification.certified_return(U, info, theta, tol, strict=True)`
  turns that into an error.
- Parameter targeting, eccentricity reduction and the GRTeclyn export refuse an
  uncertified field.

## Spatial accuracy

The difference between the discrete and the continuum solution is the truncation
error of the $45 \times 40 \times 8$ grid. It is measured, not certified, in two
independent ways in the paper [1]: by refining the grid, and by comparison with the
TwoPunctures code [2] on quasi-circular configurations (appendix *Validation against
TwoPunctures*). The paper also evolves exported data in an independent evolution code
and measures the constraint violation on its grid (*Constraint violation on an
evolution grid*).

## Interpolation and compression accuracy

The error of a bare prediction, `model.evaluate(theta)`, is measured against
certified solves at random parameter points that are not interpolation nodes. The
paper reports it as the relative field error over a thousand such points, per axis,
jointly over the box, and as a function of the compression rank (sections
*Convergence and its analyticity limits*, *Offline cost and certified refinement*
and *Reduced-basis compression of the model*). Those are the numbers to use when you
decide whether a bare prediction is good enough.

The error is not uniform over the box. It grows towards small separations and large
spins, where the map from parameters to solution approaches a singularity
([the Bernstein ellipse](bernstein.md) explains why that slows convergence).

## Check it yourself

At any parameter point you can measure the prediction error directly, because the
refined field is certified:

```python
import numpy as np
from lemaitre.initial_data.conformally_flat_puncture import models

model = models.load("4d", solver=True)
theta = np.array([5.0, 1.5, 0.3, -0.2])
U_guess = model.evaluate(theta)
U_cert, info = model.evaluate_polished(theta)
print(np.linalg.norm(U_guess - U_cert) / np.linalg.norm(U_cert), info.residual_norm)
```

The first number is the model's actual relative error at that point, on the model's
grid. It includes interpolation and compression, not spatial truncation.

## References

1. F. De Ceuster, *A differentiable parametric model of binary-black-hole initial
   data: I. Conformally flat Bowen–York punctures*,
   [arXiv:2609.10173](https://arxiv.org/abs/2609.10173).
2. M. Ansorg, B. Brügmann and W. Tichy, *Single-domain spectral method for black
   hole puncture data*, Phys. Rev. D **70**, 064011 (2004),
   [doi:10.1103/PhysRevD.70.064011](https://doi.org/10.1103/PhysRevD.70.064011).
