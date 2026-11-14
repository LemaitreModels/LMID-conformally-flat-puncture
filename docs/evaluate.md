# Evaluate a model

The fastest route: interpolate stored solutions, with no constraint solve. Use it
when the model's interpolation error is small enough for what you do with the
answer, for example in a parameter scan, a plot or a gradient. For data that must
satisfy the constraints to the gate, [refine the prediction](refine.md).

The models are downloaded from Zenodo ([Available models](MODELS.md)); pass `path=` to
load a model file you built yourself.

```python
import numpy as np
import jax, jax.numpy as jnp
from lemaitre.initial_data.conformally_flat_puncture import models

model = models.load("4d")                 # downloads once, then reads the cache
print(model.meta["axis_names"])           # ['b', 'q', 'chi_Ay', 'chi_By']
print(model.coeff_model.axes)             # the box: b in [3, 10], q in [1, 3], chi in [-0.9, 0.9]

theta = np.array([5.0, 1.5, 0.3, -0.2])   # (b, q, chi_Ay, chi_By)
U = model.evaluate(theta)                 # u at the grid nodes, shape (45, 40, 8)
dU = jax.jacfwd(model.evaluate_jax)(jnp.asarray(theta))   # shape (45, 40, 8, 4)
```

`"8d"` is the general-spin model, θ = (b, q, χ_Ax, χ_Ay, χ_Az, χ_Bx, χ_By, χ_Bz),
on the same box for every spin component. [Choose a workflow](workflows.md) lists the
conventions behind the parameters.

What you get:

- **`U`** is the regular part $u$ of the conformal factor at the nodes of the
  model's spectral grid, not $\psi$ and not values at points of your choosing.
  [Use the field](field.md) evaluates it anywhere and rebuilds $\psi$, the metric
  and the extrinsic curvature.
- **`dU`** is the derivative of the *interpolant*, at fixed grid coordinates. Its
  last index follows the parameter order. It is exact for the polynomial the model
  stores, and it differs from the derivative of the true solution by the derivative
  of the interpolation error, which is typically larger than the error in the value
  ([Sensitivities](sensitivities.md) explains why). The grid is tied to $b$, so for
  the $b$ axis a derivative at a fixed *physical* point also needs the motion of the
  grid.
- **Nothing is certified.** The error of `U` is the model's interpolation and
  compression error, which [Accuracy and certification](accuracy.md) reports as
  measured on held-out solves.
- **Stay inside the box.** The model extrapolates outside it, badly.

`evaluate_jax` works at every parameter value, including the interpolation nodes,
where zero spin, a box edge and the box centre all lie.

## Cost

A query is a few matrix–vector products against the stored coefficients and one
decoding product with the reduced basis. For many queries, compile the JAX
evaluator once:

```python
from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd import jax_evaluator

f = jax_evaluator(model)      # compiled on the first call
U = f(jnp.asarray(theta))
```

The model's cost on disk and in memory is on [Available models](MODELS.md).
