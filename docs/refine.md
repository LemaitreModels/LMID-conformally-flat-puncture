# Refine a model prediction

The certified route through a model: the interpolated field is the starting guess,
and a few Newton steps of the full solver bring the constraint residual below the
gate. The answer is then as good as a direct solve, and it costs fewer Newton steps
because it starts close.

The models are downloaded from Zenodo ([Available models](MODELS.md)); pass `path=` to
load a model file you built yourself.

```python
import numpy as np
from lemaitre.initial_data.conformally_flat_puncture import models

model = models.load("4d", solver=True)       # attaches the solver on the model's own grid
theta = np.array([5.0, 1.5, 0.3, -0.2])      # (b, q, chi_Ay, chi_By)

U, info = model.evaluate_polished(theta)     # certified, or CertificationError
print(info.residual_norm, info.iters)
```

`solver=True` builds the spectral grid the model was built on and attaches the
Newton–Krylov solver with the model's quasi-circular momentum prescription. Building
the grid's operators takes a few seconds, once.

What `evaluate_polished` does:

1. **Guess.** `model.evaluate(theta)`, the interpolated field.
2. **Solve.** Newton–Krylov steps from that guess (`newton_steps=2` by default).
   If they miss the gate, the solver retries with a damped Newton iteration, which
   is what rescues the hardest corners of the box. Two steps is therefore the usual
   cost, not a hard limit.
3. **Check.** The residual of the returned field is compared with the gate. A field
   that misses it raises `CertificationError`, which names the parameter point and
   the residual reached.

The interpolation error changes how many steps the solve needs, not the answer:
the gate is checked on the field the solver returns.

To get the field and its residual whatever happens, for example in a sweep that
records failures instead of stopping, pass `strict=False` and read
`info.residual_norm` yourself.

## Compatibility

A model's stored fields are values at the nodes of one spectral grid, solved with
one momentum prescription. Refining on another grid, or with other momenta, would
polish towards different data, so `models.load(..., solver=True)` builds the grid
from the model's own metadata and the attach step refuses a model whose
`qc_prescription` differs from the code's. [Available models](MODELS.md) explains how
the download picks a model this code can read.

Next: [Use the field](field.md).
