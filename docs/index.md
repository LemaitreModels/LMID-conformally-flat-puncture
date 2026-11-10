# Conformally flat puncture ID

A differentiable parametric model for binary-black-hole initial data.

```{raw} html
<div class="lm-spin"></div>
```

Every simulation of a binary black hole starts from initial data: the geometry of space at the first moment, which must satisfy Einstein's constraint equations.
These are usually solved from scratch for each new binary.
This model turns that solve into a function of the binary's separation, mass ratio and spins.
It interpolates between stored solutions to get a good first guess, then takes a few Newton steps until the discrete constraint residual drops below $10^{-11}$, so every answer it returns is certified.
Because it is written in JAX, it also returns the derivatives of the initial data with respect to the parameters.
These let you aim a binary at a chosen ADM mass and angular momentum in a few solves, reduce its eccentricity, or see how the data respond to each parameter.
This first version covers quasi-circular, conformally flat Bowen–York punctures, with four parameters (aligned spins) or eight (general spins), checked against TwoPunctures.

These pages hold the implementation: the grids, the solver settings, the
certification gate, the shipped models, the export format and how every
figure in the paper is regenerated. The paper explains the method; it links
here for the details.

```{toctree}
:maxdepth: 1
:caption: Getting started

overview
```

```{toctree}
:maxdepth: 1
:caption: Background

solver
bernstein
sensitivities
smolyak
```

```{toctree}
:maxdepth: 1
:caption: Implementation

MODELS
DATA
STRUCTURE
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```
