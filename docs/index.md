# Conformally flat puncture ID

A differentiable parametric model for binary-black-hole initial data.

```{raw} html
<div class="lm-spin"></div>
```

Every simulation of a binary black hole starts from initial data: the geometry of space at the first moment, which must satisfy Einstein's constraint equations.
These are usually solved from scratch for each new binary.
This model turns that solve into a function of the binary's separation, mass ratio and spins.
It interpolates between stored solutions to get a good first guess, then takes a few Newton steps until the discrete constraint residual drops below $10^{-11}$, and it refuses to return data that miss this bound.
Because the interpolant is written in JAX, it also gives the derivatives of the initial data with respect to the parameters.
These let you aim a binary at a chosen ADM mass and angular momentum in a few solves, reduce its eccentricity, or see how the data respond to each parameter.
This first version covers quasi-circular, conformally flat Bowen–York punctures, with four parameters (aligned spins) or eight (general spins), checked against TwoPunctures.

You can use the model's interpolated field on its own, solve a single binary without the model, or combine the two for certified data; [Choose a workflow](workflows.md) compares the three.
The pages under *Understand the method* explain how the interpolation works, and the paper [arXiv:2609.10173](https://arxiv.org/abs/2609.10173) gives the measurements.

```{toctree}
:maxdepth: 1
:caption: Start here

workflows
install
```

```{toctree}
:maxdepth: 1
:caption: Get initial data

evaluate
solve
refine
field
```

```{toctree}
:maxdepth: 1
:caption: Understand the method

method
smolyak
sensitivities
compression
accuracy
solver
```

```{toctree}
:maxdepth: 1
:caption: Further background

bernstein
```

```{toctree}
:maxdepth: 1
:caption: Reference

MODELS
api
```

```{toctree}
:maxdepth: 1
:caption: Development

DATA
STRUCTURE
overview
```
