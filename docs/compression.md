# Compression

The uncompressed 8-D model stores four fields of 14 400 numbers at each of 15 713
nodes, about 7 GB. Those fields are very far from independent: they are smooth
solutions of one equation for nearby binaries, and a few hundred spatial patterns
describe all of them to high accuracy. The shipped models therefore store the fields
in a reduced basis, by proper orthogonal decomposition (POD) [1].

## One basis for values and derivatives

Stack the stored fields as columns of one matrix: the value at every node with the
mean field $\bar U$ subtracted, and the tangents and cross derivatives as they are
(the derivative of a constant mean is zero). Its singular value decomposition gives
orthonormal spatial modes $\Phi = (\phi_1, \ldots, \phi_r)$, ordered by how much of
the stored fields each one carries. Values and derivatives share one basis, so a
derivative is decoded with the same modes as a value.

## Interpolate coefficients, then decode

Every stored field is replaced by its $r$ coefficients in the basis:

$$
c_i \ = \ \Phi^{\mathsf T} (U_i - \bar U), \qquad
\partial c_i \ = \ \Phi^{\mathsf T} \, \partial U_i .
$$

Parameter interpolation is linear in the stored data, so interpolating the
coefficients and then decoding,

$$
\tilde U(\theta) \ = \ \bar U + \Phi \, \tilde c(\theta) ,
$$

gives exactly the projection onto the basis of the uncompressed interpolant. The
weights of [From parameters to a field](method.md) act on vectors of length $r$
instead of 14 400, which is where the saving in memory and time comes from.

## Choosing the rank

Truncating the basis at $r$ modes discards the fields' components outside it. The
singular values bound that error for the stored fields themselves, but what matters
is the error at parameter points *between* the nodes, so the rank is chosen by
measuring the held-out error of the whole model as $r$ grows, and taking the rank
where it stops improving: there, compression is no longer the limiting error and
costs no extra Newton step in the refinement. The shipped ranks are $r = 250$ in 4-D
and $r = 500$ in 8-D, which store 36 MiB and 295 MiB of arrays. The paper shows the
measurement [2]; [Available models](MODELS.md) lists the storage of every variant.

## Four kinds of error

A model prediction differs from the true continuum solution for four separate
reasons, and it helps to keep them apart:

1. **Spatial discretization.** The solution on a $45 \times 40 \times 8$ grid differs
   from the continuum one. Only resolution studies and comparison with an
   independent code measure it.
2. **Parameter interpolation.** Between the nodes the interpolant differs from the
   solution on the same grid.
3. **Compression.** The truncated basis drops what lies outside it.
4. **Iteration.** A solve stopped at a finite residual differs from the exact
   discrete solution.

Refinement removes 2 and 3 and controls 4, through the certification gate. It
cannot touch 1. [Accuracy and certification](accuracy.md) says how each is measured.

## References

1. G. Berkooz, P. Holmes and J. L. Lumley, *The proper orthogonal decomposition in
   the analysis of turbulent flows*, Annu. Rev. Fluid Mech. **25**, 539–575 (1993).
2. F. De Ceuster, *A differentiable parametric model of binary-black-hole initial
   data: I. Conformally flat Bowen–York punctures*,
   [arXiv:2609.10173](https://arxiv.org/abs/2609.10173), sections *Reduced-basis
   re-encoding* and *Reduced-basis compression of the model*.
