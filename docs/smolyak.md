# Sparse grids

The parametric models store one elliptic solve per parameter node. On a full
tensor grid with $q$ nodes per axis that is $q^d$ solves, which grows
exponentially with the number of parameters $d$. At the 33 nodes per axis of the
four-dimensional model's finest level, a tensor grid would need
$33^4 \approx 1.2$ million solves, and $33^8 \approx 1.4 \times 10^{12}$ in eight
dimensions. The shipped models use a Smolyak sparse grid [1] instead, with 1105
and 15 713 nodes. This page explains how a sparse grid is built, what it keeps,
what it throws away, and when that trade pays. It builds on the page on
[the Bernstein ellipse](bernstein.md), which sets the convergence rate along a
single parameter.

## Nested levels in one dimension

Each parameter axis, mapped onto $[-1, 1]$, is resolved in *levels*. Level $l$
has $m(0) = 1$ node, the midpoint, and

$$
m(l) \ = \ 2^l + 1, \qquad \xi_j \ = \ \cos\frac{j\pi}{2^l}, \qquad j = 0, \dots, 2^l,
$$

for $l \ge 1$. These are the Chebyshev points of the Bernstein page, also known
as Clenshaw–Curtis points [6]. Doubling the number of intervals at each level
makes the levels *nested*: the nodes of level $l$ are the even-numbered nodes of
level $l + 1$, so every level reuses all the solves of the levels below it and
adds only the new midpoints.

Let $I_l$ be the interpolant on the nodes of level $l$. What level $l$ adds is the
*hierarchical surplus*

$$
\Delta_l \ = \ I_l - I_{l-1}, \qquad \Delta_0 \ \equiv \ I_0 ,
$$

and the levels telescope, $I_L = \sum_{l \le L} \Delta_l$. A surplus is the
difference of two interpolation errors, so it is no larger than the error of the
level below. For an analytic map it therefore shrinks geometrically in the number
of nodes, at the rate set by the Bernstein ellipse.

## Products, and the simplex

In $d$ parameters the full tensor-product interpolant at level $L$ on every axis
telescopes the same way, into a sum of *mixed surpluses*,

$$
\bigotimes_{k=1}^{d} I_L \ = \ \sum_{l_1, \dots, l_d \le L} \Delta_{l_1} \otimes \cdots \otimes \Delta_{l_d} .
$$

A mixed surplus such as $\Delta_{l_1} \otimes \Delta_{l_2}$ with both levels high
is the part of the map that needs fine resolution in two parameters *at once*.
If the parameters act on the solution nearly independently, those parts are
small, and the Smolyak construction drops them. It keeps only the multi-indices
in the simplex $|\mathbf{l}|_1 = l_1 + \dots + l_d \le \ell$,

$$
\mathcal{A}_\ell \ = \ \sum_{|\mathbf{l}|_1 \le \ell} \Delta_{l_1} \otimes \cdots \otimes \Delta_{l_d} .
$$

Collecting terms gives the *combination technique* [1, 3, 7], a signed sum of
ordinary tensor interpolants on small, anisotropic grids,

$$
\mathcal{A}_\ell \ = \ \sum_{\ell - d + 1 \le |\mathbf{l}|_1 \le \ell}
(-1)^{\ell - |\mathbf{l}|_1} \binom{d - 1}{\ell - |\mathbf{l}|_1}
\bigotimes_{k=1}^{d} I_{l_k} .
$$

In two parameters this is the tensor grids on the diagonal $l_1 + l_2 = \ell$ with
coefficient $+1$, minus those on the diagonal below with $-1$. The subtracted grids
are exactly the overlaps the added ones count twice. Because the one-dimensional
levels are nested, the union of all these subgrids is the sparse grid, every node
is solved once, and the sparse interpolant reproduces the stored value at every
node [2].

The node count grows like $m (\log m)^{d-1}$ for $m$ nodes along each axis,
instead of $m^d$ [4]:

| $d$ | level $\ell$ | sparse grid | tensor grid, $33^d$ |
|---|---|---|---|
| 2 | 5 | 145 | 1089 |
| 4 | 5 | 1105 | 1 185 921 |
| 8 | 5 | 15 713 | $\approx 1.4 \times 10^{12}$ |

## Try it

```{raw} html
<div class="lm-smol"></div>
```

The figure is a two-parameter sparse grid, computed in the browser.

- **The square** shows the sparse grid at level $\ell$, with larger
  dots for nodes from coarser levels, over the faint full tensor grid of the
  same finest resolution.
- **The multi-index plane** shades the simplex
  $l_1 + l_2 \le \ell$ and labels the cells the combination technique uses with
  their coefficients. The animation steps through those cells and rings the
  corresponding subgrid on the square, blue for $+1$ and red for $-1$. Hover over
  or tap a cell to hold it.
- **The error panel** is the largest relative error on a $40 \times 40$ test
  grid, against the number of nodes, which for the model is the number of
  elliptic solves. Blue is the sparse grid at levels 0 to 7, black the full
  tensor grid with $q \times q$ Chebyshev points for $q = 2, \dots, 60$.

The test function blends a sum of one-parameter functions into a function of the
two parameters together,

$$
f(x, y) \ = \ (1 - \kappa) \, \frac{g(x) + g(y)}{2} + \kappa \, h(x, y),
$$

with

$$
g(t) \ = \ \frac{1}{1 + 4 (t - 0.15)^2}, \qquad
h(x, y) \ = \ \frac{1}{1 + 4 \left[ (x - 0.15)^2 + (y - 0.15)^2 \right]} ,
$$

and the slider sets the coupling $\kappa$.

- **At $\kappa = 0$**, $f$ is a sum of one-parameter functions, every mixed
  surplus vanishes, and the sparse grid is as accurate as its finest
  one-dimensional level. It reaches $4 \times 10^{-14}$ with 321 nodes at
  $\ell = 6$, where the tensor grid needs about 3600 nodes for $3 \times 10^{-13}$.
- **At $\kappa = 0.02$**, two percent of coupling is enough to stall the sparse
  grid. At $\ell = 7$ it has 705 nodes and an error of $1 \times 10^{-4}$, while a
  $27 \times 27$ tensor grid with 729 nodes reaches $4 \times 10^{-6}$.

Both $g$ and $h$ are analytic on the whole square. What separates them is how far
the mixed surpluses reach, which is a property of the two parameters *together*,
not of either one alone.

## What a sparse grid buys, and what it does not

The table compares node counts at the same finest resolution. That measures the
saving in solves, not the accuracy. At that resolution the tensor grid is usually the
more accurate one, by roughly the weight of the discarded mixed surpluses, and the
figure shows that this can be a great deal. A sparse grid pays off when the
coupling between parameters is weak or smooth compared with the dependence on
each parameter alone, and the balance tips further towards it as the dimension
grows, because the tensor grid pays $m^d$ while the sparse grid pays only the
logarithmic factor. Error bounds for analytic maps are in Refs. [2] and [3].

Whether the trade pays for puncture initial data is therefore measured rather
than assumed. The paper [8] reports the held-out error of the shipped models
level by level, against the number of solves, in its section *Offline cost and
certified refinement*. Because the levels are nested, the corpus of level 5
contains every lower level, and that error curve costs no extra solves. The same
section estimates what a tensor grid would need for the *same accuracy*, by
extrapolating the measured per-axis convergence rates over independent axes. At
the level-5 accuracy of the value-only models that is about four times as many
nodes as the sparse grid in four dimensions, and about sixteen times as many in
eight. That is the real saving for this problem. It is much smaller than the
factor of a thousand at the same finest resolution in the table, and it grows
with the dimension, as the theory says it should.

When the parameters differ in difficulty, the isotropic simplex spends nodes
where they are not needed. A *dimension-adaptive* index set [5] grows the simplex
greedily along the axes whose surpluses stay large, and is the natural next step
for the slowly converging spin axes.

The node counts in the table come from the package itself:

```python
from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd_2c import smolyak_points

print([smolyak_points(2, level) for level in range(8)])   # [1, 5, 13, 29, 65, 145, 321, 705]
print(smolyak_points(4, 5), smolyak_points(8, 5))         # 1105 15713
```

## References

1. S. A. Smolyak, *Quadrature and interpolation formulas for tensor products of
   certain classes of functions*, Soviet Math. Dokl. **4**, 240–243 (1963).
2. V. Barthelmann, E. Novak and K. Ritter, *High dimensional polynomial
   interpolation on sparse grids*, Adv. Comput. Math. **12**, 273–288 (2000).
3. F. Nobile, R. Tempone and C. G. Webster, *A sparse grid stochastic collocation
   method for partial differential equations with random input data*, SIAM J.
   Numer. Anal. **46**, 2309–2345 (2008),
   [doi:10.1137/060663660](https://doi.org/10.1137/060663660).
4. H.-J. Bungartz and M. Griebel, *Sparse grids*, Acta Numer. **13**, 147–269
   (2004).
5. T. Gerstner and M. Griebel, *Dimension-adaptive tensor-product quadrature*,
   Computing **71**, 65–87 (2003).
6. C. W. Clenshaw and A. R. Curtis, *A method for numerical integration on an
   automatic computer*, Numer. Math. **2**, 197–205 (1960).
7. M. Griebel, M. Schneider and C. Zenger, *A combination technique for the
   solution of sparse grid problems*, in *Iterative Methods in Linear Algebra*,
   edited by P. de Groen and R. Beauwens (North-Holland, Amsterdam, 1992),
   pp. 263–281.
8. F. De Ceuster, *A differentiable parametric model of binary-black-hole initial
   data: I. Conformally flat Bowen–York punctures*,
   [arXiv:2609.10173](https://arxiv.org/abs/2609.10173).
