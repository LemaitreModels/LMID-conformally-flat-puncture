# From parameters to a field

The model interpolates complete solutions. Offline, the constraint equation is solved
at a set of parameter points, the *nodes*, and each solution is stored as its values
on one spectral grid. At a new parameter point the model combines the stored fields
into a prediction of the whole field. That prediction can be used directly, or it
becomes the starting guess of the full solver, whose residual is checked independently
of how good the guess was. This page follows one query through the model, and links
to the pages that explain each ingredient in depth.

## What is interpolated

One solution is the array `U` of the regular part $u$ of the conformal factor at the
$45 \times 40 \times 8 = 14\,400$ nodes of the [single spectral patch](solver.md).
The patch is built around the two punctures, so its nodes are fixed in
computational coordinates and move in physical space when $b$ changes. Parameter
interpolation therefore always works on arrays of the same shape, whatever the
binary, and evaluating the result at a physical point is a separate, later step
([Use the field](field.md)).

## One parameter: Lagrange interpolation

With one parameter $\theta$ and solutions $U_i$ at nodes $\theta_i$, the value model is

$$
\tilde U(\theta) \ = \ \sum_i L_i(\theta) \, U_i ,
$$

with $L_i$ the Lagrange cardinal polynomials, $L_i(\theta_j) = \delta_{ij}$. The same
scalar weights $L_i(\theta)$ multiply every one of the 14 400 numbers in $U_i$, so a
query costs one weighted sum of stored arrays. The weights are evaluated in
barycentric form, which is fast and numerically stable [1]. On Chebyshev-type nodes
the error of an analytic map falls geometrically with the number of nodes, at a rate
set by the nearest singularity in the complex parameter plane; [the Bernstein
ellipse](bernstein.md) shows how.

## Many parameters: a sparse grid

A tensor grid of $m$ nodes per axis costs $m^d$ solves, which is out of reach in
eight dimensions. The model uses Smolyak's sparse grid instead, written as a signed
sum of small, coarse tensor grids (the *combination technique*):

$$
\tilde U(\theta) \ = \ \sum_{\boldsymbol\ell} c_{\boldsymbol\ell} \,
\mathcal I_{\boldsymbol\ell} U(\theta) .
$$

Each $\mathcal I_{\boldsymbol\ell}$ is a tensor-product interpolant with $\ell_k$
levels of nested Clenshaw–Curtis nodes on axis $k$, and the integers
$c_{\boldsymbol\ell}$ make the sum exact on the sparse polynomial space. The nodes
are nested, so the subgrids share most of their points: at level 5 the 4-D model
needs 1105 distinct solves and the 8-D model 15 713. [Sparse grids](smolyak.md)
builds this step by step.

## Derivatives where they help: Hermite on two axes

A solved node can also store its tangent $\partial U/\partial\theta_k$, at the cost of
one linear solve. On an *enhanced* axis the model interpolates values and tangents
together (Hermite interpolation), which roughly doubles the convergence rate per
node. The shipped models enhance exactly two axes, the aligned spins $\chi_{Ay}$ and
$\chi_{By}$, which converge most slowly [2], and store the one cross derivative
$\partial^2 U/\partial\chi_{Ay}\partial\chi_{By}$ that a two-axis tensor Hermite
interpolant needs. [Sensitivities](sensitivities.md) derives both and explains why
the cross term cannot be left out.

Inside the sparse grid, an axis is enhanced in a subgrid only when that subgrid has
more than one node on it. A one-node Hermite factor would be the linear Taylor
expansion about the box centre, which is a poor predictor far from it, so level-0
factors stay value-only.

## Compression

Storing four fields of 14 400 numbers at each of 15 713 nodes takes about 7 GB. The
fields are highly redundant, so the model stores them in a reduced basis of $r$
spatial modes ($r = 500$ in 8-D) and interpolates the $r$ coefficients instead of
the 14 400 values. [Compression](compression.md) explains the basis and what it
costs in accuracy.

## One query, step by step

At a new parameter point $\theta$:

1. **One-dimensional weights.** For every axis and every node count that occurs on
   it, evaluate the cardinal weights at $\theta_k$: Lagrange weights on a value axis,
   the Hermite pair on an enhanced axis.
2. **Combine.** Each subgrid's weight on its nodes is the outer product of its axis
   weights times its coefficient $c_{\boldsymbol\ell}$. These weights are added into
   one vector over the distinct nodes, so a node shared by many subgrids is read once.
3. **Contract.** Multiply that vector with the stored coefficient arrays: values,
   tangents on the two enhanced axes, and the cross term. The result is the $r$
   reduced coefficients $c(\theta)$.
4. **Decode.** $\tilde U(\theta) = \bar U + \Phi \, c(\theta)$, with $\bar U$ the mean
   field and $\Phi$ the basis.
5. **Refine, if asked.** `evaluate_polished` hands $\tilde U$ to the solver as its
   starting guess and checks the residual of what comes back ([Refine a model
   prediction](refine.md)).

Steps 1–4 are `model.evaluate` (NumPy) and `model.evaluate_jax` (JAX, the same
arithmetic). The JAX version is what `jax.jacfwd` differentiates.

## Two meanings of "the derivative"

`jax.jacfwd(model.evaluate_jax)` is the exact derivative of the interpolant
$\tilde U$. The *sensitivity* of the solution, $\partial U/\partial\theta$ at a
certified solve, is a different quantity, computed by one linear solve
(`applications.sensitivity_3d_qc`). They differ by the derivative of the
interpolation error. A single tensor-product Hermite interpolant reproduces the
stored sensitivity at its nodes along its enhanced axes. The shipped model only
approximates that: subgrids with one node on an axis are value-only there, so the
sparse combination need not reproduce a stored tangent at every node, and the
compression projects the tangents onto the basis. Everywhere the interpolant's
derivative is an approximation, usually less accurate than its value
([Sensitivities](sensitivities.md)).

## References

1. J.-P. Berrut and L. N. Trefethen, *Barycentric Lagrange interpolation*,
   SIAM Rev. **46**, 501–517 (2004).
2. F. De Ceuster, *A differentiable parametric model of binary-black-hole initial
   data: I. Conformally flat Bowen–York punctures*,
   [arXiv:2609.10173](https://arxiv.org/abs/2609.10173).
