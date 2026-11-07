# Sensitivities and Hermite interpolation

*Differentiable* in the title of the paper [5] means that for every queried parameter
point $\theta$ the model returns the derivative $\partial U/\partial\theta$ of the
initial data along with the data. In numerical relativity the usual way to see how
initial data change with a parameter is to solve again at a nearby parameter
value and take a finite difference. Here the derivative comes out of the solve
itself, for the price of one linear solve, and the stored derivatives make the
parameter interpolation more accurate as well. This page explains both, building
on the page on [the Bernstein ellipse](bernstein.md).

## The tangent equation

A certified solve finds the discrete field $U$ that makes the residual of the
constraint equation vanish, $R(U, \theta) = 0$, with Newton's method. As $\theta$
varies the solution follows, so $R(U(\theta), \theta) \equiv 0$ holds identically,
and differentiating that identity once gives the *tangent equation*

$$
J \, \frac{\partial U}{\partial \theta_k} \ = \ - \frac{\partial R}{\partial \theta_k},
\qquad J \ \equiv \ \frac{\partial R}{\partial U} .
$$

This is the implicit function theorem [1]. Wherever the Jacobian $J$ is invertible,
the solution is a smooth function of the parameters (an analytic one, when $R$ is
analytic), and its derivative is the solution of a *linear* equation.

- **It is cheap.** $J$ is the Jacobian that Newton's method already uses, so each
  tangent is one more linear solve with the same operator and preconditioner [4],
  and no further nonlinear solve. A Newton solve performs several such linear solves.
- **It is exact.** The right-hand side $\partial R/\partial\theta_k$ is known in
  closed form on every axis, because each Bowen–York tensor is linear in its own
  momentum or spin. No finite difference enters, so there is no step size to
  choose and no loss to cancellation. A finite difference in double precision is
  limited to about $10^{-8}$ by the competition between truncation and rounding.
- **Forward mode is the right mode.** There are at most eight parameters and the
  output is a whole field, so one linear solve per parameter is the cheap
  direction. An adjoint (reverse-mode) calculation pays off in the opposite case,
  many parameters and one scalar output [2].

The theorem also says where the parameter-to-solution map can fail to be
analytic, which is what the Bernstein ellipse page needs. A singularity sits where
the source term itself is singular, as when the punctures coincide at $b = 0$, or
where $J$ stops being invertible. In the toy problem of the figure below it is the
second kind.

## Two different derivatives

Between the nodes the model is an interpolant $\tilde U(\theta)$, a polynomial in
each parameter, and it is implemented in JAX [3], so forward-mode automatic
differentiation [2] returns the exact derivative $\partial\tilde U/\partial\theta$
of that polynomial. That is not the same thing as the sensitivity of the certified
solution. The difference is the derivative of the interpolation error, and the
derivative of a small error need not be small. The interpolation error oscillates,
and differentiating it multiplies each oscillation by its frequency, so the
derivative error is typically an order of magnitude larger than the value error,
and largest near the ends of the interval. The figure shows both.

## Hermite interpolation

A node that has been solved can also store its tangent, for one linear solve.
Interpolating the values alone throws that information away. *Hermite*, or
osculatory, interpolation matches both the value and the derivative at every
node, so $Q$ nodes determine a polynomial of degree $2Q - 1$ instead of $Q - 1$.
The paper builds the Hermite cardinal functions from the same barycentric
Lagrange cardinals $L_i$ as the value interpolant,

$$
H_i(\theta) \ \equiv \ \left[ 1 - 2 L_i'(\theta_i) (\theta - \theta_i) \right] L_i(\theta)^2,
\qquad
\hat H_i(\theta) \ \equiv \ (\theta - \theta_i) \, L_i(\theta)^2,
$$

with $L_i'(\theta_i) = \sum_{j \ne i} (\theta_i - \theta_j)^{-1}$. They satisfy
$H_i(\theta_j) = \delta_{ij}$, $H_i'(\theta_j) = 0$, $\hat H_i(\theta_j) = 0$ and
$\hat H_i'(\theta_j) = \delta_{ij}$, so the interpolant
$\sum_i \left( H_i \, U_i + \hat H_i \, \partial_\theta U_i \right)$ reproduces the
certified value and the certified tangent at every node. The derivative the model
exposes at a node is then the certified sensitivity itself, not the derivative of
a value interpolant.

For an analytic map the Hermite error falls about twice as many decades per node
as the Lagrange error, so it needs about half the nonlinear solves for the same
accuracy. That saving matters most where many nodes are needed, on the slow axes
near a Bernstein wall. The shipped models therefore enhance only the two aligned
spin components, their slowest axes, and keep plain Lagrange interpolation on the
others.

## Try it

```{raw} html
<div class="lm-herm"></div>
```

The figure stands in for the constraint solve with one nonlinear equation for one
number,

$$
R(u, \theta) \ = \ u^3 + u - g(\theta) \ = \ 0, \qquad
g(\theta) \ = \ \frac{2}{3\sqrt3} \, \frac{\theta - a}{\beta} .
$$

For real $\theta$ it has a single real root, which the figure finds by Newton's
method. Its tangent equation is $(3u^2 + 1)\,\partial u/\partial\theta = g'(\theta)$.
The Jacobian $3u^2 + 1$ vanishes at $u = \pm i/\sqrt3$, which the solution reaches
at $\theta_\ast = a \pm i\beta$. These are square-root branch points, the
singularities that set the Bernstein ellipse, and the first two sliders move them.

- **The solution** $u(\theta)$ from the exact solve in black, its
  Lagrange interpolant on the $Q$ nodes in blue and its Hermite interpolant in
  dashed red. The red ticks at the nodes are the stored tangents. A probe moves
  along the interval and draws the tangent line of each curve; hovering over the
  curves moves it by hand.
- **The derivative**, from the tangent equation in black, with the
  derivatives of the two interpolants. The red dots are the stored tangents,
  which the Hermite interpolant matches exactly.
- **The errors**, the largest relative errors on the interval against the
  number of nodes, which is the number of nonlinear solves. Filled dots are value
  errors, open dots derivative errors, blue for Lagrange and red for Hermite.

Things to try:

- **Few nodes.** Set $Q$ to 4 or 5 and watch the probe. The blue tangent line
  tilts away from the black one between the nodes, while the red one stays close.
- **Value against derivative.** In the error panel the open dots sit about an
  order of magnitude above the filled ones, for both methods.
- **The rate.** With the default singularity at $\theta_\ast = \pm 0.3i$, so
  $\rho = 1.344$, the Lagrange value error falls from $3 \times 10^{-4}$ at
  $Q = 16$ to $1 \times 10^{-6}$ at $Q = 32$, while the Hermite error falls from
  $2.5 \times 10^{-6}$ to $7.5 \times 10^{-11}$. That is about twice as many
  decades per node.
- **A far singularity.** Raise $\operatorname{Im}\theta_\ast$ to 1. Both methods now
  need only a handful of nodes, and the Hermite saving is worth fewer solves.

## Several enhanced axes need the cross derivative

With two enhanced axes $A$ and $B$, the tensor-product Hermite interpolant needs
four numbers at each node: the value, the two tangents and the cross derivative
$\partial_A \partial_B U$. The cross derivative comes from differentiating the
residual identity a second time,

$$
J \, \frac{\partial^2 U}{\partial\theta_A \, \partial\theta_B} \ = \ - \left[
\frac{\partial^2 R}{\partial\theta_A \, \partial\theta_B}
+ \frac{\partial^2 R}{\partial\theta_A \, \partial U} \frac{\partial U}{\partial\theta_B}
+ \frac{\partial^2 R}{\partial\theta_B \, \partial U} \frac{\partial U}{\partial\theta_A}
+ \frac{\partial^2 R}{\partial U^2} \left[ \frac{\partial U}{\partial\theta_A}, \frac{\partial U}{\partial\theta_B} \right]
\right] ,
$$

which is one more linear solve with the same Jacobian.

Leaving it out is not a harmless saving. At a node, $H_i' = 0$, so every product
of cardinals except $\hat H_{i_A} \hat H_{i_B}$ has a vanishing mixed derivative
there. The interpolant's mixed derivative at a node is therefore exactly the
stored cross term, and dropping that term forces
$\partial_A \partial_B \tilde U = 0$ at every node, which is wrong. The same holds
in more enhanced axes, where all $2^k$ mixed derivatives are needed. The paper
measures the consequence: enhancing all four axes of the aligned-spin model with
tangents alone makes the held-out median about a hundred times worse than
enhancing none, and adding the pairwise cross terms does not repair it [5]. These are the same mixed terms that
decide whether a [sparse grid](smolyak.md) pays off.

## References

1. S. G. Krantz and H. R. Parks, *The Implicit Function Theorem: History, Theory,
   and Applications* (Birkhäuser, Boston, 2002).
2. A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques
   of Algorithmic Differentiation*, 2nd edition (SIAM, Philadelphia, 2008).
3. J. Bradbury *et al.*, *JAX: composable transformations of Python+NumPy
   programs* (2018), [github.com/jax-ml/jax](https://github.com/jax-ml/jax).
4. D. A. Knoll and D. E. Keyes, *Jacobian-free Newton–Krylov methods: a survey of
   approaches and applications*, J. Comput. Phys. **193**, 357–397 (2004).
5. F. De Ceuster, *A differentiable parametric model of binary-black-hole initial
   data: I. Conformally flat Bowen–York punctures*,
   [arXiv:2609.10173](https://arxiv.org/abs/2609.10173), sections
   *Differentiability*, *Gradient-enhanced (Hermite) collocation* and *Offline
   cost and certified refinement*.
