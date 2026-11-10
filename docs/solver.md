# The constraint solver

Every initial-data set the package returns comes out of the same elliptic solve.
The parametric model only supplies the starting guess. This page explains that solve
step by step: the equation, the single coordinate patch it is discretized on, the
Newton–Krylov iteration, and the NumPy and SciPy calls that implement each step. The
method is the single-domain spectral method of Ansorg, Brügmann and Tichy [1], the
method of the TwoPunctures code, with Gauss–Legendre instead of Chebyshev–Gauss nodes
in the angular direction.

## The equation

The initial data are conformally flat and maximally sliced, in York's conformal
transverse-traceless decomposition [4]. The spatial metric is
$\gamma_{ij} = \psi^4 \delta_{ij}$, and the extrinsic curvature is
$K^{ij} = \psi^{-10} \hat A^{ij}$, with $\hat A^{ij}$ the sum of the Bowen–York
momentum and spin tensors of the two punctures [3]. Each of those tensors is
transverse and trace-free, so the momentum constraint holds analytically for any
momenta and spins. The Hamiltonian constraint becomes one nonlinear elliptic equation
for the conformal factor,

$$
\Delta \psi \ = \ -\tfrac{1}{8} \, \psi^{-7} \, \hat A_{ij} \hat A^{ij} ,
$$

with $\Delta$ the flat Laplacian. Following Brandt and Brügmann [2], the singular part
of $\psi$ is written down in closed form, $\psi = \psi_{\rm BL} + u$ with
$\psi_{\rm BL} = 1 + m_A / 2 r_A + m_B / 2 r_B$, and the solver works with the regular
correction $u$,

$$
\Delta u \ = \ -\tfrac{1}{8} \, (\psi_{\rm BL} + u)^{-7} \, \hat A_{ij} \hat A^{ij} .
$$

The punctures sit on the $z$-axis at $z = \pm b$, so $b$ is half the coordinate
separation. The source vanishes at each puncture, so $u$ is finite there, but it
carries an $r^3$ term that is only twice differentiable in Cartesian coordinates. A
spectral method on a Cartesian grid would converge slowly because of it. The chart
below removes the problem.

## One patch for all of space

The solver uses prolate-spheroidal coordinates $(A, B, \phi)$ with the two punctures
at the foci [1],

$$
\frac{\rho}{b} \ = \ \frac{2 A \sqrt{1 - B^2}}{1 - A^2}, \qquad
\frac{z}{b} \ = \ \frac{(1 + A^2) \, B}{1 - A^2},
$$

with $A \in [0, 1]$, $B \in [-1, 1]$ and $\phi$ the azimuthal angle around the
$z$-axis. Curves of constant $A$ are ellipses and curves of constant $B$ are
hyperbolas, all with foci at the punctures. One rectangle covers all of space:

- $A = 1$ is spatial infinity, so no outer boundary has to be placed at a finite
  radius.
- $A = 0$ is the segment of the axis between the punctures, $|z| \le b$.
- $B = \pm 1$ are the two outer parts of the axis, $|z| \ge b$.
- The corners $(A, B) = (0, \pm 1)$ are the punctures themselves.

None of these edges is a physical boundary. They are places where the chart
degenerates, and the solver needs no excision surface. In these coordinates the
distance to each puncture is a rational function,
$r_\pm / b = \left[ A^2 (1 \pm B) + (1 \mp B) \right] / (1 - A^2)$, so the $r^3$ term
in $u$ becomes analytic and no longer slows convergence. What does limit it are
logarithmic terms at $A = 1$, of the form $(1 - A)^5 \log(1 - A)$ when the total
momentum vanishes, which Ansorg, Brügmann and Tichy derive for axisymmetric data [1].

The angle is represented in a second coordinate $s \in (-1, 1)$, with

$$
B \ = \ \frac{2 s}{1 + s^2}, \qquad \sqrt{1 - B^2} \ = \ \frac{1 - s^2}{1 + s^2} ,
$$

the angular coordinate of Eq. (61) of Ref. [1], up to orientation. Near a puncture
$1 - B$ behaves like $(1 - s)^2$, so equally spaced steps in $s$ become ever finer
steps in $B$ there. Nodes placed in $s$ crowd around each puncture, where the
solution varies fastest.

```{raw} html
<div class="lm-abt"></div>
```

The figure draws the grid in the computational square $(A, s)$ and its image in the
meridian plane, the plane through the axis. The lower half of the meridian panel is
the half-plane $\phi + \pi$. Each node in the square is a ring of $N_\phi$ points
around the axis in three dimensions.

- **The sweep** moves one coordinate line through both panels, first a line of
  constant $A$ and then a line of constant $s$. Hovering over either panel picks a
  point and shows its two coordinate lines in both panels, with its coordinates.
- **The node sliders** set the number of radial and angular nodes. The shipped
  models use 45 and 40.
- **The view slider** zooms the meridian panel, from just around the two punctures
  out to sixty times $b$.
- **The checkbox** puts the Gauss–Legendre nodes in $B$ instead of $s$, which is how
  this package first built its grid. Zoom in on the punctures and toggle it: with the nodes
  in $B$ they spread out along the inner axis, and with the nodes in $s$ they gather
  in rings around each puncture.

## The discrete unknowns

The grid has three kinds of nodes:

- **Radially,** $K = N_A + 1$ Chebyshev–Gauss nodes
  $A_i = \sin^2 \left[ \pi (i + \tfrac{1}{2}) / 2K \right]$ [Eq. (8) of Ref. 1]. No
  node sits on $A = 0$ or $A = 1$.
- **In angle,** $N_B$ Gauss–Legendre nodes in $s$, so no node sits on the axis
  either.
- **Azimuthally,** $N_\phi$ equally spaced nodes in $\phi$.

The unknowns are the values $U_{ijk} = u(A_i, B_j, \phi_k)$ at the nodes. On the
production grid that is $45 \times 40 \times 8 = 14\,400$ numbers. In between, $u$ is
a Fourier series in $\phi$, and each Fourier mode $u_m$ is a polynomial in $A$ and $s$
times a known factor,

$$
u_m(A, B) \ = \ w_m(A, B) \, v_m(A, s), \qquad
w_m \ = \ (A - 1) \, (1 - B^2)^{k_m / 2}, \qquad k_m \ = \ \min(|m|, 4) .
$$

The factor $A - 1$ makes $u$ vanish at infinity. The factor $(1 - B^2)^{k_m/2}$ makes
each azimuthal mode vanish on the axis as fast as regularity requires. For $|m| > 4$
the remaining power is a polynomial in $s$ and is carried by $v_m$. Both factors are
differentiated analytically, and the spectral differentiation matrices act only on
the smooth factor $v_m$. Because the boundary behavior is built into the basis, every
collocation equation is the PDE itself, and there are no boundary rows. The $1/A$
terms of the Laplacian select the regular solution on the inner axis.

The coordinates are orthogonal, so the flat Laplacian has no mixed $A$–$B$
derivatives,

$$
\Delta u \ = \ \frac{(1 - A^2)^2}{b^2 \mathcal{D}} \left[
\frac{(1 - A^2)^2}{4 A} \, \partial_A \left( A \, \partial_A u \right)
+ \partial_B \left( (1 - B^2) \, \partial_B u \right) \right]
+ \frac{\partial_\phi^2 u}{\rho^2},
$$

with $\mathcal{D} \equiv (1 + A^2)^2 - B^2 (1 - A^2)^2$. In the Fourier basis $\partial_\phi^2 \to -m^2$, so the discrete Laplacian splits into
one two-dimensional block per mode $m$, and the full three-dimensional matrix is never
built.

## A certified solve, step by step

Collocation turns the equation into a nonlinear system $R(U) = 0$ with

$$
R(U) \ = \ M_0 \, U + \tfrac{1}{8} \, (\psi_{\rm BL} + U)^{-7} \, \hat A^2 ,
\qquad
J(U) \ = \ M_0 - \tfrac{7}{8} \, \mathrm{diag} \left[ (\psi_{\rm BL} + U)^{-8} \, \hat A^2 \right] ,
$$

where $M_0$ is the discrete Laplacian and $J = \partial R / \partial U$ its Jacobian.
The nonlinear term is local: it multiplies each node by its own number. The solve
takes five steps.

**1. Build the grid, once.** `solver_3d.make_problem(44, 40, 8)` lays the nodes and
builds the one-dimensional differentiation matrices (`operators_abt.build_grid`). The
first argument is $N_A$, so this grid has 45 radial nodes. The Gauss–Legendre nodes
come from `numpy.polynomial.legendre.leggauss`, and the differentiation matrices are
barycentric. Nothing here depends on the binary.

**2. Set up one binary.** A `solver_3d.Slice3D` holds the separation, the bare masses
and the momentum and spin vectors. `solver_3d.assemble` evaluates $\psi_{\rm BL}$ and
$\hat A_{ij} \hat A^{ij}$ at every node, with the Bowen–York tensors summed point by
point (`source_3d.A2_at_nodes_3d`). It also attaches the linear operator, taken from
a per-grid cache.

**3. Evaluate the residual.** `numpy.fft.rfft` along $\phi$ turns the node values into
modes $m = 0, \dots, N_\phi / 2$. Each mode's Laplacian block acts on its smooth factor
$v_m$. The nonlinear term is evaluated at the nodes and transformed the same way. The
rows of the system differ in scale by many orders of magnitude, from $(1 - A^2)^4$
near infinity to $1/A$ near the inner axis. The residual is therefore measured after
dividing each row by its largest coefficient, referred to the physical field $u$,

$$
\|R\|^{(u)}_\infty \ \equiv \ \max_m \max_i
\frac{\bigl| \hat R^{(m)}_i \bigr|}{\max_k \bigl| (M_0^{(m)})_{ik} \bigr| / |w_{m,k}|} .
$$

This is an exact rescaling of the equations, so it does not move the solution. It
does make residuals comparable across grids and across ways of factoring the basis.

**4. Take a Newton step.** Each step solves $J \, \delta U = -R$ with GMRES [6], from
`scipy.sparse.linalg.gmres`. $J$ is never assembled. A `LinearOperator` applies it
term by term, the Laplacian mode by mode and the nonlinear term at the nodes,

$$
\widehat{(J v)}^{(m)} \ = \ \widehat{(M_0 v)}^{(m)}
+ \Bigl\{ \mathcal{F}_\phi \bigl[ D_{\rm nl} \, \mathcal{F}_\phi^{-1} (\hat v) \bigr] \Bigr\}^{(m)},
\qquad
D_{\rm nl} \ \equiv \ -\tfrac{7}{8} \, (\psi_{\rm BL} + U)^{-8} \, \hat A^2 ,
$$

with $\mathcal{F}_\phi$ the FFT in $\phi$. This action is exact, so the converged
Newton step is the true one. A Newton method whose linear solves use only
Jacobian–vector products is a Newton–Krylov method [5]. Because $D_{\rm nl}$ varies
with $\phi$, it couples the Fourier modes, so $J$ is not block diagonal even though
$M_0$ is. The preconditioner, a second `LinearOperator`, is the exact inverse of $M_0$
alone (next section). It leaves out $D_{\rm nl}$, and that costs only a few GMRES
iterations per step, three in the example below. GMRES runs to a loose relative tolerance of $10^{-4}$, an inexact Newton
step, and the measured steps below still converge quadratically.

**5. Stop and certify.** The loop (`newton_loop.py`) keeps the best iterate, stops
once the residual is a decade below the requested tolerance, and also stops if the
best residual fails to halve twice in a row. A solve is certified only if
$\|R\|^{(u)}_\infty \le 10^{-11}$ (`certification.CERT_TOL_U`), and every point where
data leave the package checks that gate.

## The preconditioner: fast diagonalization

Each Laplacian block becomes *separable* after a row scaling. Only the common
prefactor $(1 - A^2)^2 / (b^2 \mathcal{D})$ of the Laplacian depends on $A$ and $B$
together. Dividing each row by it, and by the two basis factors, leaves a Kronecker
sum of two one-dimensional operators,

$$
\mathcal{G}^{(m)}_A \otimes I + I \otimes \mathcal{G}^{(m)}_B ,
$$

of sizes $K$ and $N_B$. On the field array this is the Sylvester operator
$v \mapsto \mathcal{G}_A v + v \, \mathcal{G}_B^{T}$. Diagonalizing each factor once,
$\mathcal{G}_A = S_A \Lambda_A S_A^{-1}$ and
$\mathcal{G}_B^{T} = S_B \Lambda_B S_B^{-1}$, turns the solve into a division by
$\lambda^A_i + \lambda^B_j$ between four small matrix products. This is the fast
diagonalization method of Lynch, Rice and Thomas [7].

- **Setup** is two small eigendecompositions per mode, with `numpy.linalg.eig`
  (`separable.py`). Both factors are nonsymmetric, and the conditioning of $S_A$ and
  $S_B$ limits the accuracy of the inverse.
- **One application** costs $O \bigl( N_A N_B (N_A + N_B) \bigr)$ operations, against
  the $O \bigl( (N_A N_B)^2 \bigr)$ of a dense factorization.
- **The binary never enters.** Every coefficient carries exactly one factor of
  $b^{-2}$, which the row scaling cancels, and the nonlinear term is left out. The
  factors are therefore built once per grid and reused for every binary, every Newton
  step and every tangent solve. At the production grid they take about 0.1 MiB,
  against 79 MiB for the dense blocks.

On the production grid this preconditioner makes a certified solve 56 to 58 times
cheaper than dense LU factorizations of the same blocks [8]. For axisymmetric data,
$N_\phi = 1$, there is no mode coupling. The solver then factors the block of the full
Jacobian with `scipy.linalg.lu_factor`, and GMRES converges in one iteration.

## Which library does what

| Step | Library | Module |
|---|---|---|
| Nodes and differentiation matrices | NumPy | `solver/operators_abt.py` |
| Bowen–York source $\hat A_{ij} \hat A^{ij}$ | NumPy | `solver/source_3d.py` |
| FFT in $\phi$ | `numpy.fft.rfft`, `irfft` | `solver/solver_3d.py`, `solver/solver_3d_nk.py` |
| Newton loop and certification | NumPy | `solver/newton_loop.py`, `parametric/certification.py` |
| Krylov solve | `scipy.sparse.linalg.gmres`, `LinearOperator` | `solver/solver_3d_nk.py` |
| Preconditioner setup | `numpy.linalg.eig` | `solver/separable.py` |
| Axisymmetric case, $N_\phi = 1$ | `scipy.linalg.lu_factor` | `solver/solver_3d_nk.py` |
| Tangents $\partial U / \partial\theta$ | the same GMRES and preconditioner | `applications/sensitivity_3d.py` |
| Parametric model and its derivatives | JAX | `parametric/` |

The elliptic solve is NumPy and SciPy, in double precision throughout. JAX carries
the parametric model on top of it, where forward-mode automatic differentiation gives
the derivative of the interpolant with respect to the parameters. The solve itself is
never differentiated through. Its parameter derivatives come from the tangent equation,
one more linear solve with the same Jacobian, as explained on the page on
[sensitivities](sensitivities.md). JAX appears in the solver only for small
closed-form pieces, such as evaluating $\psi_{\rm BL}$, and in checks such as an
automatic-differentiation test that the Bowen–York tensors are transverse.

## Try it

A quasi-circular, equal-mass binary with aligned spins, solved on the production grid
from $u = 0$:

```python
import numpy as np
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d, solver_3d_nk
from lemaitre.initial_data.conformally_flat_puncture.parametric import quasicircular

prob = solver_3d.make_problem(44, 40, 8)          # 45 x 40 x 8 nodes

b, m = 5.0, 0.5                                   # punctures at z = +-b, bare masses m
S = (0.0, 0.5 * m**2, 0.0)                        # spin 0.5 m^2 along the orbital axis y
P_A, P_B = quasicircular.qc_momenta(b, m, m, S, S)
sl = solver_3d.Slice3D(b=b, m_A=m, m_B=m, P_A_vec=P_A, P_B_vec=P_B,
                       S_A_vec=S, S_B_vec=S)

U, info = solver_3d_nk.newton_solve_nk(prob, sl)  # cold start from u = 0
print(U.shape, info.converged)
print(["%.1e" % r for r in info.history], info.gmres_iters)

u = solver_3d.evaluate_field(prob, U, np.array([1.0]), np.array([2.0]),
                             np.array([0.0]), b)
print("u(rho=1, z=2, phi=0) = %.10f" % u[0])
```

```text
(45, 40, 8) True
['7.0e-03', '1.4e-04', '3.9e-08', '2.7e-13'] [3, 3, 3]
u(rho=1, z=2, phi=0) = 0.0099423403
```

The residual falls from $7 \times 10^{-3}$ to $3 \times 10^{-13}$ in three Newton
steps, each with three GMRES iterations, and the number of correct digits roughly
doubles at every step. On a laptop (Apple M2) the first call takes about 0.6 s,
mostly for building the preconditioner of the grid. Every later solve on the same
grid takes 0.04 to 0.06 s, also from $u = 0$, for separations between $b = 3$ and $8$
and aligned spins up to $0.7$.

The last lines evaluate the solution away from the nodes. `evaluate_field` inverts
the chart, transforms the node values to Fourier modes, and evaluates each mode with
barycentric interpolation in $A$ and $s$, so the field is available at any point an
evolution code asks for.

## References

1. M. Ansorg, B. Brügmann and W. Tichy, *Single-domain spectral method for black hole
   puncture data*, Phys. Rev. D **70**, 064011 (2004),
   [doi:10.1103/PhysRevD.70.064011](https://doi.org/10.1103/PhysRevD.70.064011).
2. S. Brandt and B. Brügmann, *A simple construction of initial data for multiple
   black holes*, Phys. Rev. Lett. **78**, 3606–3609 (1997),
   [doi:10.1103/PhysRevLett.78.3606](https://doi.org/10.1103/PhysRevLett.78.3606).
3. J. M. Bowen and J. W. York, Jr., *Time-asymmetric initial data for black holes and
   black-hole collisions*, Phys. Rev. D **21**, 2047–2056 (1980),
   [doi:10.1103/PhysRevD.21.2047](https://doi.org/10.1103/PhysRevD.21.2047).
4. G. B. Cook, *Initial data for numerical relativity*, Living Rev. Relativ. **3**, 5
   (2000), [doi:10.12942/lrr-2000-5](https://doi.org/10.12942/lrr-2000-5).
5. D. A. Knoll and D. E. Keyes, *Jacobian-free Newton–Krylov methods: a survey of
   approaches and applications*, J. Comput. Phys. **193**, 357–397 (2004).
6. Y. Saad and M. H. Schultz, *GMRES: a generalized minimal residual algorithm for
   solving nonsymmetric linear systems*, SIAM J. Sci. Stat. Comput. **7**, 856–869
   (1986).
7. R. E. Lynch, J. R. Rice and D. H. Thomas, *Direct solution of partial difference
   equations by tensor product methods*, Numer. Math. **6**, 185–199 (1964),
   [doi:10.1007/BF01386067](https://doi.org/10.1007/BF01386067).
8. F. De Ceuster, *A differentiable parametric model of binary-black-hole initial
   data: I. Conformally flat Bowen–York punctures*,
   [arXiv:2609.10173](https://arxiv.org/abs/2609.10173), section *Constraint solver*
   and the appendix *Details of the constraint solver*.
