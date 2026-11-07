# The Bernstein ellipse

The parametric model in this package samples the puncture solution at Chebyshev
nodes in each of the binary's parameters, the separation $b$, the mass ratio $q$
and the spins, and interpolates between them. How many nodes an axis needs, and
why some axes need far more than others, is decided by one classical result of
approximation theory, Bernstein's ellipse theorem [1, 2]. Spectral
methods are familiar in numerical relativity, from the single-domain puncture
solver TwoPunctures [6] to the codes reviewed in Ref. [7], but the theorem that
sets their convergence rate is rarely spelled out. This page introduces it, with
a figure to play with.

## Chebyshev interpolation

A parameter range $[\theta_{\min}, \theta_{\max}]$ is first mapped affinely onto
the interval $[-1, 1]$,

$$
\xi \ = \ \frac{2\theta - \theta_{\max} - \theta_{\min}}{\theta_{\max} - \theta_{\min}} .
$$

A function $f(\xi)$ is sampled at the $Q$ Chebyshev points

$$
\xi_j \ = \ \cos\frac{j\pi}{Q-1}, \qquad j = 0, \dots, Q-1,
$$

which crowd towards the two ends of the interval, and $p_Q$ is the polynomial of
degree $Q-1$ through those $Q$ samples. Interpolating at equally spaced points
instead can diverge as $Q$ grows, even for a perfectly smooth function, which is
the Runge phenomenon [4]. At Chebyshev points the interpolant converges for any
reasonably smooth $f$, and the useful question is how fast.

## The ellipses

For every $\rho > 1$ the *Bernstein ellipse* $E_\rho$ is the image of the circle
$|z| = \rho$ under the Joukowski map

$$
\xi \ = \ \tfrac12 \left( z + z^{-1} \right) .
$$

It has its foci at $\pm 1$, a semi-major axis $\tfrac12(\rho + \rho^{-1})$ along
the real axis and a semi-minor axis $\tfrac12(\rho - \rho^{-1})$ along the
imaginary axis, so $\rho$ is the sum of the two semi-axes. As $\rho \to 1$ the
ellipse collapses onto the interval itself, and for large $\rho$ it approaches a
circle of radius $\rho/2$.

The reason for this shape is that Chebyshev expansion is Fourier expansion in
disguise. With $\xi = \cos\vartheta$ and $z = e^{i\vartheta}$ the Chebyshev
polynomials become $T_k(\xi) = \tfrac12 (z^k + z^{-k})$, so a Chebyshev series in
$\xi$ is a Laurent series in $z$. A Laurent series converges in an annulus
$\rho^{-1} < |z| < \rho$ whose outer radius is set by the nearest singularity, its
coefficients then fall off like $\rho^{-k}$, and the Joukowski map carries that
annulus onto the inside of $E_\rho$.

## The theorem

Let $f$ be analytic on $[-1, 1]$ and analytically continuable to the open
ellipse $E_\rho$, where it satisfies $|f| \le M$. Then its Chebyshev coefficients
obey $|a_k| \le 2 M \rho^{-k}$, and its interpolant in $n+1$ Chebyshev points
obeys ([2], Theorems 8.1 and 8.2)

$$
\left\| f - p_{n+1} \right\|_\infty \ \le \ \frac{4 M \rho^{-n}}{\rho - 1} .
$$

The error therefore falls geometrically, like $\rho^{-Q}$ in the number of nodes,
and each extra node buys $\log_{10}\rho$ decimal digits. The bound holds for every
ellipse inside which $f$ is analytic, so the best rate comes from the largest such
ellipse, the one that passes through the singularity of $f$ nearest to the
interval. That rate is also sharp. Bernstein showed that no sequence of
polynomials of degree $n$ can approximate $f$ asymptotically faster than
$\rho^{-n}$ for that largest $\rho$ [1], and Chebyshev interpolants attain it [2, Ch. 8]. The rate
is a property of the function, not of the method.

A singularity at $\xi_\ast$ lies on the ellipse with

$$
\rho \ = \ \left| \xi_\ast + \sqrt{\xi_\ast^2 - 1} \right| ,
$$

taking the branch of the square root that gives $\rho > 1$. Two examples set the
scale.

- Runge's function $1/(1 + 25\xi^2)$ has poles at $\xi_\ast = \pm i/5$, so
  $\rho = (1 + \sqrt{26})/5 \approx 1.220$, which is $0.086$ decades per node, and
  about $185$ nodes reach double precision.
- A pole on the real axis at $\xi_\ast = -1.5$ gives
  $\rho = (3 + \sqrt5)/2 \approx 2.618$, which is $0.418$ decades per node, and
  about $40$ nodes reach double precision.

The distance from the interval alone does not decide the rate. A singularity at
distance $0.2$ from the middle of the interval ($\xi_\ast = 0.2i$) gives
$\rho \approx 1.220$, while one at the same distance beyond an end
($\xi_\ast = 1.2$) gives $\rho \approx 1.863$, more than three times the rate.
The nodes crowd towards the ends, so the ends tolerate a nearby singularity much
better than the middle does.

## Try it

```{raw} html
<div class="lm-bern"></div>
```

The function in the figure is $f(\xi) = 1/\left((\xi - a)^2 + b^2\right)$, which
is analytic everywhere except at the conjugate pair of poles
$\xi_\ast = a \pm i b$, the red crosses. It starts as Runge's function. Drag a
cross in the complex plane or use the sliders to move it, and change the number of
nodes $Q$.

- **The complex plane.** The thick black segment is the interval
  $[-1, 1]$ with the $Q$ nodes on it, and the thin grey curves are Bernstein
  ellipses at fixed $\rho$. The blue ellipse is the largest one free of
  singularities, the one through the poles, and its $\rho$ and rate are printed
  in the corner. A lighter ellipse grows out of the interval until it meets the
  poles, which is the construction in the theorem. The hollow circles mark where
  $E_\rho$ crosses the real axis, at $\pm\tfrac12(\rho + \rho^{-1})$.
- **The error.** The dots are the largest relative error of $p_Q$
  on the interval, measured in the browser, for every $Q$ from 2 to 60. The
  dashed line has the slope $\rho^{-Q}$ that the theorem predicts. The theorem
  fixes only the slope, so the line is shifted onto the dots. The red ring marks
  the current $Q$.
- **The function.** $f$ in black and its interpolant $p_Q$ in blue, with
  the nodes.

Things to try:

- Slide $\operatorname{Im}\xi_\ast$ down towards the interval and watch the
  ellipse shrink and the error line flatten.
- Put the pole on the real axis beyond an end, at $\operatorname{Im}\xi_\ast = 0$
  and $\operatorname{Re}\xi_\ast = -1.5$, then slide it towards $-1$. The rate
  collapses as the pole reaches the end of the interval.
- When the error reaches about $10^{-16}$ it stops falling. That is the rounding
  floor of double precision, not a failure of the theorem.
- For Runge's function the dots alternate between odd and even $Q$. The function
  is even, and an odd number of nodes puts one at its peak, $\xi = 0$. The slope
  is the same for both.

## What this means for the parametric model

Every parameter axis has a complex plane of its own, and the rate along that axis
is $\log_{10}\rho$ for the nearest singularity in that plane. Some singularities
are physical, such as coincident punctures at $b = 0$. Others sit off the real
axis, where nothing physical happens at all, and yet they limit the rate just as
much. In several dimensions the same argument applies axis by axis, which is why
the tensor and sparse grids can spend different numbers of nodes on different
parameters.

The theorem can also be read backwards. Fitting the measured error against $Q$
gives a rate $\zeta = \log_{10}\rho$ along one axis. Then $\rho = 10^{\zeta}$ fixes
an ellipse, and its real-axis crossing at $|\xi_\ast| = \tfrac12(\rho + \rho^{-1})$,
a hollow circle in the figure, is mapped back through the affine map on the
physically singular side of the box (small $b$, large $q$, large $|\chi|$) to give
an inferred parameter value $\theta_\ast$.

The fit measures only $\rho$, so it always returns a *real* $\theta_\ast$, even
when the true singularities are a complex-conjugate pair with nothing singular on
the real axis. The returned point is the real point on the same ellipse as the
pair. You can see this in the figure: put the poles off the real axis and the
hollow circles still sit on it, on the blue ellipse. Changing the sampling range
tells the two cases apart. A real branch point stays at the same $\theta_\ast$
however the range is chosen. A complex pair sits at a fixed place in the
parameter plane too, but each range maps it to a different point in $\xi$, on a
different ellipse, whose real point maps back to a different $\theta_\ast$. In the
paper's sweeps that $\theta_\ast$ marches outward as the range grows, staying
roughly in proportion to it. The accompanying paper [8] uses this test to classify the
limits of the quasi-circular family in its section *Convergence and its
analyticity limits*. There, the separation axis is limited by a real singularity
at $b = 0$, while the mass-ratio and spin axes behave like complex-conjugate
pairs.

The fitted rate is only as good as the window it is fitted over.

- **The window ends at the floor.** Once the error saturates at the float64
  rounding floor, extra nodes buy nothing, and including those points flattens
  the fit. The window must stop where the saturation starts.
- **The rate is asymptotic.** The theorem bounds the error for large $Q$. At small
  $Q$ the error can fall faster or slower than $\rho^{-Q}$, and other floors, such
  as the spatial resolution of each solve, can enter before the float64 one. Near
  extremal spin both effects mix into the fit, so the paper treats its spin-wall
  characterization as quantitative only for $\chi_{\max} \lesssim 0.8$.

The theorem rests on assumptions that are worth keeping in view.

- **Analyticity.** The geometric rate needs $f$ to be analytic in a neighbourhood
  of the interval. A function with a kink or a jump in a derivative inside the
  interval converges only algebraically, at a rate set by its smoothness [2, Ch. 7].
- **Chebyshev points.** The bound is for interpolation at Chebyshev points.
  Equally spaced nodes give a much worse rate and can diverge [4].
- **Floating point.** In double precision nothing converges below about $10^{-16}$
  relative to the size of $f$. The interpolant in the figure is evaluated with the
  barycentric formula, which is numerically stable [5].

You can check the two examples above with NumPy, whose `chebinterpolate` uses
the Chebyshev points of the first kind. The rate is the same for both kinds.

```python
import numpy as np
from numpy.polynomial import chebyshev as C

def rho_of(xi):                        # Bernstein parameter of the ellipse through xi
    s = np.sqrt(xi * xi - 1 + 0j)
    return max(abs(xi + s), abs(xi - s))

x = np.linspace(-1, 1, 4001)
for f, xi in [(lambda t: 1 / (1 + 25 * t * t), 0.2j), (lambda t: 1 / (t + 1.5), -1.5)]:
    Q = np.arange(4, 61, 4)
    err = [np.max(np.abs(C.chebval(x, C.chebinterpolate(f, q - 1)) - f(x))) / np.max(np.abs(f(x))) for q in Q]
    ok = np.array(err) > 1e-13          # stay above the rounding floor
    rate = -np.polyfit(Q[ok], np.log10(np.array(err)[ok]), 1)[0]
    print(f"predicted {np.log10(rho_of(xi)):.4f}, measured {rate:.4f} decades per node")
```

It prints a predicted rate of 0.0863 against a measured 0.0857 for Runge's
function, and 0.4180 against 0.4180 for the real pole.

## References

1. S. N. Bernstein, *Sur l'ordre de la meilleure approximation des fonctions
   continues par des polynômes de degré donné*, Mém. Acad. Roy. Belg. (2) **4**,
   1–103 (1912).
2. L. N. Trefethen, *Approximation Theory and Approximation Practice*, Extended
   Edition (SIAM, Philadelphia, 2019), Chapter 8, Theorems 8.1 and 8.2,
   [doi:10.1137/1.9781611975949](https://doi.org/10.1137/1.9781611975949).
3. J. P. Boyd, *Chebyshev and Fourier Spectral Methods*, 2nd edition (Dover,
   Mineola, 2001), Chapter 2.
4. C. Runge, *Über empirische Funktionen und die Interpolation zwischen
   äquidistanten Ordinaten*, Z. Math. Phys. **46**, 224–243 (1901).
5. J.-P. Berrut and L. N. Trefethen, *Barycentric Lagrange interpolation*,
   SIAM Rev. **46**, 501–517 (2004),
   [doi:10.1137/S0036144502417715](https://doi.org/10.1137/S0036144502417715).
6. M. Ansorg, B. Brügmann and W. Tichy, *A single-domain spectral method for
   black hole puncture data*, Phys. Rev. D **70**, 064011 (2004),
   [arXiv:gr-qc/0404056](https://arxiv.org/abs/gr-qc/0404056).
7. P. Grandclément and J. Novak, *Spectral methods for numerical relativity*,
   Living Rev. Relativ. **12**, 1 (2009),
   [arXiv:0706.2286](https://arxiv.org/abs/0706.2286).
8. F. De Ceuster, *A differentiable parametric model of binary-black-hole initial
   data: I. Conformally flat Bowen–York punctures*,
   [arXiv:2609.10173](https://arxiv.org/abs/2609.10173).
