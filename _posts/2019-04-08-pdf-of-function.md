---
layout: post
title:  "PDF of a dependent variable"
date:   2019-04-08 00:18:23 +0700
categories: [statistics]
---

Many-a-time when working with model fitting to data, one encounters a
situation where one need to find the probability distribution of a random
variable, which itself is a function of another random variable.

$y = f(x)$, if $x \sim f_X(x)$, then $y \sim ?$

Naively, one might think for a given value $y'$, one can just *pick up* its
probability as $f_X(x')$, where $y' = f(x')$. This is not correct. One should **understand** that the probability of a single point has no meaning, it is the probability of finding something in a **given interval** that is conserved across transformations. In other words -

$$f_X(x)\ dx = g_Y(y)\ dy$$

Let's look at a few examples:

1.
    LogNormal Distribution: If $x$ is Gaussian distributed, then $y = e^x$ is said to have lognormal distribution. Let's find its pdf using the above relation.

$$
\begin{align}
g_Y(y) &= f_X(x) (dx / dy) \\
       &= \frac{1}{2 \pi \sigma^2} e^{-\frac{(x -\mu)^2}{2 \sigma^2}} \frac{1}{y} \\
       &= \frac{1}{2 \pi y \sigma^2} e^{-\frac{(\ln y -\mu)^2}{2 \sigma^2}}
\end{align}
$$

![composite](/static/img/pdf-of-function.png){:class="img-responsive"}

2.
 Flux transmission:
The flux field we observe from the absorption of Lyman-alpha photons by neutral Hydrogen in the intergalactic medium is related to the density field which is Gaussian in nature through the relation:

$$F(\mathbf{x}) = \exp \left[-A \exp \left(\beta \left[\delta_g(\mathbf{x}) - \frac{\sigma_g^2}{2} \right] \right) \right]$$

The dependence of flux on density is thus highly non-linear. However, at earlier times when the variance on $\delta$ was small, the flux was roughly Gaussian distributed too. To find the exact pdf, we start by noting that:

$$\frac{dF}{d\delta_g} = \beta\ F \ln F$$

Thus we have,

$$p(F) = -\frac{p(\delta_g)}{\beta\ F\ ln F},$$

where $\delta_g$ for a given $F$ can be calculated from the first equation. We need a negative sign cause $F$ is a monotonically decreasing function of $\delta_g$.


