One of the most common test case in supervised machine learning is that of classification:

> Given a data point, classify it into one of the many labels available

The simplest case of this is binary classification. For e.g. classify emails as spam or ham,
classify objects as star or galaxy, classify images as a cat or a dog, etc. In all these cases, to be able to apply the
mathematical modeling of machine learning one needs to *abstract* away some *feature(s)* of the object. Most often choosing or engineering the right features is what gives more power to predictive modeling rather than sophisticated models.

<blockquote>
If your features are crappy, no amount of sophisticated models can come to the rescue.
</blockquote>

However, in this post we are going to assume that we already have the features in hand. Given this, there are broadly two different ways of approaching classification problems - parametric and non-parametric. The topic we are going to discuss falls in the non-parametric approaches; more specifically as a linear model. It creates a linear boundary separating objects of one class from the other given as:


$$y = sign(x'^T w' + w_0) = x^T w \qquad y \in \{-1, 1\}.$$

Conceptually, one can also think of the boundary as a set of points whose probability of belonging to class $y=1$ (or $y=-1$) is half. As we move away from the boundary, on one side the probability of belonging to $y=1$ diminishes, while on the other end it approaches 1.

<p align="center">
  <img src="/static/img/log_reg_eg.png" width="400"/>
</p>

<!-- $$P(y=1) + P(y=-1) = 1$$

$$\mathrm{Odds} = \frac{P(y=1)}{P(y=-1)} = \frac{P(y=1)}{1 - P(y=1)}$$ -->

> Mathematically, what we want is a *specific* function of the perpendicular distance of any point from the boundary given by $d_\perp = x^T w$, whose value on the boundary be 0.5 and should range from 0 to 1 as we move from one side of the boundary to the other.

Given these properties, the value of this function for any point can be interpreted as its probability of belonging to class $y=1$. There are many such functions that we could choose as shown below.

<p align="center">
  <img src="/static/img/link_function.png" width="600"/>
</p>

The functional form used by Logistic regression is:

$$\mathrm{Prob}[x \in (y=1)] = f(d_\perp) = \frac{e^{d_\perp}}{1 + e^{d_\perp}} = \frac{e^{x^T w}}{1 + e^{x^T w}}$$

Our goal in now to estimate the value of the parameters of the model, $w$, from training data. However, to assess which values of the parameters are *good*, we have to first define the likelihood. Under the assumption that each data-point is independent of every other data-point, we have:

$$L = \prod_{i=1}^n p(y_i | x_i, w) = \prod_{i=1}^n [\mathcal{I}(y_i = 1) f(d_\perp^i)][\mathcal{I}(y_i = -1) 1 - f(d_\perp^i)] $$

Remember, $f(d_\perp)$ gives the probability of belonging to class $y=1$. So if a data-point belongs to $y=-1$ we need to use $1 - f(d_\perp)$.

$$\begin{align*} \mathcal{L} = \prod_{i=1}^n p(y_i | x_i, w) &= \prod_{i=1}^n \left[\mathcal{I}(y_i = 1) \frac{e^{x_i^T w}}{1 + e^{x_i^T w}} \right] \left[\mathcal{I}(y_i = -1) \left(1 - \frac{e^{x_i^T w}}{1 + e^{x_i^T w}} \right) \right] \\ &= \prod_{i=1}^n \frac{e^{y_i x_i^T w}}{1 + e^{y_i x_i^T w}}\end{align*}$$

Writing in terms of log-likelihood this becomes:

$$ \begin{align*} \log\ \mathcal{L} &= \sum_{i=1}^n \log [e^{y_i x_i^T w}] - \sum_{i=1}^n \log [1 + e^{y_i x_i^T w}] \\
&= \sum_{i=1}^n y_i x_i^T w - \sum_{i=1}^n \log [1 + e^{y_i x_i^T w}] \end{align*}$$
