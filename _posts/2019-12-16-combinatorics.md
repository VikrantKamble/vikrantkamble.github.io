---
layout: post
title:  "Combinatorics, Tail sum for expectation and Simplices"
date:   2019-12-16 00:23:00 +0700
categories: [statistics]
---

**1. How many different ways are possible to have $k-tuple$ with *non-negative integer* values such that they sum to a given value $n$?**

$\to $To arrive at the solution, one can use the $stars-and-bars$ method. Let's draw a sequence of n stars. There are $n-1$ spaces between the stars. What we need to do now to create $k$ numbers, is to place $k-1$ bars in those spaces. One such possibility for $n=6$ and $k=3$ is as follows:

$$\star\ | \star\  \star\ \star\ | \star\ \star \quad \quad \to 3-tuple\ (1, 3, 2)$$

Thus we need to choose $k-1$ locations out of the $n-1$ spaces available. The number of such combinations possible is thus $\binom{n-1}{k-1}$.

**2. What if we allowed for the values to be 0? How many combinations are possible then?**

$\to\$ In the above case when all is done we placed $n+k-1$ objects(stars + bars). We were restricted by the fact that the values had to be *non-negative*. So we had to place the bars in the gaps between stars. However, now we do not have any such restriction. We can first choose $k-1$ slots out of $n+k-1$ slots available to place the bars. Once that is done, we just place the $n$ stars in the $n$ slots remaining. One such possibility for $n=6$ and $k=3$ is as follows:

$$\underset{\_}{|}\ \underset{\_}{\star}\ \underset{\_}{\star}\ \underset{\_}{\star}\ \underset{\_}{|}\ \underset{\_}{\star}\ \underset{\_}{\star}\ \underset{\_}{\star} \quad \quad \to 3-tuple\ (0, 3, 3)$$

Given this discussion, the number of combinations of $k-tuple$ possible is $\binom{n+k-1}{k-1} = \binom{n+k-1}{n}$. This number is also known as the [multiset coefficient](https://en.wikipedia.org/wiki/Multiset).

**3. We roll a fair dice $n$ times. What is the probability that all faces have appeared?**

Lets $\mathcal{I_i}$ be the number of times a face $i$ has appeared after the dice has been rolled $n$ times. What we need is a $6-tuple\ (\mathcal{I_1}, \mathcal{I_2}, \mathcal{I_3}, \mathcal{I_4}, \mathcal{I_5}, \mathcal{I_6})$ such that the elements sum to $n$, and **none** of the values $\mathcal{I_i}$ are 0. The probability that all faces have appeared is then the result given in Que. 1 divided by the results given in Que 2.

<p>
\begin{align}
prob = \binom{n-1}{k-1} / \binom{n+k-1}{k-1} &= \frac{n!\ (n-1)!}{(n-k)!\ (n+k-1)!}\\
  &= \frac{n!\ (n-1)!}{(n-6)!\ (n+5)!} \qquad \qquad k = 6
\end{align}
</p>

<hr>

The *tail sum for expectation* formula for a non-negative integer random number is given as:

$$E[X] = \sum_{x=0}^\infty x\ P(X = x) = \sum_{x=0}^\infty P(X > x)$$

Proof: To show this, one can use an interesting identity for any non-negative integer given by:

$$x = \sum_{k=0}^\infty \mathcal{I}(x > k),$$

where $\mathcal{I}(condition)$ is an indicator function that evaluates to $1$ if condition is true, else 0. The well known formula for expectation can then becomes:

$$E[X] = \sum_{x=0}^\infty x\ P(X = x) = \sum_{x=0}^\infty \sum_{k=0}^\infty  \mathcal{I}(x > k)\ P(X = x).$$

Switching the order of summation, we get the required result:

$$E[X] = \sum_{k=0}^\infty \sum_{x=0}^\infty \mathcal{I}(x > k)\ P(X = x) = \sum_{k=0}^\infty P(X > k).$$

