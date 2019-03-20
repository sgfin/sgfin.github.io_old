---
layout: post
comments: true
include: true
title:  "Deriving probability distributions using the Principle of Maximum Entropy"
excerpt: "Uniform, gaussian, exponential, and another distribution all from first principles."
date:   2017-03-16 10:00:00
mathjax: true
---

* TOC
{:toc}

## Introduction

In this post, I derive the uniform, gaussian, exponential, and another funky probability distribution from the first principles of information theory. I originally did it for a class, but I enjoyed it and learned a lot so I am adding it here so I don't forget about it.

I actually think it's pretty magical that these common distributions just pop out when you are using the information framework.  It feels so much more satisfying/intuitive than it did before.

### Maximum Entropy Principle

Recall that [information entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) is a mathematical framework for quantifying "uncertainty."  The formula for the information entropy of a random variable is
$$H(x) = \int p(x)\ln p(x)dx $$
.  	
In statistics/information theory, the [maximum entropy probability distribution](https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution) is (you guessed it!) the distribution that, given any constraints, has maximum entropy.  Given a choice of distributions, the "Principle of Maximum Entropy" tells us that the maximum entropy distribution is the best.  Here's a snippet of the idea from the [wikipedia page](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy):
>The principle of maximum entropy states that, subject to precisely stated prior data (such as a proposition that expresses testable information), the probability distribution which best represents the current state of knowledge is the one with largest entropy.

> Another way of stating this: Take precisely stated prior data or testable information about a probability distribution function. Consider the set of all trial probability distributions that would encode the prior data. According to this principle, the distribution with maximal information entropy is the proper one.

>...

>In ordinary language, the principle of maximum entropy can be said to express a claim of epistemic modesty, or of maximum ignorance. The selected distribution is the one that makes the least claim to being informed beyond the stated prior data, that is to say the one that admits the most ignorance beyond the stated prior data.

### Lagrange Multipliers

Given the above, we can use the maximum entropy principle to derive the best probability distribution for a given use.  A useful tool in doing so is the Lagrange Multiplier \([Khan Acad article](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/constrained-optimization/a/lagrange-multipliers-single-constraint), [wikipedia](https://en.wikipedia.org/wiki/Lagrange_multiplier)\), which helps us maximize or minimize a function under a given set of constraints.

For a single variable function $$f(x)$$ subject to the constraint $$g(x) = c$$, the lagrangian is of the form:
$$\mathcal{L}(x,\lambda) = f(x) - \lambda(g(x)- c)$$
, which is then differentiated and set to zero to find a solution.

The above can then be extended to additional variables and constraints as:

$$\mathcal{L}(x_{1}\dots x_{n},\lambda_{1}\dots\lambda{n}) = f(x_{1}\dots x_{n}) - \Sigma_{k=1}^{M}\lambda_{k}g_{k}(x_{1}\dots x_{n})$$

and solving

$$\nabla x_{1},\dots,x_{n},\lambda_{1}\dots \lambda_{M}\mathcal{L}(x_{1}\dots x_{n},\lambda_{1}\dots\lambda{n})=0$$

or, equivalently, solving

$$\begin{cases}
\nabla f(x)-\Sigma_{K=1}^{M}\lambda_{k}\nabla g_{k}(x)=0\\
g_{1}(x)=\dots=g_{M}(x)=0
\end{cases}$$

In this case, since we are deriving probability distributions, the integral of the pdf must sum to one, and as such, every derivation will include the constraint $$(\int p(x)dx-1)=0$$.

With all that, we can begin:

## 1. Derivation of maximum entropy probability distribution with no other constraints (uniform distribution)

First, we solve for the case where the only constraint is that the distribution is a pdf, which we will see is the uniform distribution. To maximize entropy, we want to minimize the following function:

$$J(p)=\int_{a}^{b} p(x)\ln p(x)dx-\lambda_{0}\left(\int_{a}^{b} p(x)dx-1\right)$$

.  Taking the derivative with respect ot $$p(x)$$ and setting to zero,

$$\frac{\delta J}{\delta p(x)}=1+\ln p(x)-\lambda_{0}=0$$

$$\ln p(x)=1-\lambda_{0}$$

$$p(x)=e^{1 -\lambda_{0}}$$

, which in turn must satisfy

$$\int_{a}^{b} p(x)dx=1=\int_{a}^{b} e^{-\lambda_{0}+1}dx$$

Note: To check if this is a minimum (which would maximize entropy given the
way the equation was set up), we also need to see if the second
derivative with respect to $$p(x)$$ is positive here or not, which it
clearly always is:

$$\frac{\delta J}{\delta p(x)^{2}dx}=\frac{1}{p(x)}$$


### Satisfy  constraint

$$\int_{a}^{b} p(x)dx=\int_{a}^{b} e^{1 -\lambda_{0}}dx=1$$

$$\int_{a}^{b} e^{-\lambda_{0}+1}dx=1$$

$$e^{-\lambda_{0}+1} \int_{a}^{b} dx=1$$

$$ e^{-\lambda_{0}+1} (b-a) = 1$$

$$e^{-\lambda_{0}+1} = \frac{1}{b-a}$$

$$-\lambda_{0}+1 = \ln\frac{1}{b-a}$$

$$\lambda_{0} = 1 -\ln \frac{1}{b-a}$$


### Putting Together

Plugging the constraint $$\lambda_{0} = 1 -\ln \frac{1}{b-a}$$ into the pdf $$p(x)=e^{1 -\lambda_{0}}$$, we have:

$$p(x)=e^{1 -\lambda_{0}}$$

$$p(x)=e^{1 -(1 -\ln \frac{1}{b-a})}$$

$$p(x)=e^{1 -1 + \ln \frac{1}{b-a}}$$

$$p(x)=e^{\ln \frac{1}{b-a}}$$

$$p(x)=\frac{1}{b-a}$$

.  Of course, this is only defined in the range between $$a$$ and $$b$$, however, so the final function is:

$$p(x)=\begin{cases}
\frac{1}{b-a} & a\leq x \leq b\\
0 & \text{otherwise}
\end{cases}$$

## 2. Derivation of maximum entropy probability distribution for given fixed mean $$\mu$$ and variance $$\sigma^{2}$$ (gaussian distribution)

Now, for the case when we have a specified mean and variance, which we will see is the gaussian distribution.  To maximize entropy, we want to minimize the following function:

$$J(p)=\int p(x)\ln p(x)dx-\lambda_{0}\left(\int p(x)dx-1\right)-\lambda_{1}\left(\int p(x)(x-\mu)^{2}dx-\sigma^{2}\right)$$

, where the first constraint is the definition of pdf and the second is the definition of the variance (which also gives us the mean for free).  Taking the derivative with respect ot p(x) and setting to zero,

$$\frac{\delta J}{\delta p(x)}=1+\ln p(x)-\lambda_{0}-\lambda_{1}(x-\mu)^{2}=0$$

$$\ln p(x)=1-\lambda_{0}-\lambda_{1}(x-\mu)^{2}$$

$$p(x)=e^{-\lambda_{0}+1-\lambda_{1}(x-\mu)^{2}}$$

, which in turn must satisfy

$$\int p(x)dx=1=\int e^{-\lambda_{0}+1-\lambda_{1}(x-\mu)^{2}}dx$$

and

$$\int p(x)(x-\mu)^{2}dx=\sigma^{2}=\int e^{-\lambda_{0}+1-\lambda_{1}(x-\mu)^{2}}(x-\mu)^{2}dx$$

Again, $$\frac{\delta J}{\delta p(x)^{2}dx}=\frac{1}{p(x)}$$ is always positive, so our solution will be minimum.

### Satisfy first constraint

$$\int p(x)dx=1=\int e^{-\lambda_{0}+1-\lambda_{1}(x-\mu)^{2}}dx$$

$$1=\int e^{-\lambda_{0}+1-\lambda_{1}z^{2}}dz$$

$$1=\int e^{-\lambda_{0}+1-\lambda_{1}z^{2}}dz$$

$$1=\int e^{-\lambda_{0}+1}*e^{-\lambda_{1}z^{2}}dz$$
$$1=e^{-\lambda_{0}+1}\int e^{-\lambda_{1}z^{2}}dz$$
$$e^{\lambda_{0}-1}=\int e^{-\lambda_{1}z^{2}}dz$$
$$e^{\lambda_{0}-1}=\int e^{-\lambda_{1}z^{2}}dz$$
$$e^{\lambda_{0}-1}=\sqrt{\frac{\pi}{\lambda_{1}}}$$

### Satisfy second constraint

$$\int p(x)(x-\mu)^{2}dx=\sigma^{2}=\int e^{-\lambda_{0}+1-\lambda_{1}(x-\mu)^{2}}(x-\mu)^{2}dx$$

$$\sigma^{2}=\int e^{-\lambda_{0}+1-\lambda_{1}(x-\mu)^{2}}(x-\mu)^{2}dx$$

$$\sigma^{2}=\int e^{-\lambda_{0}-1-\lambda_{1}z^{2}}z^{2}dz$$

$$\sigma^{2}e^{\lambda_{0}-1}=\int e^{-\lambda_{1}z^{2}}z^{2}dz$$

$$\sigma^{2}e^{\lambda_{0}-1}=\frac{1}{2}\sqrt{\frac{\pi}{\lambda_{1}^{3}}}$$

$$\sigma^{2}e^{\lambda_{0}-1}=\frac{1}{2\lambda_{1}}\sqrt{\frac{\pi}{\lambda_{1}}}$$

$$2\lambda_{1}\sigma^{2}e^{\lambda_{0}-1}=\sqrt{\frac{\pi}{\lambda_{1}}}$$

### Putting together

$$\sqrt{\frac{\pi}{\lambda_{1}}}=e^{\lambda_{0}-1}=2\lambda_{1}\sigma^{2}e^{\lambda_{0}-1}$$

so

$$e^{\lambda_{0}-1}=2\lambda_{1}\sigma^{2}e^{\lambda_{0}-1}$$

$$1=2\lambda_{1}\sigma^{2}$$

$$\lambda_{1}=\frac{1}{2\sigma^{2}}$$

. Plugging in for the other lambda,

$$\sqrt{\frac{\pi}{\lambda_{1}}}=e^{\lambda_{0}-1}$$

$$\sqrt{2\sigma^{2}\pi}=e^{\lambda_{0}-1}$$

$$\ln\sqrt{2\sigma^{2}\pi}=\lambda_{0}-1$$

$$\lambda_{0}=\ln\sqrt{2\sigma^{2}\pi}+1$$

Now, we plug back into the first equation

$$p(x)=e^{-\lambda_{0}-1-\lambda_{1}(x-\mu)^{2}}$$

$$=e^{-\ln\sqrt{2\sigma^{2}\pi}-\frac{1}{2\sigma^{2}}(x-\mu)^{2}}$$

$$=e^{-\ln\sqrt{2\sigma^{2}\pi}}e^{-\frac{1}{2\sigma^{2}}(x-\mu)^{2}}$$

$$=\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}}$$

which we can note is, by definition, the pdf of the Gaussian!

## 3. Derivation of maximum entropy probability distribution of half-bounded random variable with fixed mean $$\bar{r}$$ exponential distribution)

Now, constrain on a fixed mean, but no fixed variance, which we will see is the exponential distribution. To maximize entropy, we want to minimize the following function:

$$J(p)=\int p(x)\ln p(x)dx-\lambda_{0}\left(\int_{0}^{\infty}p(x)dx-1\right)-\lambda\left(\int_{0}^{\infty}x*p(x)dx-\bar{r}\right)$$

Now take derivative

$$\frac{\delta J}{\delta p(x)dx}=1+\ln p(x)-\lambda_{0}-\lambda_{1}x$$

To check if this is a minimum of the function, we need to see if the
second derivative is positive with respect to p(x), which it is:

$$\frac{\delta J}{\delta p(x)^{2}dx}=\frac{1}{p(x)}$$ Setting the first
derivative to zero, we have

$$0=1+\ln p(x)-\lambda_{0}-\lambda_{1}x$$

$$p(x)=e^{-\lambda_{0}+1+-\lambda x}$$

, which must satisfy the constaints $$\int_{0}^{\infty}p(x)dx=1$$ and
$$\int_{0}^{\infty}x*p(x)dx-\bar{r}$$.

### Satisfying first constraint 

$$\int_{0}^{\infty}p(x)dx=1$$

$$\int_{0}^{\infty}e^{-\lambda_{0}+1-\lambda_{1}x}dx=1$$

$$\int_{0}^{\infty}e^{-\lambda_{1}x}dx=e^{\lambda_{0}-1}$$

$$\frac{1}{\lambda_{1}}=e^{\lambda_{0}+1}$$

$$\lambda_{1}=e^{-\lambda_{0}+1}$$

### Satisfying the second constraint

$$\int_{0}^{\infty}x*e^{-\lambda_{0}+1-\lambda_{1}x}dx=\bar{r}$$

$$\int_{0}^{\infty}x*e^{-\lambda_{0}+1}e^{\lambda_{1}x}dx=\bar{r}$$

substituting in $$\lambda_{1}=e^{-\lambda_{0}+1}$$ from above

$$\int_{0}^{\infty}x*\lambda_{1}e^{\lambda_{1}x}dx=\bar{r}$$

### Putting together

Rather than evaluating this last integral above, we can simply stop and
note that in evaluating our constraints we have stumbled upon the
formula for an exponential random variable with parameter $$\lambda$$!

More explicitly:

$$\int_{0}^{\infty}x*\lambda_{1}e^{\lambda_{1}x}dx=\bar{r}$$

$$\int_{0}^{\infty}x*p(x)dx=\bar{r}$$

where $$p(x)=\lambda e^{\lambda x}$$, the pdf of the exponential function
for $$x\ge0$$, where $$\lambda=\frac{1}{\bar{r}}$$.

In other words,

$$p(x)=\begin{cases}
\frac{1}{\bar{r}}e^{-\frac{x}{\bar{r}}} & x\ge0\\
0 & x<0
\end{cases}$$

## 4. Maximum entropy of random variable over range $$R$$ with set of constraints $$\left\langle f_{n}(x)\right\rangle =\alpha_{n}$$ with $$n=1\dots N$$ and $$f_{n}$$ is of polynomial order

$$f_{n}$$ must be even order for all enforced constraints.

Following the same approach as above:

$$J(p)=-\int p(x)\ln p(x)dx+\lambda_{0}\left(\int p(x)dx-1\right)+\Sigma_{i=1}^{N}\lambda_{i}\left(p(x)f_{i}(x)dx-a_{i}\right)$$

$$\frac{\delta J}{\delta p(x)dx}=-1-\ln p(x)+\lambda_{0}+\Sigma_{i=1}^{N}\lambda_{i}f_{i}(x)$$

$$0=-1-\ln p(x)+\lambda_{0}+\Sigma_{i=1}^{N}\lambda_{i}f_{i}(x)$$

$$p(x)=e^{\lambda_{0}-1+\Sigma_{i=1}^{N}\lambda_{i}f_{i}(x)}$$

all where $$f_{i}(x)=\Sigma_{j=1}^{M}b_{j}x^{j}$$.

We now consider the conditions in which the random variable can be
defined in the entire domain $$(-\infty,\infty)$$. Looking at the
normalization constraint,

$$\int p(x)dx=\int e^{\lambda_{0}-1+\Sigma_{i=1}^{N}\lambda_{i}f_{i}(x)}dx=1$$

we note that we need our exponential function to integrate to 1. In
order for this equation to be defined in the entire real domain, we thus
will need the exponential function to integrate to a finite value, so
that we can provide a normalization constant that will result in
integration to 1.

Looking at the function
$$e^{\lambda_{0}-1+\Sigma_{i=1}^{N}\lambda_{i}f_{i}(x)}$$ (which must
remain finite for all x), we can thus conclude that
$$\lambda_{0}-1+\Sigma_{i=1}^{N}\lambda_{i}f_{i}(x)$$ must not converge to
positive infinity, but may converge to negative infinity (because it
would cause the exponential to converge to zero) or to any finite value
as $$x$$ approaches positive or negative infinity. The only components of
this function that depend on $$x$$ are the polynomail constraints of form
$$f_{i}(x)=\Sigma_{j=1}^{M}b_{j}x^{j}$$. As such, these constraints are
the only components at risk to force the function towards infinity,
provided that $$\lambda_{0}\neq\infty.$$ Therefore, because the
$$\lambda_{i}$$ corresponding to can any $$f_{i}$$ can be positive or
negative, the function will be able to be defined so long
$$f_{i}(x)=\Sigma_{j=1}^{M}b_{j}x^{j}<\infty$$ for all $$x$$, or
$$f_{i}(x)=\Sigma_{j=1}^{M}b_{j}x^{j}>-\infty$$ for all $$x.$$

Finally, we can consider the conditions for which these criteria for
$$f_{i}$$ will be satisfied. In short, the only way to guarantee that
$$f_{i}$$ remain either positive for negative will be if the dominant
component of the polynomial $$f_{i}$$ is of an EVEN order for all $$i$$ s.t.
$$\lambda_{i}\neq0$$. If the dominant component is odd, then $$f_{i}$$ will
either move from negative infinity to positive infinity (or, if negated,
from positive infinity to negative infinity) as x moves across the
domain, which means that no finite and nonzero $$\lambda_{i}$$ could be
chosen to maintain the criteria outlined above.
