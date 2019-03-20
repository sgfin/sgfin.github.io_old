---
layout: post
comments: true
include: true
title:  "Deriving the information entropy of the multivariate gaussian"
excerpt: "Derivation of the gaussian's entropy, in part to explain a trace trick."
date:   2017-03-11 10:00:00
mathjax: true
---

* TOC
{:toc}

## Introduction and Trace Tricks

The pdf of a [multivariate gaussian](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) is as follows:

$$p(x) = \frac{1}{(\sqrt{2\pi})^{N}\sqrt{\det\Sigma}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}$$

, where

$$\Sigma_{i,j} = E[(x_{i} - \mu_{i})(x_{j} - \mu_{j})]$$

is the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix), which can be expressed in vector notation as

$$\Sigma = E[(X-E[X])(X-E[X])^{T}] = \int p(x)(x-\mu)(x-\mu)^{T}dx$$

. I might make the derivation of this formula its own post at some point, but it is in Strang's intro to linear algebra text so I will hold off.  Instead, this post derives the *entropy* of the multivariate gaussian, which is equal to:

$$H=\frac{N}{2}\ln\left(2\pi e\right)+\frac{1}{2}\ln\det C$$

Part of the reason why I do this is because the second part of the derivation involves a "trace trick" that I want to remember how to use for the future.  The key to the "trace trick" is to recognize that a matrix (slash set of multiplied matrices) is 1x1, and that the value of any such matrix is, by definition, equal to its trace.  This then allows you to invoke the quasi-commutative property of the trace:

$$\text{tr}(UVW)=\text{tr}(WUV)$$

to push around the matrices however you desire until they become something tidy/useful.  The whole thing feels rather devious to me, personally.

## Derivation
### Setup 

Beginning with the definition of entropy

$$H(x)=-\int p(x)*\ln p(x)dx$$

substituting in the probability function for the multivariate gaussian
in only its second occurence in the formula,

$$H(x)=-\int p(x)*\ln\left(\frac{1}{(\sqrt{2\pi})^{N}\sqrt{\det\Sigma}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}\right)dx$$

$$=-\int p(x)*\ln\left(\frac{1}{(\sqrt{2\pi})^{N}\sqrt{\det\Sigma}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}\right)dx$$

$$=-\int p(x)*\ln\left(\frac{1}{(\sqrt{2\pi})^{N}\sqrt{\det\Sigma}}\right)dx-\int p(x)\ln\left(e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}\right)dx$$

We will now consider these two terms separately.

### First term

First, we concern ourselves with the first ln term:

$$-\int p(x)*\ln\left(\frac{1}{(\sqrt{2\pi})^{N}\sqrt{\det C}}\right)dx$$

$$=\int p(x)*\ln\left((\sqrt{2\pi})^{N}\sqrt{\det C}\right)$$

since all the terms other than $$p(x)$$ form a constant,

$$=\left(\ln\left((\sqrt{2\pi})^{N}\sqrt{\det C}\right)\right)\int p(x)$$

and because $$p(x)$$ is a PDF, it integrates to 1. Thus, this component of
the equation is

$$\ln\left((\sqrt{2\pi})^{N}\sqrt{\det C}\right)$$

$$=\frac{N}{2}\ln\left(2\pi\right)+\frac{1}{2}\ln\det C$$

### Second term  (Trace Trick Coming!)

Now we consider the second ln term

$$-\int p(x)\ln\left(e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}\right)dx$$

$$=\int p(x)\frac{1}{2}\ln\left(e^{(x-\mu)^{T}\Sigma^{-1}(x-\mu)}\right)dx$$

because $$(x-\mu)^{T}$$ is a 1 x N matrix, $$\Sigma^{-1}$$ is a N x N
matrix, and $$(x-\mu)$$ is a N x 1 matrix, the matrix product
$$(x-\mu)^{T}\Sigma^{-1}(x-\mu)$$ is a 1 x 1 matrix. Further, because the
trace of any 1 x 1 matrix
$$\text{tr}(A)=\Sigma_{i=1}^{n}A_{i,i}=A_{1,1}=A$$, we can conclude that
the 1 x 1 matrix
$$(x-\mu)^{T}\Sigma^{-1}(x-\mu)=\text{tr}((x-\mu)^{T}\Sigma^{-1}(x-\mu))$$.

As such, our term becomes

$$=\int p(x)\frac{1}{2}\ln\left(e^{\text{tr}\left[(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right]}\right)dx$$

$$=\frac{1}{2}\int p(x)\ln\left(e^{\text{tr}\left[(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right]}\right)dx$$

, which, by the quasi-commutativity property of the trace function,
$$\text{tr}(UVW)=\text{tr}(WUV)$$,

$$=\frac{1}{2}\int p(x)\ln\left(e^{\text{tr}\left[\Sigma^{-1}(x-\mu)(x-\mu)^{T}\right]}\right)dx$$

. Because $$p(x)$$ is a scalar and the natural logarithm and exponentials
may cancel, the properties of the trace function allow us to push the
$$p(x)$$ and the integral inside of the trace, so

$$=\frac{1}{2}\int\ln\left(e^{\text{tr}\left[\Sigma^{-1}p(x)(x-\mu)(x-\mu)^{T}\right]}\right)dx$$

$$=\frac{1}{2}\ln\left(e^{\text{tr}\left[\Sigma^{-1}\int p(x)(x-\mu)(x-\mu)^{T}dx\right]}\right)$$

But, $\int p(x)(x-\mu)(x-\mu)^{T}dx=\Sigma$ is just the definition of the covariance matrix!  As such,

$$=\frac{1}{2}\ln\left(e^{\text{tr}\left[\Sigma^{-1}\Sigma\right]}\right)$$

$$=\frac{1}{2}\ln\left(e^{\text{tr}\left[I_{N}\right]}\right)$$

$$=\frac{1}{2}\ln\left(e^{N}\right)$$

$$=\frac{N}{2}\ln\left(e\right)$$

### Recombining the terms

Bringing the above terms back together, we have

$$H(x)=\frac{N}{2}\ln\left(2\pi\right)+\frac{1}{2}\ln\det C+\frac{N}{2}\ln\left(e\right)$$

$$=\frac{N}{2}\ln\left(2\pi e\right)+\frac{1}{2}\ln\det C$$

as desired.
