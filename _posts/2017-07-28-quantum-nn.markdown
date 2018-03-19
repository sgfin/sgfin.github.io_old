---
layout: post
comments: true
title:  "Taming wave functions with neural networks"
excerpt: "The wave function is essential to most calculations in quantum mechanics, and yet it's a difficult beast to tame. Can neural networks help?"
date:   2017-07-28 11:00:00
mathjax: true
---

_NOTE: This is a repost from [an article I wrote for **Quantum Frontiers**](https://quantumfrontiers.com/2017/08/02/taming-wave-functions-with-neural-networks/), the blog of the Institute for Quantum Information and Matter at Caltech_

<div class="imgcap">
    <img src="/assets/quantum-nn/wavf-ski.jpg" width="30%">
</div>

The wave function is essential to most calculations in quantum mechanics, and yet it's a difficult beast to tame. Can neural networks help?

## Wave functions in the wild

> "\\(\psi\\) is a monolithic mathematical quantity that contains all the information on a quantum state, be it a single particle or a complex molecule." -- Carleo and Troyer, [Science](http://science.sciencemag.org/content/355/6325/602.full)

The wave function, $$\psi$$ , is a mixed blessing. At first, it causes unsuspecting undergrads (me) some angst via the Schrodinger’s cat paradox. This angst morphs into full-fledged panic when they encounter concepts such as nonlocality and Bell’s theorem (which, by the way, is surprisingly hard to [verify experimentally](https://phys.org/news/2017-02-physicists-loophole-bell-inequality-year-old.html)). The real trouble with $$\psi$$, though, is that it grows exponentially with the number of entangled particles in a system. We couldn’t even hope to write the wavefunction of 100 entangled particles, much less perform computations on it…but there’s a lot to gain from doing just that.

The thing is, we (a couple of luckless physicists) love $$\psi$$ . Manipulating wave functions can give us [ultra-precise timekeeping](https://www.nature.com/news/2010/100331/full/news.2010.163.html), [secure encryption](http://physicsworld.com/cws/article/news/2017/jul/11/quantum-satellites-demonstrate-teleportation-and-encryption), and [polynomial-time factoring of integers](https://quantumfrontiers.com/2013/03/17/post-quantum-cryptography/) (read: break RSA). Harnessing quantum effects can also produce [better machine learning](https://www.technologyreview.com/s/544421/googles-quantum-dream-machine/), [better physics simulations](https://phys.org/news/2013-10-feynman-wasnt-quantum-dynamics-ground.html), and even [quantum teleportation](https://quantumfrontiers.com/2012/09/17/how-to-build-a-teleportation-machine-teleportation-protocol/).

## Taming the beast

Though $$\psi$$  grows exponentially with the number of particles in a system, most physical wave functions can be described with a lot less information. Two algorithms for doing this are the Density Matrix Renormalization Group (DMRG) and Quantum Monte Carlo (QMC).

<div class="imgcap">
    <img src="/assets/quantum-nn/bonsai.png" width="30%">
</div>

**Density Matrix Renormalization Group (DMRG).** Imagine we want to learn about trees, but studying a full-grown, 50-foot tall tree in the lab is too unwieldy. One idea is to keep the tree small, like a bonsai tree. DMRG is an algorithm which, like a bonsai gardener, prunes the wave function while preserving its most important components. It produces a compressed version of the wave function called a Matrix Product State (MPS). One issue with DMRG is that it doesn’t extend particularly well to 2D and 3D systems.

<div class="imgcap">
    <img src="/assets/quantum-nn/leaf.jpg" width="15%">
    <img src="/assets/quantum-nn/acorn.jpg" width="15%">
    <img src="/assets/quantum-nn/bark.jpg" width="15%">
</div>

**Quantum Monte Carlo (QMC).** Another way to study the concept of “tree” in a lab (bear with me on this metaphor) would be to study a bunch of leaf, seed, and bark samples. Quantum Monte Carlo algorithms do this with wave functions, taking “samples” of a wave function (pure states) and using the properties and frequencies of these samples to build a picture of the wave function as a whole. The difficulty with QMC is that it treats the wave function as a black box. We might ask, “how does flipping the spin of the third electron affect the total energy?” and QMC wouldn’t have much of a physical answer.

## Brains $$\gg$$ Brawn

<div class="imgcap_noborder">
    <img src="/assets/quantum-nn/nqs.jpg" width="40%">
    <div class="thecap" style="text-align:center">A schema of the Neural Quantum State (NQS) model introduced By Carleo and Troyer. The model has a Restricted Boltzman Machine (RBM) architecture. Increasing <em>M</em>, the number of units in the hidden layer, increases accuracy.</div>
</div>

Neural Quantum States (NQS). Some state spaces are far too large for even Monte Carlo to sample adequately. Suppose now we’re studying a forest full of different species of trees. If one type of tree vastly outnumbers the others, choosing samples from random trees isn’t an efficient way to map biodiversity. Somehow, we need to make the sampling process “smarter”. Last year, Google DeepMind used a technique called deep reinforcement learning to do just that – and achieved fame for [defeating the world champion human Go player.](https://deepmind.com/research/alphago/) 

A recent [Science paper](http://science.sciencemag.org/content/355/6325/602.full) by Carleo and Troyer (2017) used the same technique to make QMC “smarter” and effectively compress wave functions with neural networks. This approach, called “Neural Quantum States (NQS)”, produced several state-of-the-art results.

<div class="imgcap_noborder">
    <img src="/assets/quantum-nn/mps-learn-schema.png" width="100%">
    <div class="thecap" style="text-align:center">A schema of the neural network model I used to obtain MPS coefficients. The Hamiltonian I'm using is a Heisenberg Hamiltonain plus extra coupling terms (see <a href="https://github.com/greydanus/psi0nn/blob/master/static/greydanus-dartmouth-thesis.pdf">my thesis</a> for details). Colors denote the magnitudes of scalar matrix elements.</div>
</div>

**My thesis.** My undergraduate thesis, which I conducted under fearless [Professor James Whitfield](http://jdwhitfield.com/) of Dartmouth College, centered upon much the same idea. In fact, I had to abandon some of my initial work after reading the NQS paper. I then focused on using machine learning techniques to obtain MPS coefficients. Like Carleo and Troyer, I used neural networks to approximate  \psi . Unlike Carleo and Troyer, I trained my model to output a set of Matrix Product State coefficients which have physical meaning (MPS coefficients always correspond to a certain state and site, e.g. “spin up, electron number 3”).

$$
  \label{eqn:mps-definition}
  \lvert \psi_{mps} \rangle=\sum_{s_1,\dots,s_N=1}^d Tr(A[1]^{s_1}, \dots A[N]^{s_N}) \lvert s_1, \dots s_N \rangle
$$

**A word about MPS.** I should quickly explain what, exactly, a Matrix Product State _is_. Check out the equation above, which is the definition of MPS. The idea is to multiply a set of matrices, $$A$$ together and take the trace of the result. Each $$A$$ matrix corresponds to a particular site, $$A[n]$$, (e.g. "electron 3") and a particular state, $$A^{s_i}$$ (e.g. "spin $$\frac{1}{2}$$"). Each of the values obtained from the trace operation becomes a single coefficient of $$\psi$$, corresponding to a particular state $$\lvert s_1, \dots s_N \rangle$$.

## Cool – but does it work?

**Yes – for small systems.** In my thesis, I considered a toy system of 4 spin-\frac{1}{2} particles interacting via the Heisenberg Hamiltonian. Solving this system is not difficult so I was able to focus on fitting the two disparate parts – machine learning and Matrix Product States – together.

Success! My model solved for ground states with arbitrary precision. Even more interestingly, I used it to automatically obtain MPS coefficients. Shown below, for example, is a visualization of my model’s coefficients for the [GHZ state](https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state), compared with coefficients [taken from the literature](http://www2.mpq.mpg.de/Theorygroup/CIRAC/wiki/images/9/9f/Eckholt_Diplom.pdf).

<div class="imgcap_noborder">
    <img src="/assets/quantum-nn/ghz-literature.png" width="46%">
    <img src="/assets/quantum-nn/ghz-mps-learn.png" width="46%">
    <img src="/assets/quantum-nn/ghz-colorscale.png" width="7%">
    <div class="thecap" style="text-align:center">A visual comparison of a 4-site Matrix Product State for the GHZ state <b>a)</b> listed in the literature <b>b)</b> obtained from my neural network model.</div>
</div>

**Limitations.** The careful reader might point out that, according to the schema of my model (above), I still have to write out the full wave function. To scale my model up, I instead trained it variationally over a subspace of the Hamiltonian (just as the authors of the NQS paper did). Results are decent for larger (10-20 particle) systems, but the training itself [is still unstable](https://stats.stackexchange.com/questions/265964/why-is-deep-reinforcement-learning-unstable). I’ll finish ironing out the details soon, so keep an eye on arXiv[^fn1] :).

## Outside the ivory tower

<div class="imgcap_noborder">
    <img src="/assets/quantum-nn/qcomputer.jpg" width="40%">
    <div class="thecap" style="text-align:center">A quantum computer developed by Joint Quantum Institute, U. Maryland.</div>
</div>

Quantum computing is a field that’s poised to [take on commercial relevance](https://www.nature.com/news/quantum-computers-ready-to-leap-out-of-the-lab-in-2017-1.21239). Taming the wave function is one of the big hurdles we need to clear before this happens. Hopefully my findings will have a small role to play in making this happen.

On a more personal note, thank you for reading about my work. As a recent undergrad, I’m still new to research and I’d love to hear constructive comments or criticisms. If you found this post interesting, check out my research blog.

[^fn1]: arXiv is an online library for electronic preprints of scientific papers