---
layout: post
comments: true
title:  "Parameter bloat"
excerpt: "Do we REALLY need over 100,000 free parameters to build a good MNIST classifier? It turns out that we can eliminate 80-90% of them."
date:   2017-10-30 11:00:00
mathjax: true
---

<div class="imgcap">
    <img src="/assets/subspace-nn/rube.png" width="40%">
    <div class="thecap" style="text-align:center">Some things, like the "self-operated napkin", are just too complicated!</div>
</div>

Do we REALLY need over 100,000 free parameters to build a good MNIST classifier? It turns out that we can eliminate 50-90% of them.

## How many parameters is enough?

The fruit fly was to genetics what the MNIST dataset is to deep learning: the ultimate case study. The idea is to classify handwritten digits between 0 and 9 using 28x28 pixel images. A few examples of these images are shown: 

<div class="imgcap_noborder">
    <img src="/assets/subspace-nn/mnist.png" width="25%">
    <div class="thecap" style="text-align:center">Examples taken from the MNIST dataset.</div>
</div>

**Ughh, not MNIST tutorials again.** There is an endless supply of tutorials that describe how to train an MNIST classifier. Looking over them recently, I noticed something: they all use a LOT of parameters. Here are a few examples:

Author | Framework | Type | Free parameters | Test accuracy 
:--- | :---: | :---: | :---: | :---:
[TensorFlow](https://www.tensorflow.org/get_started/mnist/beginners) | TensorFlow | Fully connected | 7,850 | 92%
[PyTorch](https://github.com/pytorch/examples/tree/master/mnist) | PyTorch | Convolutional | 21,840 | 99.0%
[Elite Data Science](https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-10) | Keras | Convolutional | 113,386 | 99\\(^+\\)%
[Machine Learning Mastery](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/) | Keras | Convolutional | 149,674 | 98.9%
[Lasagne](https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py) | Lasagne | Convolutional | 160,362 | 99\\(^+\\)%
[Caffe](http://caffe.berkeleyvision.org/gathered/examples/mnist.html) | Caffe | Convolutional | 366,030 | 98.9%
[Machine Learning Mastery](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/) | Keras | Fully connected | 623,290 | 98%
[Lasagne](https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py) | Lasagne | Fully connected | 1,276,810 | 98-99%
[TensorFlow](https://www.tensorflow.org/get_started/mnist/pros) | Tensorflow | Convolutional | 3,274,625 | 99.2%

It would seem that you have two options: use a small number of weights and get low accuracy (TensorFlow's logistic regression example) or use 100,000\\(^+\\) weights and get 99\\(^+\\)% accuracy (the PyTorch example is a notable exception). I think this leads to a common misconception that the best way to gain a few percentage points in accuracy is to double the size of your model.

**MNIST lite**. I was interested in how slightly smaller models would perform on the MNIST task, so I built and trained a few:

Framework | Structure | Type | Free parameters | Test accuracy 
:--- | :---: | :---: | :---: | :---:
PyTorch | 784 $$\mapsto$$ 16 $$\mapsto$$ 10 | Fully connected | 12,730 | 94.5%
PyTorch | 784 $$\mapsto$$ 32 $$\mapsto$$ 10 | Fully connected | 25,450 | 96.5%
PyTorch | 784 $$\mapsto$$ (6,4x4) $$\mapsto$$ (6,4x4) $$\mapsto$$ 25 $$\mapsto$$ 10 | Convolutional | 3,369 | 95.6%
PyTorch | 784 $$\mapsto$$ (8,4x4) $$\mapsto$$ (16,4x4) $$\mapsto$$ 32 $$\mapsto$$ 10 | Convolutional | 10,754 | 97.6%

You can find the code on my [GitHub](https://github.com/greydanus/subspace-nn). As you can see, it's still possible to obtain models that get 95\\(^+\\)% accuracy with fewer than 10$$^4$$ parameters. That said, we can do even better...

## Optimizing subspaces

> "Make everything as simple as possible, but not simpler." -- Albert Einstein

**The trick.** When I interviewed with _anonymous_ at _anonymous_ earlier this year, we discussed the idea of optimizing _subspaces_. In other words, if the vector $$\theta$$ contains all the parameters of a deep network, you might define $$\theta = P \omega$$ where the vector $$\omega$$ lives in some smaller-dimensional space and $$P$$ is a projector matrix. Then, instead of optimizing $$\theta$$, you could optimize $$\omega$$. With this trick, we can choose an arbitrary number of free parameters to optimize without changing the model's architecture. In math, the training objective becomes:

$$\omega = \arg\min_{\mathbf{\omega}}  -\frac{1}{n} \sum_X (y\ln \hat y +(1-y)\ln (1-\hat y)) \quad \mathrm{where} \quad \hat y = f_{NN}(\theta, X) \quad \mathrm{and} \quad \theta = P \omega$$

If you want to see this idea in code, check out my [subspace-nn](https://github.com/greydanus/subspace-nn) repo on GitHub.

**Results.** I tested this idea on the model from the PyTorch tutorial because it was the smallest model that achieved 99\\(^+\\)% test accuracy. The results were fascinating.

As shown below, cutting the number of free parameters in half (down to 10,000 free parameters) causes the test accuracy to drop by only 0.2%. This is substantially better than the "MNIST lite" model I trained with 10,754 free parameters. Cutting the subspace down to 3,000 free parameters produces a test accuracy of 97% which is still pretty good.

Framework | Free parameters | Test accuracy 
:--- | :---: | :---:
PyTorch | 21,840 | 99.0%
PyTorch | 10,000 | 98.8%
PyTorch | 3,000 | 97.0%
PyTorch | 1,000 | 93.5%

<div class="imgcap_noborder">
    <img src="/assets/subspace-nn/conv-large-accuracy.png" width="70%">
</div>

**Ultra small subspaces.** We can take this idea even futher and optimize our model in subspaces with well below 1,000 parameters. Shown below are test accuracies for subspaces of size 3, 10, 30, 100, 300, and 1000. Interestingly, I tried the same thing with a fully connected model and obtained nearly identical curves.

<div class="imgcap_noborder">
    <img src="/assets/subspace-nn/conv-accuracy.png" width="70%">
    <div class="thecap" style="text-align:center">Performing optimization in very small subspaces. Some of these training trajectories have not reached convergence; the 1,000-parameter model, for example, will converge to 93.5% accuracy after several more epochs.</div>
</div>

<!-- <div class="imgcap_noborder">
    <img src="/assets/subspace-nn/fc-accuracy.png" width="70%">
</div> -->

## Takeaways

**Update.** My friend, (anonymous), just finished a [paper about this](https://openreview.net/forum?id=ryup8-WCW&noteId=ryup8-WCW). Much more **#rigorous** than this post and definitely worth checking out: _"solving the cart-pole RL problem is in a sense 100 times easier than classifying digits from MNIST."_

**Literature.** The idea that MNIST classifiers are dramatically overparameterized is not new. The most common way to manage this issue is by adding a sparsity term (weight decay) to the loss function. At the end of the day, this doesn't exactly place a hard limit on the number of free parameters. One interesting approach, from [Convolution by Evolution](https://arxiv.org/pdf/1606.02580.pdf)$$^\dagger$$, is to _evolve_ a neural network with 200 parameters. The authors used this technique to train a denoising autoencoder so it's difficult to directly compare their results to ours.

There are several other papers that try to minimize the number of free parameters. However, I couldn't find any papers that used the subspace optimization trick. Perhaps it is a new and interesting tool.

**Bloat.** The immediate takeaway is that most MNIST classifiers have _parameter bloat_. In other words, they have many more free parameters than they really need. You should be asking yourself: is it worth doubling the parameters in a model to gain an additional 0.2% accuracy? I imagine that the more complex deep learning models (ResNets, VGG, etc.) also have this issue.

Training a model with more parameters than you'll need is fine as long as you know what you're doing. However, you should have a general intuition for _how much_ parameter bloat you've introduced. I hope this post helps with that intuition :)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$^\dagger$$Parts of this paper are a little questionable...for example, the 'filters' they claim to evolve in Figure 5 are just circles.