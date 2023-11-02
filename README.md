# Deep Learning Architectures from Scratch

This repository attempts to build different neural network architectures from scratch using only Pure Python. The goal is to keep my understanding of the inner workings of neural networks and the math behind them fresh, since it is easy to forget the details when using high-level libraries.

The code is written in a way that is easy to understand and to follow along. Explanations are provided in the form of Jupyter notebooks. The code is also tested to make sure the implementation is correct.

We first start with implementing AutoGrad, which is the backbone of PyTorch and is used to do automatic differentiation. We then use AutoGrad to implement a basic neural network. The goal is to later implement more complex architectures like CNNs, RNNs and Transformers.

## Layout

The repository is structured as follows:

* `dlafs`: The source code for autograd and the implementation of neural network architectures.
* `notebooks`: Jupyter notebooks for explaining the code in `dlafs` in detail.
* `tests`: Unit tests for the code in `dlafs`.
* `data`: Where the datasets used in `notebooks` is stored. We use several different datasets to test the models based on the task they are designed for.

## Status

The following models and tasks have been implemented:

* [x] AutoGrad
* [x] ValueArray
* [x] Vanilla Neural Network
* [x] Recurrent Neural Network
* [ ] LSTM
* [ ] Transformer
* [ ] Convolutional Neural Network

## Requirements

There is no requirements to run the models themself, but `numpy` and `graphviz` is used for some convenience functions found in `dlafs/helpers`. For running the notebooks, libraries like `graphviz`, `pandas`, `numpy` and `matplotlib`. `pytest`, `torch`, and `numpy` is required to run the tests.

## Sources

The following sources have been used to implement the models:

* [Andrej Karpathy's Intro to Backprop and Neural Nets](https://www.youtube.com/watch?v=VMj-3S1tku0)
* [PyTorch`s Autograd](https://pytorch.org/docs/stable/notes/autograd.html)
* [An Introduction to Recurrent Neural Networks and the Math That Powers Them](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Andrej Karpathy's GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
