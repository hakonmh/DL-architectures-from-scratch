# Deep Learning Architectures from Scratch

This repository attempts to build different neural network architectures from scratch using only Pure Python. The goal is to keep my understanding of the inner workings of neural networks and the math behind them fresh, since it is easy to forget the details when using high-level libraries.

The code is written in a way that is easy to understand and to follow along. Explanations are provided in the form of Jupyter notebooks. The code is also tested against PyTorch implementations of the same models to make sure the implementation is correct.

We first start with implementing AutoGrad, which is the backbone of PyTorch and is used to do automatic differentiation. We then use AutoGrad to implement a simple neural network with one hidden layer. The goal is to later implement more complex architectures like CNNs, RNNs and Transformers.

## Layout

The repository is structured as follows:

* `dlafs`: The source code for autograd and the implementation of neural network architectures.
* `notebooks`: Jupyter notebooks for explaining the code in `dlafs` in detail.
* `tests`: Unit tests for the code in `dlafs`.
* `data`: Where the datasets used in `notebooks` is stored. We use several different datasets to test the models based on the task they are designed for.

## Status

The following models and tasks have been implemented:

* [x] AutoGrad
* [x] Vanilla Neural Network
* [ ] Recurrent Neural Network
* [ ] LSTM
* [ ] Transformer
* [ ] Convolutional Neural Network

## Requirements

There is no requirements to run the models themself, but `numpy` is used for some convenience functions found in `helpers`. Several libraries like `graphviz`, `pandas`, `numpy` and `matplotlib` is required to run the notebooks, while `pytest`, `torch`, and `numpy` is required to run the tests.

## Sources

The following sources have been used to implement the models:

* [Andrej Karpathy's Intro to Backprop and Neural Nets](https://www.youtube.com/watch?v=VMj-3S1tku0)
* [Andrej Karpathy's GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [PyTorch`s Autograd](https://pytorch.org/docs/stable/notes/autograd.html)
