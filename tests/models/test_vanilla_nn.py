import pytest
import random

from dlafs import ValueArray
from dlafs.vanilla_nn import *
from dlafs.loss import binary_cross_entropy
from dlafs.train import Trainer


def test_neuron():
    # Arrange
    EXPECTED = 0.008904
    random.seed(42)

    x = ValueArray([1, 2, 3, 4], label='x')
    model = Neuron(4, activation='sigmoid')
    # Act
    y_hat = model(x)
    # Assert
    assert y_hat.data == pytest.approx(EXPECTED, abs=1e-6)


def test_neuron_backward():
    # Arrange
    EXPECTED_GRADS = [0.38, -0.19, -0.19]
    random.seed(42)

    x = ValueArray([-2, 1], label='x')

    model = Neuron(2, activation='sigmoid')
    y_hat = model(x)
    loss = (y_hat - 1) ** 2
    # Act
    loss.backward()
    # Assert
    for parameter, expected_grad in zip(model.parameters(), EXPECTED_GRADS):
        assert parameter.grad == pytest.approx(expected_grad, abs=1e-2)


def test_layer():
    # Arrange
    EXPECTED = [0.029, 0.937, 0.023, 0.238]
    random.seed(42)

    x = ValueArray([1, 2, 3], label='x')

    model = Layer(3, 4, activation='sigmoid')
    # Act
    y_hat = model(x).to_list()
    # Assert
    for y_hat_i, y_i in zip(y_hat, EXPECTED):
        assert y_hat_i == pytest.approx(y_i, abs=1e-3)


def test_vanilla_nn_init():
    # Arrange
    # Act
    model = VanillaNN([
        Layer(3, 12, activation='relu'),
        Layer(12, 8, activation='relu'),
        Layer(8, 6, activation='tanh'),
        Layer(6, 3, activation='relu'),
        Layer(3, 2, activation='sigmoid')
    ])
    # Assert
    assert len(model.layers) == 5
    assert len(model.parameters()) == 235


def test_vanilla_nn():
    # Arrange
    random.seed(42)
    EXPECTED = 0.308115
    model = VanillaNN([
        Layer(3, 12, activation='relu'),
        Layer(12, 8, activation='relu'),
        Layer(8, 6, activation='tanh'),
        Layer(6, 3, activation='tanh'),
        Layer(3, 2, activation='sigmoid')
    ])
    # Act
    y_hat = model([1, 2, 3])
    # Assert
    assert y_hat.shape == (2,)
    assert y_hat[0].data == pytest.approx(EXPECTED, abs=1e-6)


def test_vanilla_nn_backward():
    # Arrange
    EXPECTED_GRADS = [-0.0972, -0.0647, 0.0324]
    random.seed(42)

    x = ValueArray([-3, -2], label='x')

    model = VanillaNN([
        Layer(2, 4, activation='relu'),
        Layer(4, 3, activation='tanh'),
        Layer(3, 1, activation='sigmoid')
    ])
    y_hat = model(x)
    loss = (y_hat - 0) ** 2
    # Act
    loss.backward()
    # Assert
    for parameter, expected_grad in zip(model.parameters()[:3], EXPECTED_GRADS):
        assert parameter.grad == pytest.approx(expected_grad, abs=1e-2)


def test_vanilla_nn_zero_grad():
    # Arrange
    random.seed(42)
    model = VanillaNN([
        Layer(2, 4, activation='relu'),
        Layer(4, 3, activation='tanh'),
        Layer(3, 1, activation='sigmoid')
    ])
    x = ValueArray([1, 2], label='x')
    y_hat = model(x)
    loss = (y_hat - 1) ** 2
    loss.backward()
    # Act
    model.zero_grad()
    # Assert
    for parameter in model.parameters():
        assert parameter.grad == 0


@pytest.mark.integration
def test_train_vanilla_nn():
    # Arrange
    random.seed(42)
    x = ValueArray.random_normal(shape=(10, 2), mean=0, std=1, label='x')
    y = [[int((x_i[0] + x_i[1]) > 0)] for x_i in x]  # y = 1 if the sum of the inputs is positive
    y = ValueArray(y, label='y')

    model = VanillaNN([
        Layer(2, 12, activation='relu'),
        Layer(12, 8, activation='relu'),
        Layer(8, 6, activation='tanh'),
        Layer(6, 3, activation='relu'),
        Layer(3, 1, activation='sigmoid')
    ])
    trainer = Trainer(model, binary_cross_entropy, learning_rate=2e-1)
    # Act
    loss = trainer.train(x, y, 50, silent=True)
    # Assert
    for x_i, y_i in zip(x, y):
        assert model(x_i).data == pytest.approx(y_i[0].data, abs=1e-1)
    assert loss[-1] == pytest.approx(0.02638213, abs=1e-4)
