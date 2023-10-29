import pytest
import random

from dlafs import ValueArray
from dlafs.vanilla_nn import *
from dlafs.rnn import *
from dlafs.loss import mse, binary_cross_entropy
from dlafs.train import Trainer


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
    assert model([1, 2, 3]).shape == (2,)


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
