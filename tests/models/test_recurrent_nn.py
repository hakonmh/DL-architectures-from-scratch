import pytest
import random

from dlafs import ValueArray
from dlafs.vanilla_nn import Layer
from dlafs.rnn import *
from dlafs.loss import mse


def test_recurrent_nn():
    # Arrange
    # Act
    model = RecurrentNN([
        RecurrentLayer(num_inputs=2, hidden_size=5, activation='tanh'),
        RecurrentLayer(num_inputs=5, hidden_size=5, activation='tanh'),
        Layer(5, 2, activation='sigmoid')
    ])
    # Assert
    assert len(model.layers) == 3
    assert len(model.parameters()) == 107
    assert model([(1, 3), (4, 2), [5, 7], [-2, 15]]).shape == (4, 2)
    assert model([(1, 3), (4, 2), [5, 7], [-2, 15]])[-1].shape == (2,)


@pytest.mark.integration
def test_train_recurrent_nn():
    # Arrange
    random.seed(42)
    x = _generate_random_length_data(10, max_length=4, num_features=2)
    y = _sum_data(x)

    model = RecurrentNN([
        RecurrentLayer(num_inputs=2, hidden_size=8, activation='tanh'),
        Layer(8, 4, activation='tanh'),
        Layer(4, 1, activation='linear')
    ])
    # Act
    loss = _train_rnn(model, mse, x, y, epochs=30, lr=5e-2)
    # Assert
    for x_i, y_i in zip(x, y):
        assert model(x_i)[-1].data == pytest.approx(y_i[0].data, abs=6e-1)
    assert loss[-1] == pytest.approx(0.0594, abs=1e-3)


def _generate_random_length_data(num_examples, max_length, num_features):
    x = []
    for _ in range(num_examples):
        length = random.randint(1, max_length)
        x.append(ValueArray.random_uniform(shape=(length, num_features), low=-1, high=1).values)
    return x


def _sum_data(x):
    return ValueArray([[sum([sum(x_ij) for x_ij in x_i])] for x_i in x], label='y')


def _train_rnn(model, loss_fn, inputs, labels, epochs=50, lr=3e-1):
    data = []
    for _ in range(epochs):
        outputs = [[model(x)[-1]] for x in inputs]  # Many-to-one.
        loss = loss_fn(labels, outputs)
        labels.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.data -= lr * param.grad
        model.zero_grad()
        data.append(loss.data)
    return data
