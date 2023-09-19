import pytest
import math
from dlafs import loss, Value as V


@pytest.mark.parametrize(
    "y, yhat, expected",
    [
        ([1, 2, -1, 0, -1], [0, 2, 1, 0, -1], 1.0),
        (
            [V(1), V(2), V(-1), V(0), V(-1)],
            [V(0), V(2), V(1), V(0), V(-1)],
            1.0
        ),
        (V(1), V(0), 1.0),
        (1, 0, 1),
    ]
)
def test_mse_loss(y, yhat, expected):
    # Arrange
    # Act
    mse = loss.mse(y, yhat)
    if isinstance(mse, V):
        mse = mse.data
    # Assert
    math.isclose(mse, expected)


@pytest.mark.parametrize(
    "y, yhat, expected",
    [
        ([1, 2, 0, 1], [1, 2, 1, 0], 0.75),
        (
            [V(1), V(2), V(0), V(1)],
            [V(1), V(2), V(1), V(0)],
            0.75
        ),
        (V(1), V(0), 0),
    ]
)
def test_accuracy(y, yhat, expected):
    # Arrange
    # Act
    accuracy = loss.accuracy(y, yhat)
    if isinstance(accuracy, V):
        accuracy = accuracy.data
    # Assert
    math.isclose(accuracy, expected)


@pytest.mark.parametrize(
    "y, yhat, expected",
    [
        ([1, 0, 0, 0], [0.7, 0.1, 0.15, 0.05], 0.356674943938731),
        (
            [V(1), V(0), V(0), V(0)],
            [V(0.7), V(0.1), V(0.15), V(0.05)],
            0.356674943938731
        ),
        (
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            [[0.7, 0.1, 0.15, 0.05], [0.1, 0.8, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7]],
            0.31216447973055683
        ),
        ([[1, ], [0, ], [1, ]], [[0.7, ], [0.1, ], [0.95, ]], 0.13598941277542767),
    ],
    ids=["int-one-sample", "value-one-sample", "int-multiple-samples", "binary-multiple-samples"]
)
def test_cross_entropy_loss(y, yhat, expected):
    # Arrange
    # Act
    cross_entropy = loss.cross_entropy(y, yhat)
    if isinstance(cross_entropy, V):
        cross_entropy = cross_entropy.data
    # Assert
    assert math.isclose(cross_entropy, expected)
