import pytest
import math
from dlafs import loss, Value


@pytest.mark.parametrize(
    "y, yhat, expected",
    [
        ([1, 2, -1, 0, -1], [0, 2, 1, 0, -1], 1.0),
        (
            [Value(1), Value(2), Value(-1), Value(0), Value(-1)],
            [Value(0), Value(2), Value(1), Value(0), Value(-1)],
            1.0
        ),
        (Value(1), Value(0), 1.0),
        (1, 0, 1),
    ]
)
def test_mse_loss(y, yhat, expected):
    # Arrange
    # Act
    mse = loss.mse(y, yhat)
    if isinstance(mse, Value):
        mse = mse.data
    # Assert
    math.isclose(mse, expected)


@pytest.mark.parametrize(
    "y, yhat, expected",
    [
        ([1, 2, 0, 1], [1, 2, 1, 0], 0.75),
        (
            [Value(1), Value(2), Value(0), Value(1)],
            [Value(1), Value(2), Value(1), Value(0)],
            0.75
        ),
        (Value(1), Value(0), 0),
    ]
)
def test_accuracy(y, yhat, expected):
    # Arrange
    # Act
    accuracy = loss.accuracy(y, yhat)
    if isinstance(accuracy, Value):
        accuracy = accuracy.data
    # Assert
    math.isclose(accuracy, expected)


@pytest.mark.parametrize(
    "y, yhat, expected",
    [
        ([1, 0, 0, 0], [0.7, 0.1, 0.15, 0.05], 0.356674943938731),
        (
            [Value(1), Value(0), Value(0), Value(0)],
            [Value(0.7), Value(0.1), Value(0.15), Value(0.05)],
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
    if isinstance(cross_entropy, Value):
        cross_entropy = cross_entropy.data
    # Assert
    assert math.isclose(cross_entropy, expected)
