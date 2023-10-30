import pytest
import math
from dlafs import loss, ValueArray as VA, Value as V


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
        ([[1, ], [0, ], [1, ]], [[0.7, ], [0.1, ], [0.95, ]], 0.17110958466136975),
        (VA([[1, ], [0, ], [1, ]]), VA([[0.7, ], [0.1, ], [0.95, ]]), 0.17110958466136975),
        (1, 0.7, 0.35667494393873245),
    ],
    ids=["int-one-sample", "value-one-sample", "int-multiple-samples", "binary-multiple-samples",
         "binary-valuearray-multiple-samples", "binary-scalar-one-sample"]
)
def test_cross_entropy_loss(y, yhat, expected):
    # Arrange
    # Act
    cross_entropy = loss.cross_entropy(y, yhat)
    if isinstance(cross_entropy, V):
        cross_entropy = cross_entropy.data
    # Assert
    assert math.isclose(cross_entropy, expected)


@pytest.mark.parametrize(
    "y, yhat, expected",
    [
        ([1, 0, 0, 1], [0.85, 0.33, 0.001, 0.001], 1.8679380688526552),
        (
            [V(1), V(0), V(0), V(1)],
            [V(0.85), V(0.33), V(0.001), V(0.001)],
            1.8679380688526552
        ),
        ([1], [0.8], 0.2231435513142097),
        (1, 0, 100),
        (0, 1, 100),
    ],
    ids=["int-multiple-sample", "value-multiple-sample", "int-one-sample",
         "scalar-outlier-one-sample_1", "scalar-outlier-one-sample_2"]
)
def test_binary_cross_entropy_loss(y, yhat, expected):
    # Arrange
    # Act
    binary_cross_entropy = loss.binary_cross_entropy(y, yhat)
    if isinstance(binary_cross_entropy, V):
        binary_cross_entropy = binary_cross_entropy.data
    # Assert
    assert math.isclose(binary_cross_entropy, expected)


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
        ([0, 1], [1, 0], 115.12925464970229),
        ([[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [1, 0, 0]], 57.564627324851145),
    ],
    ids=["int-single-sample", "value-single-sample", "int-multiple-samples",
         "outlier-single-sample", "outlier-multiple-samples"]
)
def test_multiclass_cross_entropy_loss(y, yhat, expected):
    # Arrange
    # Act
    cross_entropy = loss.multi_cross_entropy(y, yhat)
    if isinstance(cross_entropy, V):
        cross_entropy = cross_entropy.data
    # Assert
    assert math.isclose(cross_entropy, expected)
