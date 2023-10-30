import pytest
from .fixtures import *

import math
import torch
from dlafs.autograd import Value

ADD_OTHER = ((5, 2), (lambda x, y: x + y), 7, (1, 1))
ADD_SELF = ((10.5,), (lambda x: x + x), 21.0, (2,))

SUB_OTHER = ((10.5, 1.4), (lambda x, y: x - y), 9.1, (1, -1))
SUB_SELF = ((-3.2,), (lambda x: x - x), 0, (0,))

MUL_OTHER = ((3, 5), (lambda x, y: x * y), 15, (5, 3))
MUL_SELF = ((5.0,), (lambda x: x * x), 25.0, (10.0,))

DIV_OTHER = ((10, 2), (lambda x, y: x / y), 5, (0.5, -2.5))
DIV_SELF = ((100,), (lambda x: x / x), 1, (0,))

POW_OTHER = ((4, 2), (lambda x, y: x ** y), 16, (8, 16 * math.log(4)))
POW_SELF = ((2,), (lambda x: x ** x), 4, (4 * (math.log(2) + 1),))

EXP = ((2,), (lambda x: x.exp()), math.exp(2), (math.exp(2),))
LOG = ((2,), (lambda x: x.log()), math.log(2), (0.5,))

TANH = ((2,), (lambda x: x.tanh()), math.tanh(2), (1 - math.tanh(2)**2,))
RELU = ((2,), (lambda x: x.relu()), 2, (1,))
SIGMOID = ((math.log(3),), (lambda x: x.sigmoid()), 0.75, (0.1875,))


@pytest.mark.parametrize(
    ("values", "operations", "expected", "expected_grads"),
    [
        ADD_OTHER, ADD_SELF,
        SUB_OTHER, SUB_SELF,
        MUL_OTHER, MUL_SELF,
        DIV_OTHER, DIV_SELF,
        POW_OTHER, POW_SELF,
        EXP, LOG, TANH, RELU, SIGMOID,
    ],
    ids=[
        "add_other", "add_self",
        "sub_other", "sub_self",
        "mul_other", "mul_self",
        "div_other", "div_self",
        "pow_other", "pow_self",
        "exp", "log", "tanh", "relu", "sigmoid",
    ]
)
def test_value(values, operations, expected, expected_grads):
    """Test the Value class and its operations."""
    # Arrange
    values = [Value(v) for v in values]
    # Act
    actual = operations(*values)
    actual.backward()
    # Assert
    assert math.isclose(actual.data, expected)
    assert_grads_equal_expected(values, expected_grads)


@pytest.mark.parametrize(
    ("values", "operations"),
    [
        ((3,), lambda x: (x + x) + x**2 / x.exp()),
        ((10, 4.6, -24), lambda x, y, z: x + y * z),
        ((3, 5, 2), lambda x, y, z: x ** x + y**z),
        ((2, 3, 4, 5), lambda x, y, z, w: x.log()**y + z.tanh() / w),
        ((2, 3, 4, 5), lambda x, y, z, w: sum([x.log(), y.relu(), z.sigmoid(), w.exp()]) / 4),
    ]
)
def test_value_vs_torch(values, operations):
    """Test the Value class against torch.Tensor with more complex expressions."""
    # Arrange
    tensors = [torch.tensor([v], requires_grad=True, dtype=torch.float64) for v in values]
    values = [Value(v) for v in values]

    expected = operations(*tensors)
    expected.backward()
    # Act
    actual = operations(*values)
    actual.backward()
    # Assert
    assert math.isclose(actual.data, expected.item())
    assert_grads_equal_expected(values, tensors)


def test_equals():
    # Arrange
    actual = Value(3)
    expected = Value(3)

    wrong_value = Value(4)
    wrong_grad = Value(3); wrong_grad.grad = 1  # noqa
    wrong_obj_type = 3
    # Assert
    assert actual == expected
    assert actual != wrong_value
    assert actual != wrong_grad
    assert actual != wrong_obj_type
