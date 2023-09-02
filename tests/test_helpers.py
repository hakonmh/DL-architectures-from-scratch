import numpy as np
from dlafs import Value
from dlafs.helpers import *


def test_argmax():
    # Arrange
    values = [
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.15, 0.075, 0.075],
    ]
    expected = [3, 0]
    # Act
    actual = [argmax(sample) for sample in values]
    # Assert
    assert all([a == e for a, e in zip(actual, expected)])


def test_list_of_values_to_np():
    # Arrange
    shape = (2, 3, 4)
    lists = _generate_list_of_values(shape)
    # generate numpy arange from 1 to n with shape (2, 3, 4)
    expected = np.arange(1, 25).reshape(shape)
    # Act
    actual = list_of_values_to_np_array(lists)
    # Assert
    assert np.array_equal(actual, expected)
    assert actual.shape == shape


def test_np_to_list_of_values():
    # Arrange
    shape = (2, 3, 4)
    array = np.arange(1, 25).reshape(shape)
    expected = _generate_list_of_values(shape)
    # Act
    actual = np_array_to_list_of_values(array)
    # Assert
    _assert_lists_equal(actual, expected)


def _assert_lists_equal(actual, expected):
    if isinstance(expected, Value):
        assert expected.data == actual.data
        assert expected.grad == actual.grad
        return

    assert len(expected) == len(actual)
    for exp, act in zip(expected, actual):
        _assert_lists_equal(exp, act)


def _generate_list_of_values(shape):
    lists, _ = __generate_list(shape)
    return lists


def __generate_list(shape, start=1):
    if len(shape) == 1:
        end = start + shape[0]
        values = [Value(i) for i in range(start, end)]
        return values, end

    outer_size = shape[0]
    inner_shape = shape[1:]

    lists = []
    for _ in range(outer_size):
        inner_list, start = __generate_list(inner_shape, start)
        lists.append(inner_list)
    return lists, start
