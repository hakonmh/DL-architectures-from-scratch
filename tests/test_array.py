import pytest
from .fixtures import *

import numpy as np
import random
from dlafs import ValueArray
from dlafs import Value as V


@pytest.mark.parametrize(
    'shape, expected',
    [
        ((1, ), [V(0), ]),
        ((3, ), [V(0), V(0), V(0)]),
        ((1, 1), [[V(0), ]]),
        ((2, 2), [[V(0), V(0)], [V(0), V(0)]]),
        ((3, 1, 2), [[[V(0), V(0)], ], [[V(0), V(0)], ], [[V(0), V(0)], ]]),
        ((1, 2, 1, 2), [[[[V(0), V(0)], ], [[V(0), V(0)], ]]]),
    ],
    ids=['(1, )', '(3, )', '(1, 1)', '(2, 2)', '(3, 1, 2)', '(1, 2, 1, 2)']
)
def test_valuearray_zeros(shape, expected):
    # Act
    actual = ValueArray.zeros(shape)
    # Assert
    assert actual.values == expected
    assert actual.to_numpy().shape == shape


def test_valuearray_random():
    # Arrange
    random.seed(0)
    expected = [
        [V(0.941715), V(-1.396578)],
        [V(-0.679714), V(0.370504)],
        [V(-1.016349), V(-0.07212)]
    ]
    # Act
    actual = ValueArray.random_normal((3, 2))
    # Assert
    assert actual.shape == (3, 2)
    for row, exp_row in zip(actual.values, expected):
        for val, exp_val in zip(row, exp_row):
            assert math.isclose(val.data, exp_val.data, rel_tol=1e-4)


@pytest.mark.parametrize(
    'input_list',
    [
        [1, ],
        [V(1), ],
        [1.01, 2.02],
        [[1, ]],
        [[[[V(1), V(2)], [V(3), V(4)]], [[V(5), V(6)], [V(7), V(8)]]]],
        ValueArray([[1, 2], [3, 4]]),
        [ValueArray([[1, 2], [3, 4]]), ValueArray([[5, 6], [7, 8]])],
    ],
    ids=['0D-int', 'single-value', '1D-float', '2D-int', '4D-value', '2D-valuearray', 'List-of-ValueArray']
)
def test_init_from_sequence(input_list):
    # Arrange
    expected = convert_list_items_to_value(input_list)
    # Act
    actual = ValueArray(input_list)
    # Assert
    assert actual.values == expected


@pytest.mark.parametrize(
    'input_scalar',
    [
        V(0.5),
        4.2,
        -9
    ]
)
def test_init_from_scalar(input_scalar):
    # Arrange
    expected = [convert_list_items_to_value(input_scalar)]
    # Act
    actual = ValueArray(input_scalar)
    # Assert
    assert actual.values == expected


@pytest.mark.parametrize(
    'input_list',
    [
        [1, ],
        [1.01, 2.02],
        [[1, ]],
        [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]],
    ],
    ids=['single', '1D-float', '2D', '4D']
)
def test_from_numpy(input_list):
    # Arrange
    np_array = np.array(input_list)
    expected = convert_list_items_to_value(input_list)
    # Act
    actual = ValueArray.from_numpy(np_array)
    # Assert
    assert actual.values == expected
    assert actual.shape == np_array.shape


def test_to_numpy():
    # Arrange
    varray = ValueArray.zeros((2, 2, 3))
    expected = np.zeros((2, 2, 3))
    # Act
    actual = varray.to_numpy()
    # Assert
    assert np.array_equal(actual, expected)


TEST_GETITEM_ARGS = {  # num_dims: (index, ...)
    1: ([0], [[0, 2]], [-1], [slice(1, 3)], [slice(None)]),
    2: ([3, 2], [0], [[2, -1], 0],
        [slice(1, 3), slice(1, 3)],
        [3, slice(None)], [slice(None), 2],
        [slice(None), slice(None)], [slice(None)]),
    4: ([0, 1, 2, 3], [0, 1],
        [[0, 2], 0, 0, 0], [1, 2, [0, 2], 0],
        [1, slice(1, 3), 0, 0], [0, 1, slice(None)],
        [slice(None), slice(None), slice(None), slice(None)],
        [slice(None), slice(None)]),
}


@pytest.mark.parametrize(
    "num_dims, index",
    [(n, i) for n, indexer in TEST_GETITEM_ARGS.items() for i in indexer],
    ids=format_index_args_string(TEST_GETITEM_ARGS)
)
def test_getitem(num_dims, index):
    index = tuple(index)
    np_array = _create_array(num_dims)
    varray = ValueArray.from_numpy(np_array)

    expected = np_array[index]
    expected_shape = expected.shape
    expected = convert_list_items_to_value(expected.tolist())
    # Act
    actual = varray[index]
    # Assert
    if isinstance(actual, ValueArray):
        assert actual.values == expected
        assert actual.shape == expected_shape
    else:
        assert actual == expected


TEST_SETITEM_ARGS = {  # num_dims: [(index, values), ...]
    1: [
        ([0], 1),
        ([[0, 2]], [3, 3]),
        ([slice(1, 3)], [5, 5]),
        ([slice(None)], [4, 3, 2, 1]),
    ],
    2: [
        ([3, 2], 1),
        ([0, ], [5, 5, 5, 5]),
        ([0, [1, 3]], [5, 5]),
        ([[1, 3], 0], [5, 5]),
        ([slice(None), 0], [5, 5, 5, 5]),
        ([slice(1, 3), slice(1, 3)], [[5, 5], [5, 5]]),
    ],
    4: [
        ([0, 1, 2, 3], 1),
        ([0, 1, 2], [5, 5, 5, 5]),
        ([[0, 2], 0, 0, 0], [5, 5]),
        ([1, 2, [0, 2], 0], [5, 5]),
        ([0, 1, slice(None)], [[5, 5, 5, 5]] * 4),
        ([slice(None), slice(None), slice(None), slice(None)], ValueArray.zeros((4, 4, 4, 4)).to_list()),
        ([slice(None), slice(None)], ValueArray.zeros((4, 4, 4, 4)).to_list()),
    ]
}


@pytest.mark.parametrize(
    "num_dims, index, values",
    ((n, *i) for n, indexer in TEST_SETITEM_ARGS.items() for i in indexer),
    ids=format_index_args_string(TEST_SETITEM_ARGS)
)
def test_setitem(num_dims, index, values):
    index = tuple(index)

    np_array = _create_array(num_dims)
    actual = ValueArray.from_numpy(np_array)

    np_array[index] = values
    expected_shape = np_array.shape
    expected = convert_list_items_to_value(np_array.tolist())
    # Act
    actual[index] = values
    # Assert
    if isinstance(actual, ValueArray):
        assert actual.values == expected
        assert actual.shape == expected_shape
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "index, values, exception",
    [([1, 1], [-0.1, 0.1], ValueError),
     ([1, slice(1, 3)], [1, 2, 3], ValueError),
     ([slice(1, 3), 1], [1, 2, 3], ValueError),
     ([1, slice(1, 3)], 1, ValueError),
     ([slice(1, 3), 0], 2, ValueError),
     ([slice(None), slice(None)], 3, ValueError),
     ([7, 7], 4, IndexError),
     ([1, 1], '5', TypeError),
     ],
)
def test_setitem_exeptions(index, values, exception):
    # Arrange
    index = tuple(index)
    np_array = _create_array(num_dims=2)
    varray = ValueArray.from_numpy(np_array)
    # Act & Assert
    with pytest.raises(exception):
        varray[index] = values


def _create_array(num_dims, size=4):
    shape = [size] * num_dims
    return np.arange(np.prod(shape)).reshape(shape)


def test_contains():
    # Arrange
    varray = ValueArray([[[2, 7], [5, -4]], [[0, 1], [1, 5]]])
    # Act
    five_in_varray = 5 in varray
    value_in_array = V(5) in varray
    ten_in_varray = 10 in varray
    # Assert
    assert five_in_varray
    assert value_in_array
    assert not ten_in_varray
