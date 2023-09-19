import math
import torch
from dlafs import Value


def assert_grads_equal_expected(values, expected_grads):
    for value, exp_grad in zip(values, expected_grads):
        if isinstance(exp_grad, torch.Tensor):
            exp_grad = exp_grad.grad.item()
        assert math.isclose(value.grad, exp_grad)


def convert_list_items_to_value(data):
    """Recursive function to create valuearray from nested list"""
    if isinstance(data, list):
        return [convert_list_items_to_value(item) for item in data]
    else:
        if not isinstance(data, Value):
            return Value(data)
        else:
            return data


def format_index_args_string(indexers_dict):
    """Convert test setitem/getitem args into strs of format '{num_dims}D, [{index}]'."""
    ids = []
    for num_dim, indexers, in indexers_dict.items():
        for index in indexers:
            if isinstance(index, tuple):
                index = index[0]
            _id = []
            for i in index:
                if isinstance(i, slice):
                    _id.append(f"{i.start}:{i.stop}")
                else:
                    _id.append(str(i))
            _id = f'{num_dim}D, [' + ', '.join(_id) + ']'
            _id = _id.replace('None', '')
            ids.append(_id)
    return ids
