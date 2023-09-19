import random
from collections.abc import Sequence

from dlafs.autograd import Value
from dlafs.helpers import (
    np_array_to_list_of_values,
    list_of_values_to_np_array
)
from dlafs._utils import format_float_string


class ValueArray:
    """A class for representing a multidimensional array of Value() objects.

    Only indexing methods are implemented, so the class is not important to
    understand.
    """

    def __init__(self, shape):
        self.shape = shape
        self.values = self._create_array(self.shape)

    def _create_array(self, shape):
        if len(shape) == 1:
            return [Value(0) for x in range(shape[0])]
        elif len(shape) > 1:  # works recursively
            return [self._create_array(shape[1:]) for _ in range(shape[0])]

    @classmethod
    def random(cls, shape):
        """Create Array of random values from a normal dist with mean 0 and std 1"""
        instance = cls(shape)
        instance.values = _randomize_values(instance.values)
        instance.shape = tuple(instance._get_shape_from_data(instance.values))
        return instance

    @classmethod
    def from_numpy(cls, data):
        """Create an Array from a numpy array"""
        data = np_array_to_list_of_values(data)
        return cls.from_list(data)

    @classmethod
    def from_list(cls, data):
        """Create an Array from a list (potentially nested)"""
        shape = tuple(cls._get_shape_from_data(data))
        instance = cls(shape)
        instance._set_data_from_list(data)
        return instance

    @staticmethod
    def _get_shape_from_data(data):
        """Recursively find the shape of the nested list"""
        if not isinstance(data, list):
            return []
        else:
            shape = None
            for item in data:
                if shape is None:
                    shape = ValueArray._get_shape_from_data(item)
                else:
                    new_shape = ValueArray._get_shape_from_data(item)
                    if new_shape != shape:
                        raise ValueError("Array has inconsistent shape.")
            return [len(data)] + shape

    def _set_data_from_list(self, data, i=None):
        """Recursively set data from nested list"""
        if i is None:
            i = []

        for idx, val in enumerate(data):
            current_index = i + [idx]

            if isinstance(val, list):
                self._set_data_from_list(val, current_index)
            else:
                target_data = self.values
                for ind in current_index[:-1]:  # navigate to the target list
                    target_data = target_data[ind]
                val = Value(val) if not isinstance(val, Value) else val
                target_data[current_index[-1]] = val

    def to_numpy(self):
        """Convert the Array to a numpy array"""
        return list_of_values_to_np_array(self.values)

    def to_list(self, preserve_dtype=False):
        """Convert the Array to a nested list"""
        return self.to_numpy().tolist()

    def __len__(self):
        """Return the length of the Array"""
        return len(self.values)

    def __getitem__(self, index):
        """Get an item from the Array using the given index"""
        if not isinstance(index, tuple):
            index = (index,)
        item = self._recursive_getitem(self.values, index)
        if isinstance(item, list):  # Convert the list to an Array
            return ValueArray.from_list(item)
        return item

    def _recursive_getitem(self, data, indices):
        current_idx, *remaining_idx = indices
        if isinstance(current_idx, slice):
            current_idx = _convert_slice_to_range(current_idx, data)

        is_last_dim = not remaining_idx
        if is_last_dim:
            if isinstance(current_idx, Sequence):
                return [data[idx] for idx in current_idx]
            else:
                return data[current_idx]
        elif isinstance(current_idx, Sequence):
            return [self._recursive_getitem(data[idx], remaining_idx) for idx in current_idx]
        elif isinstance(current_idx, int):
            return self._recursive_getitem(data[current_idx], remaining_idx)

    def __setitem__(self, index, value):
        """Set an item in the Array using the given index"""
        if not isinstance(index, tuple):
            index = (index,)
        for _ in range(len(self.shape) - len(index)):
            index = index + (slice(None),)

        data = self._recursive_setitem(self.values, index, value)
        new_shape = tuple(self._get_shape_from_data(data))
        if new_shape != self.shape:
            raise ValueError(f"Can't reshape data using setitem.")

        self.values = data

    def _recursive_setitem(self, data, indices, value):
        current_idx, *remaining_idx = indices
        if isinstance(current_idx, slice):
            current_idx = _convert_slice_to_range(current_idx, data)

        is_last_dim = not remaining_idx
        if is_last_dim:
            _verify_integrity(current_idx, value)
            if isinstance(current_idx, Sequence):
                for idx, val in zip(current_idx, value):
                    data[idx] = Value(val)
            else:
                data[current_idx] = Value(value)
        elif isinstance(current_idx, Sequence):
            _verify_integrity(current_idx, value)
            for idx, val in zip(current_idx, value):
                data[idx] = self._recursive_setitem(data[idx], remaining_idx, val)
        elif isinstance(current_idx, int):
            data[current_idx] = self._recursive_setitem(
                data[current_idx], remaining_idx, value
            )
        return data

    def __repr__(self):
        self._max_len = 0
        return f"ValueArray(\n    {self._repr_helper(self.values, depth=len(self.shape))}\n)"

    def _repr_helper(self, data, depth):
        if depth == 1:  # Base case: innermost list
            str_data = [format_float_string(val.data) for val in data]
            # Add spaces to align the strings
            self._max_len = max(max(len(val) for val in str_data), self._max_len)
            str_data = [val + ' ' * (self._max_len - len(val)) for val in str_data]
            return "[" + ", ".join(str_data) + "]"
        else:
            newlines = '\n' * (depth - 1)
            spaces = ' ' * (5 + len(self.shape) - depth)
            join_str = f",{newlines}{spaces}"
            return "[" + join_str.join(self._repr_helper(item, depth - 1) for item in data) + "]"


def _convert_slice_to_range(slice_obj, data):
    return range(*slice_obj.indices(len(data)))


def _verify_integrity(current_idx, value):
    if not isinstance(current_idx, Sequence):
        current_idx = [current_idx]
    if not isinstance(value, Sequence):
        value = [value]

    if not len(current_idx) == len(value):
        raise ValueError(
            f"Shape mismatch: Trying to set data of lenth {len(value)} "
            f"on slice of length {len(current_idx)}"
        )


def _randomize_values(data):
    if isinstance(data[0], list):
        return [_randomize_values(subdata) for subdata in data]
    else:
        return [Value(random.gauss(0, 1)) for _ in range(len(data))]
