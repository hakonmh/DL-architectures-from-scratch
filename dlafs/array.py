from dlafs.autograd import Value
from dlafs.helpers import np_array_to_list_of_values


class ValueArray:
    """A class for representing a multidimensional array of Value() objects.

    Only indexing methods are implemented, so the class is not important to
    understand.
    """

    def __init__(self, shape, dtype='float'):
        self.shape = shape
        self.values = self._create_array(self.shape)
        self.dtype = dtype

    def _create_array(self, shape):
        if len(shape) == 1:
            return [Value(0) for x in range(shape[0])]
        elif len(shape) > 1:  # works recursively
            return [self._create_array(shape[1:]) for _ in range(shape[0])]

    @classmethod
    def from_numpy(cls, data):
        """Create an Array from a numpy array"""
        dtype = 'float' if 'float' in str(data.dtype) else 'int'
        data = np_array_to_list_of_values(data)
        return cls.from_list(data, dtype)

    @classmethod
    def from_list(cls, data, dtype='float'):
        """Create an Array from a list (potentially nested)"""
        shape = cls._get_shape_from_data(data)
        instance = cls(shape, dtype)
        instance._set_data_from_list(data)
        return instance

    @staticmethod
    def _get_shape_from_data(data):
        """Recursively find the shape of the nested list"""
        if not isinstance(data, list):
            return []
        return [len(data)] + ValueArray._get_shape_from_data(data[0])

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

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        item = self._recursive_getitem(self.values, index)
        if isinstance(item, list):  # Convert the list to an Array
            return ValueArray.from_list(item, self.dtype)
        return item

    def _recursive_getitem(self, data, indices):
        current_idx, *remaining_idx = indices

        if not remaining_idx:  # If this is the last dimension
            return data[current_idx]
        elif isinstance(current_idx, slice):  # If current dimension is a slice
            return [self._recursive_getitem(subdata, remaining_idx) for subdata in data[current_idx]]
        elif isinstance(current_idx, int):  # If current dimension is an integer
            return self._recursive_getitem(data[current_idx], remaining_idx)

    def __setitem__(self, index, value):
        if not isinstance(index, tuple):
            index = (index,)
        for _ in range(len(self.shape) - len(index)):
            index = index + (slice(None),)

        data = self._recursive_setitem(self.values, index, value)
        self.values = data

    def _recursive_setitem(self, data, indices, value):
        current_idx, *remaining_idx = indices

        if not remaining_idx:  # If this is the last dimension
            data = self._setdata(data, current_idx, value)
        elif isinstance(current_idx, slice):
            for subdata, subvalue in zip(data[current_idx], value):
                data[current_idx] = self._recursive_setitem(subdata, remaining_idx, subvalue)
        elif isinstance(current_idx, int):
            data[current_idx] = self._recursive_setitem(data[current_idx], remaining_idx, value)
        return data

    def _setdata(self, data, index, value):
        if isinstance(index, slice):
            for i in range(len(data[index])):
                data = self._setdata(data[index], i, value[i])
        else:
            if isinstance(data[index], list):
                raise IndexError("setting a list to a scalar value. Check your indices.")
            if isinstance(value, list):
                raise ValueError("setting an array element with a sequence. Check your indices.")
            value = value if isinstance(value, list) else Value(value)
            data[index] = value
        return data

    def _repr_helper(self, data, depth):
        if depth == 1:  # Base case: innermost list
            if self.dtype == 'float':
                return "[" + ", ".join(f"{val.data:.2f}" for val in data) + "]"
            elif self.dtype == 'int':
                return "[" + ", ".join(f"{val.data}" for val in data) + "]"
        else:
            newlines = '\n' * (depth - 1)
            spaces = ' ' * (5 + len(self.shape) - depth)
            join_str = f",{newlines}{spaces}"
            return "[" + join_str.join(self._repr_helper(item, depth - 1) for item in data) + "]"

    def __repr__(self):
        return f"ValueArray(\n    {self._repr_helper(self.values, depth=len(self.shape))}\n)"
