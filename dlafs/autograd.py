import math
from numbers import Number
from dlafs._utils import format_float_string


class Value:

    def __new__(cls, data, label=''):
        if isinstance(data, Value):
            instance = data
            if label:
                instance.label = label
            return instance
        else:
            return super().__new__(cls)

    def __init__(self, data, label=''):
        if isinstance(data, Value):
            return
        if not isinstance(data, Number):
            raise TypeError(f"Value must be a number, not {type(data)}")

        self.data = data
        self.grad = 0
        self._children = set()
        self._operator = ''
        self.label = label
        self._backward = lambda: None

    def item(self):
        """Return self, a convenience method for working with ValueArray.item()"""
        return self

    @classmethod
    def _from_operation(cls, data, children, operator):
        """Create new object from an operation which stores the operator and operands used
        to produce it.
        """
        out = cls(data)
        out._children = set(children)
        out._operator = operator
        return out

    def __repr__(self):
        value = format_float_string(self.data)
        grad = format_float_string(self.grad)
        if self.grad == 0:
            grad_str = ''
        else:
            grad_str = f', grad={grad}'
        if self.label:
            return f"Value({value}{grad_str}, label='{self.label}')"
        else:
            return f"Value({value}{grad_str})"

    def __add__(self, other):
        other = Value(other)  # Convert to Value if needed
        out = Value._from_operation(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        # Implemented a-b as a+(b*-1) to reuse already implemented operations
        return self + (-other)

    def __mul__(self, other):
        other = Value(other)
        out = Value._from_operation(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = Value(other)
        out = Value._from_operation(self.data ** other.data, (self, other), '**')

        def _backward():
            self.grad += (other.data * self.data**(other.data - 1)) * out.grad
            try:
                other.grad += (out.data * math.log(self.data)) * out.grad
            except ValueError:
                other.grad += 0
        out._backward = _backward
        return out

    def __truediv__(self, other):
        # Implemented as a/b as a*b^-1 to reuse already implemented operations
        return self * other**(-1)

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**(-1)

    def exp(self):
        out = Value._from_operation(math.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        try:
            out = Value._from_operation(math.log(self.data), (self, ), 'log')
        except ValueError:
            out = Value._from_operation(float('-inf'), (self, ), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value._from_operation(math.tanh(self.data), (self, ), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value._from_operation(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Value._from_operation(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')

        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def _build_topo_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    _build_topo_graph(child)
                topo.append(v)

        _build_topo_graph(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __eq__(self, other):
        if not isinstance(other, Value):
            return False
        return self.data == other.data and self.grad == other.grad

    def __gt__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self.data > other.data

    def __lt__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self.data < other.data

    def __ge__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self.data >= other.data

    def __le__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self.data <= other.data

    def __hash__(self):
        return hash(id(self))

    def __copy__(self):
        new = Value(self.data, self.label)
        new.grad = self.grad
        new._children = self._children
        new._operator = self._operator
        new._backward = self._backward
        return new
