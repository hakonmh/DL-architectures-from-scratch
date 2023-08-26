import math


class Value:

    def __init__(self, data, label=''):
        self.data = data
        self._children = set()
        self._operator = ''
        self.grad = 0
        self.label = label
        self._backward = lambda: None

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
        if hasattr(self, 'label') and self.label:
            return f"Value({self.data:.4f}, grad={self.grad:.4f}, label={self.label})"
        else:
            return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # Convert to Value if needed
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
        other = other if isinstance(other, Value) else Value(other)
        out = Value._from_operation(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other.data if isinstance(other, Value) else other
        out = Value._from_operation(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
            # other.grad += (math.log(self.data) * self.data**other.data) * out.grad
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
        out = Value._from_operation(math.log(self.data), (self, ), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value._from_operation(tanh, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - tanh**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value._from_operation(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
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
