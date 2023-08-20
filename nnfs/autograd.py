import math


class Value:

    def __init__(self, data):
        self.data = data
        self._children = set()
        self._operator = ''
        self.grad = 0
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
        return f"Value({self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # Convert to Value if needed
        out = Value._from_operation(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        self._backward = _backward
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
        self._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value._from_operation(self.data ** other.data, (self, other), '**')

        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        # Implemented as a/b as a*b^-1 to reuse already implemented operations
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**-1

    def exp(self):
        out = Value._from_operation(math.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = self._build_topo_graph()
        # dy/dy is always equal to 1, we need to set it to 1 before backpropagating
        # the gradient through the graph
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def _build_topo_graph(self):
        # Builds a topological graph of all children of the current Value object
        topo = []
        visited = set()
        if self not in visited:
            visited.add(self)
            for child in self._children:
                topo.extend(child._build_topo_graph())
            topo.append(self)
        return topo
