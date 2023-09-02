from .autograd import Value
from graphviz import Digraph  # Download at https://graphviz.org/download/
import numpy as np


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        if hasattr(n, 'label') and n.label:
            dot.node(name=str(id(n)),
                     label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                     shape='record')

        else:
            dot.node(name=str(id(n)),
                     label="{ data %.4f | grad %.4f }" % (n.data, n.grad),
                     shape='record')
        if n._operator:
            dot.node(name=str(id(n)) + n._operator, label=n._operator)
            dot.edge(str(id(n)) + n._operator, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._operator)

    return dot


def argmax(values):
    """Returns the index of the maximum value in the list.

    Used to convert one-hot encoded vectors to class labels.
    """
    if isinstance(values[0], Value):
        values = [o.data for o in values]
    return max(enumerate(values), key=lambda x: x[1])[0]


def np_array_to_list_of_values(array):
    """Converts a numpy array to a list of Value objects."""
    if len(array.shape) == 1:
        return [Value(x) for x in array]
    elif len(array.shape) > 1:  # works recursively
        return [np_array_to_list_of_values(x) for x in array]


def list_of_values_to_np_array(values):
    """Converts a list of Value objects to a numpy array."""
    if isinstance(values[0], Value):
        return np.array([v.data for v in values])
    elif isinstance(values[0], list):  # works recursively
        return np.array([list_of_values_to_np_array(x) for x in values])
