from graphviz import Digraph  # Download at https://graphviz.org/download/


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
