from tinycompiler.graph import Node, Graph
from tinycompiler.passes.dce import eliminate_dead_code


def test_dead_node_is_removed():
    # x -> relu -> output
    # z -> square (dead, nothing uses square, but z is a placeholder)
    x = Node("placeholder", "x")
    z = Node("placeholder", "z")
    relu = Node("call_function", "relu", inputs=[x])
    square = Node("call_function", "square", inputs=[z])
    output = Node("output", "output", inputs=[relu])

    g = Graph()
    g.add_node(x)
    g.add_node(z)
    g.add_node(relu)
    g.add_node(square)
    g.add_node(output)

    eliminate_dead_code(g)

    assert square not in g.nodes, "square should have been eliminated"
    assert z in g.nodes, "z is a model input and must survive even when unused"
    assert z.users == [], "z should have no users left after square was removed"    
    assert relu in g.nodes, "relu should survive"
    assert x in g.nodes, "x should survive as it feeds into relu"
    assert output in g.nodes, "output should always survive"


def test_no_dead_nodes():
    # Every node feeds into the output, nothing should be removed
    x = Node("placeholder", "x")
    relu = Node("call_function", "relu", inputs=[x])
    output = Node("output", "output", inputs=[relu])

    g = Graph()
    g.add_node(x)
    g.add_node(relu)
    g.add_node(output)

    before = len(g.nodes)
    eliminate_dead_code(g)

    assert len(g.nodes) == before, "no nodes should have been removed"