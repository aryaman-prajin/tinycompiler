from tinycompiler.graph import Graph


def eliminate_dead_code(graph: Graph) -> Graph:
    while True:
        dead_nodes = [
            node for node in graph.nodes 
            if not node.users and node.op not in ("output", "placeholder") # includes floating placeholders
        ]
        for node in dead_nodes:
            graph.remove_node(node)
        if not dead_nodes:  # nothing was eliminated
            break
    return graph