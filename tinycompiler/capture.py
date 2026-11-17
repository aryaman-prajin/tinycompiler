import torch.fx as fx
import torch.nn as nn
from tinycompiler.graph import Node, Graph


def capture(model: nn.Module) -> Graph:
    traced = fx.symbolic_trace(model) # proxy tracing
    graph = Graph()
    node_map = {} # convert from fx objs to our objs

    for fx_node in traced.graph.nodes:
        if fx_node.kwargs: # just don't need kwarg support for now. Worth considering later
            raise NotImplementedError(f"'{fx_node.name}' uses keyword arguments, which capture doesn't handle yet")

        args = [] # nodes, constants
        inputs = [] # les nodes seulement
        for arg in fx_node.args:
            # tuples are treated as constants, which leads to nasty bugs. Worth considering later
            if isinstance(arg, (tuple, list)) and any(isinstance(each_arg, fx.Node) for each_arg in arg):
                raise NotImplementedError(f"'{fx_node.name}' \
                has nodes inside a tuple, like multiple o/ps, which capture doesn't handle yet")
            if isinstance(arg, fx.Node):
                args.append(node_map[arg])
                inputs.append(node_map[arg])
            else:
                args.append(arg) # constants case

        node = Node(fx_node.op, 
                    fx_node.name, 
                    target=fx_node.target, 
                    inputs=inputs, 
                    args=args) # the first 3 can be copied from fx

        graph.add_node(node)
        node_map[fx_node] = node

    return graph