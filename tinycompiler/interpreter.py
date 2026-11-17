from typing import Any
import torch
import torch.nn as nn
from tinycompiler.codegen.autotune import best_config
from tinycompiler.codegen.triton_gen import run_fused
from tinycompiler.graph import Node, Graph


def get_value(arg: Any, values: dict) -> Any:
    # result of a node, else it's a constant
    if isinstance(arg, Node):
        return values[arg]
    return arg


def run_node(node: Node, model: nn.Module, values: dict, use_triton: bool = False, autotune: bool = False) -> Any:
    args = [get_value(arg, values) for arg in node.args]

    if node.op == "call_function":
        return node.target(*args)

    if node.op == "call_module":
        # the target's the layer's name such as 'layer_1', so fetch the real layer from the model
        layer = model.get_submodule(node.target)
        return layer(*args)

    if node.op == "fused":
        if use_triton:
            # one generated kernel for the whole group
            inputs = [values[input_node] for input_node in node.inputs]
            config = best_config(node, inputs) if autotune else None
            return run_fused(node, inputs, config)

        '''Run the absorbed nodes one by one.
        The fast version is the use_triton branch above'''
        for each_node in node.target:
            values[each_node] = run_node(each_node, model, values)
        return values[node.target[-1]]
    raise NotImplementedError(f"interpreter can't run '{node.name}' with op '{node.op}'")


'''Walks the whole graph every call and checks each op, costs ~ 3 µs.
torch.compile just generates a straight-line wrapper instead (kernel A, then B, return).
Worth doing here later to kill the overhead'''
def run(graph: Graph, model: nn.Module, *inputs: torch.Tensor, use_triton: bool = False, autotune: bool = False) -> torch.Tensor:
    values = {}
    next_input = 0

    for node in graph.nodes:
        if node.op == "placeholder":
            values[node] = inputs[next_input]
            next_input += 1
        elif node.op == "output":
            return get_value(node.args[0], values)
        else:
            values[node] = run_node(node, model, values, use_triton, autotune)

    raise ValueError("graph has no output node")