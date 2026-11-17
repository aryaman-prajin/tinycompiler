import torch
import torch.nn as nn
import pytest
from tinycompiler.capture import capture


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(4, 8)
        self.layer_2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        return x


def test_capture_tiny_mlp():
    graph = capture(TinyMLP())

    # same nodes, same order, as FX recorded them
    names = [node.name for node in graph.nodes]
    assert names == ["x", "layer_1", "relu", "layer_2", "output"]

    x, layer_1, relu, layer_2, output = graph.nodes

    # op and target copied across
    assert x.op == "placeholder"
    assert relu.op == "call_function"
    assert relu.target is torch.relu
    assert output.op == "output"

    # inputs point at OUR nodes, not FX's
    assert relu.inputs == [layer_1]
    assert output.inputs == [layer_2]

    # users were rebuilt by add_node
    assert x.users == [layer_1]
    assert output.users == []


class TwoOutputs(nn.Module):
    def forward(self, x, c):
        return x + c, torch.relu(x)


def test_multiple_outputs_are_rejected_loudly():
    with pytest.raises(NotImplementedError):
        capture(TwoOutputs())