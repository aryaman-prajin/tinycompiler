import operator
import torch
import torch.nn as nn
from tinycompiler.capture import capture
from tinycompiler.passes.fusion import fuse_elementwise


def find(graph, name):
    for node in graph.nodes:
        if node.name == name:
            return node
    raise ValueError(f"no node called '{name}'")


class AddRelu(nn.Module):
    def forward(self, x, c):
        return torch.relu(x + c)

def test_add_relu_becomes_one_node():
    graph = fuse_elementwise(capture(AddRelu()))

    assert [node.op for node in graph.nodes] == ["placeholder", "placeholder", "fused", "output"]
    x, c, fused, output = graph.nodes

    assert [node.target for node in fused.target] == [operator.add, torch.relu]
    assert fused.inputs == [x, c]
    assert output.inputs == [fused]
    assert fused.users == [output]
    assert x.users == [fused]
    assert c.users == [fused]



class LongChain(nn.Module):
    def forward(self, x, c):
        return torch.sigmoid(torch.relu(x + c) * c)


def test_chain_of_four_collapses():
    graph = fuse_elementwise(capture(LongChain()))
    fused = find(graph, "fused_add_relu_mul_sigmoid")
    add, relu, mul, sigmoid = fused.target

    assert [node.target for node in fused.target] == [operator.add, torch.relu, operator.mul, torch.sigmoid]
    assert [node.name for node in fused.inputs] == ["x", "c"]
    assert relu.inputs == [add]
    assert [node.name for node in mul.inputs] == ["relu", "c"]
    assert sigmoid.inputs == [mul]


class Square(nn.Module):
    def forward(self, x, c):
        t = x + c
        return t * t

def test_one_consumer_using_a_value_twice_still_fuses():
    graph = fuse_elementwise(capture(Square()))
    assert [node.op for node in graph.nodes] == ["placeholder", "placeholder", "fused", "output"]


class TwoUsersSameGroup(nn.Module):
    def forward(self, x, c):
        t = x + c
        y = torch.relu(t)
        z = t * c
        return y + z


def test_two_users_that_end_up_in_one_group_still_fuse():
    graph = fuse_elementwise(capture(TwoUsersSameGroup()))
    assert [node.op for node in graph.nodes] == ["placeholder", "placeholder", "fused", "output"]


class TwoUsersSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x, c):
        t = x + c
        y = torch.relu(t)
        z = self.layer(t)
        return y + z

def test_value_needed_by_a_matmul_is_not_fused_away():
    graph = fuse_elementwise(capture(TwoUsersSplit()))
    add = find(graph, "add")
    assert add.op == "call_function"
    assert len(add.users) == 2


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(4, 8)
        self.layer_2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.layer_2(torch.relu(self.layer_1(x)))


def test_linear_layers_block_fusion():
    graph = fuse_elementwise(capture(TinyMLP()))
    assert [node.name for node in graph.nodes] == ["x", "layer_1", "relu", "layer_2", "output"]