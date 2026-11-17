import pytest
import torch
import torch.nn as nn

from tinycompiler.capture import capture
from tinycompiler.interpreter import run
from tinycompiler.passes.dce import eliminate_dead_code
from tinycompiler.passes.fusion import fuse_elementwise


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(4, 8)
        self.layer_2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.layer_2(torch.relu(self.layer_1(x)))


class WithConstants(nn.Module):
    def forward(self, x, c):
        return torch.sigmoid(torch.relu(x + c) * 2 - 1)


class TwoUsersSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x, c):
        t = x + c
        return torch.relu(t) + self.layer(t)


class UsesKwargs(nn.Module):
    def forward(self, x, c):
        return torch.add(x, c, alpha=2)


class UsesMethod(nn.Module):
    def forward(self, x):
        return x.relu()


CASES = [
    (TinyMLP, 1),
    (WithConstants, 2),
    (TwoUsersSplit, 2),
]


@pytest.mark.parametrize("model_class, n_inputs", CASES)
def test_captured_graph_matches_pytorch(model_class, n_inputs):
    torch.manual_seed(0)
    model = model_class()
    inputs = [torch.randn(3, 4) for _ in range(n_inputs)]

    torch.testing.assert_close(run(capture(model), model, *inputs), model(*inputs))


@pytest.mark.parametrize("model_class, n_inputs", CASES)
def test_passes_do_not_change_the_answer(model_class, n_inputs):
    torch.manual_seed(0)
    model = model_class()
    inputs = [torch.randn(3, 4) for _ in range(n_inputs)]

    graph = eliminate_dead_code(fuse_elementwise(capture(model)))

    torch.testing.assert_close(run(graph, model, *inputs), model(*inputs))


def test_constants_survive_capture():
    graph = capture(WithConstants())
    mul = [node for node in graph.nodes if node.name == "mul"][0]
    assert mul.args[1] == 2
    assert len(mul.inputs) == 1


def test_kwargs_are_rejected_loudly():
    with pytest.raises(NotImplementedError):
        capture(UsesKwargs())


def test_unsupported_ops_are_rejected_loudly():
    with pytest.raises(NotImplementedError):
        run(capture(UsesMethod()), UsesMethod(), torch.randn(3))