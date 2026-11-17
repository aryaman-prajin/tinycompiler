# Completely vibecoded. Looks fine on first glance tho

import os

import pytest
import torch
import torch.nn as nn

from tinycompiler.capture import capture
from tinycompiler.codegen.triton_gen import generate_source
from tinycompiler.interpreter import run
from tinycompiler.passes.fusion import fuse_elementwise

#Kernels need a GPU, or TRITON_INTERPRET=1, which runs them slowly on the CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    DEVICE = None

needs_kernels = pytest.mark.skipif(DEVICE is None, reason="no GPU and TRITON_INTERPRET is not set")


class AddRelu(nn.Module):
    def forward(self, x, c):
        return torch.relu(x + c)


class WithConstants(nn.Module):
    def forward(self, x, c):
        return torch.sigmoid(torch.relu(x + c) * 2 - 1)


class EveryOp(nn.Module):
    def forward(self, x, c):
        return torch.neg(torch.exp(torch.tanh(x * c) - c)) + x


class MLPWithFusedTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(16, 16)

    def forward(self, x, c):
        return torch.sigmoid(torch.relu(self.layer(x) + c) * 0.5)


def get_fused(model):
    graph = fuse_elementwise(capture(model))
    return [node for node in graph.nodes if node.op == "fused"][0]


def test_source_for_add_relu():
    source = generate_source(get_fused(AddRelu()))

    assert "tl.load(in0_ptr + offsets, mask=mask)" in source
    assert "tl.load(in1_ptr + offsets, mask=mask)" in source
    assert "v2 = v0 + v1  # add" in source
    assert "v3 = tl.maximum(v2, 0.0)  # relu" in source
    assert "tl.store(out_ptr + offsets, v3, mask=mask)" in source


def test_constants_are_written_into_the_source():
    source = generate_source(get_fused(WithConstants()))
    assert "v3 * 2  # mul" in source
    assert "v4 - 1  # sub" in source


@needs_kernels
@pytest.mark.parametrize("model_class", [AddRelu, WithConstants, EveryOp, MLPWithFusedTail])
@pytest.mark.parametrize("shape", [(16,), (1000, 16), (3, 5, 16)])
def test_triton_matches_reference(model_class, shape):
    torch.manual_seed(0)
    model = model_class().to(DEVICE)
    x = torch.randn(shape, device=DEVICE)
    c = torch.randn(shape, device=DEVICE)
    graph = fuse_elementwise(capture(model))

    expected = run(graph, model, x, c)
    actual = run(graph, model, x, c, use_triton=True)

    torch.testing.assert_close(actual, expected)


@needs_kernels
def test_broadcasting_is_rejected_loudly():
    model = AddRelu()
    graph = fuse_elementwise(capture(model))
    x = torch.randn(4, 16, device=DEVICE)
    c = torch.randn(16, device=DEVICE)
    
    with pytest.raises(NotImplementedError):
        run(graph, model, x, c, use_triton=True)