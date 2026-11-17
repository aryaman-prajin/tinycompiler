import os

import pytest
import torch
import torch.nn as nn

from tinycompiler.capture import capture
from tinycompiler.codegen.autotune import CONFIGS, best_config, best_configs, size_bucket
from tinycompiler.codegen.triton_gen import run_fused
from tinycompiler.interpreter import run
from tinycompiler.passes.fusion import fuse_elementwise

if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    DEVICE = None

needs_kernels = pytest.mark.skipif(DEVICE is None, reason="no GPU and TRITON_INTERPRET is not set")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="timing configs needs a real GPU")


class LongChain(nn.Module):
    def forward(self, x, c):
        return torch.sigmoid(torch.relu(x + c) * 2 - 1)


def test_size_bucket():
    assert size_bucket(1) == 1
    assert size_bucket(1000) == 1024
    assert size_bucket(1024) == 1024
    assert size_bucket(1025) == 2048


@needs_kernels
@pytest.mark.parametrize("config", CONFIGS)
def test_every_config_gives_the_right_answer(config):
    torch.manual_seed(0)
    model = LongChain()
    graph = fuse_elementwise(capture(model))
    fused = [node for node in graph.nodes if node.op == "fused"][0]
    x = torch.randn(5000, device=DEVICE)
    c = torch.randn(5000, device=DEVICE)

    torch.testing.assert_close(run_fused(fused, [x, c], config), model(x, c))


@needs_gpu
def test_autotune_picks_a_config_and_remembers_it():
    torch.manual_seed(0)
    model = LongChain()
    graph = fuse_elementwise(capture(model))
    fused = [node for node in graph.nodes if node.op == "fused"][0]
    x = torch.randn(5000, device="cuda")
    c = torch.randn(5000, device="cuda")

    first = best_config(fused, [x, c])
    assert first in CONFIGS
    assert (fused, 8192) in best_configs
    assert best_config(fused, [x, c]) is first

    torch.testing.assert_close(run(graph, model, x, c, use_triton=True, autotune=True), model(x, c))