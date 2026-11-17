import torch
import triton.testing

from tinycompiler.codegen.triton_gen import DEFAULT_CONFIG, launch, prepare_inputs
from tinycompiler.graph import Node

# (BLOCK_SIZE, num_warps) pairs to try
CONFIGS = [
    (block_size, num_warps)
    for block_size in (256, 512, 1024, 2048, 4096)
    for num_warps in (2, 4, 8)
]

best_configs = {}


def size_bucket(n: int) -> int:
    # next power of two, so n = 1000 and n = 1024 share one tuning result
    return 1 << (n - 1).bit_length()


def best_config(fused: Node, inputs: list[torch.Tensor]) -> tuple[int, int]:
    inputs = prepare_inputs(inputs)
    key = (fused, size_bucket(inputs[0].numel()))
    if key in best_configs:
        return best_configs[key]

    # Time every config once and keep the fastest.
    # Short warm-up and repeat times, since this runs 15 times per new size.
    out = torch.empty_like(inputs[0])
    timings = {}
    for block_size, num_warps in CONFIGS:
        timings[(block_size, num_warps)] = triton.testing.do_bench(
            lambda: launch(fused, inputs, out, block_size, num_warps),
            warmup=5,
            rep=20,
            return_mode="median",
        )

    # Only leave the default if something beats it clearly, otherwise noise picks the winner
    fastest = min(timings, key=timings.get)
    best_configs[key] = fastest if timings[fastest] < 0.97 * timings[DEFAULT_CONFIG] else DEFAULT_CONFIG
    return best_configs[key]