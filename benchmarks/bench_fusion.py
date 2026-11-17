# Benchmarking mostly vibecoded too


import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # no screen in WSL, just save the image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import triton.testing

from tinycompiler.capture import capture
from tinycompiler.codegen.triton_gen import run_fused
from tinycompiler.interpreter import run
from tinycompiler.passes.fusion import fuse_elementwise

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class AddRelu(nn.Module):
    def forward(self, x, c):
        return torch.relu(x + c)


class LongChain(nn.Module):
    def forward(self, x, c):
        return torch.sigmoid(torch.relu(x + c) * 2 - 1)


'''Bytes each version must move, per element, 4 bytes per float32.
Eager writes and re-reads every intermediate. Fused reads x and c once and writes y once.'''
MODELS = {
    "add_relu": (AddRelu, 4 * (3 + 2), 4 * 3),
    "long_chain": (LongChain, 4 * (3 + 2 + 2 + 2 + 2), 4 * 3),
}


def time_ms(fn) -> float:
    # warm-up, sync, repeats and an L2 flush between runs are all done by do_bench
    return triton.testing.do_bench(fn, return_mode="median")


def bench_one(name: str, n: int) -> dict:
    model_class, eager_bytes, fused_bytes = MODELS[name]
    model = model_class().cuda()
    x = torch.randn(n, device="cuda")
    c = torch.randn(n, device="cuda")

    graph = fuse_elementwise(capture(model))
    fused = [node for node in graph.nodes if node.op == "fused"][0]

    torch._dynamo.reset()  # fresh torch.compile for every size, otherwise it hits its recompile limit
    compiled = torch.compile(model)

    torch.testing.assert_close(run(graph, model, x, c, use_triton=True), model(x, c))
    torch.testing.assert_close(compiled(x, c), model(x, c))

    eager_ms = time_ms(lambda: model(x, c))
    compile_ms = time_ms(lambda: compiled(x, c))
    ours_ms = time_ms(lambda: run(graph, model, x, c, use_triton=True))
    kernel_ms = time_ms(lambda: run_fused(fused, [x, c]))
    copy_ms = time_ms(lambda: x.clone())

    result = {
        "model": name,
        "log2_n": n.bit_length() - 1,
        "n": n,
        "eager_ms": eager_ms,
        "compile_ms": compile_ms,
        "ours_ms": ours_ms,
        "kernel_ms": kernel_ms,
        "speedup_ours": eager_ms / ours_ms,
        "speedup_kernel": eager_ms / kernel_ms,
        "speedup_compile": eager_ms / compile_ms,
        "ceiling": eager_bytes / fused_bytes,
        "kernel_gbps": fused_bytes * n / (kernel_ms * 1e6),
        "copy_gbps": 8 * n / (copy_ms * 1e6),  # clone reads 4n and writes 4n bytes
    }

    del x, c, model, compiled
    torch.cuda.empty_cache()
    return result


def plot(rows: list[dict]):
    for name in MODELS:
        model_rows = [row for row in rows if row["model"] == name]
        sizes = [row["log2_n"] for row in model_rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

        ax1.plot(sizes, [row["speedup_ours"] for row in model_rows], "o-", label="tinycompiler, end to end")
        ax1.plot(sizes, [row["speedup_kernel"] for row in model_rows], "o--", label="tinycompiler, kernel only")
        ax1.plot(sizes, [row["speedup_compile"] for row in model_rows], "s-", label="torch.compile")
        ax1.axhline(model_rows[0]["ceiling"], color="grey", linestyle=":", label=f"bytes ceiling ({model_rows[0]['ceiling']:.2f}x)")
        ax1.axhline(1.0, color="black", linewidth=0.8)
        ax1.set_xlabel("log2(n)")
        ax1.set_ylabel("speedup over eager PyTorch")
        ax1.set_title(f"{name}: speedup vs size")
        ax1.legend()

        ax2.plot(sizes, [row["kernel_gbps"] for row in model_rows], "o-", label="fused kernel")
        ax2.plot(sizes, [row["copy_gbps"] for row in model_rows], "s--", label="torch clone, for reference")
        ax2.set_xlabel("log2(n)")
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("achieved bandwidth (GB/s)")
        ax2.set_title(f"{name}: memory bandwidth")
        ax2.legend()

        for ax in (ax1, ax2):
            ax.set_xticks(sizes)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, f"{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-log2", type=int, default=27)  # 2^27 is the most that fits in 4 GB of VRAM
    parser.add_argument("--min-log2", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("benchmarks need a GPU")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, Triton {triton.__version__}")

    rows = []
    for name in MODELS:
        for log2_n in range(args.min_log2, args.max_log2 + 1):
            row = bench_one(name, 2 ** log2_n)
            rows.append(row)
            print(
                f"{name:11s} 2^{log2_n:<2d}  eager {row['eager_ms']:8.4f} ms  "
                f"ours {row['speedup_ours']:5.2f}x  kernel {row['speedup_kernel']:5.2f}x  "
                f"compile {row['speedup_compile']:5.2f}x  kernel {row['kernel_gbps']:6.1f} GB/s"
            )

    csv_path = os.path.join(RESULTS_DIR, "fusion.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")

    plot(rows)


if __name__ == "__main__":
    main()