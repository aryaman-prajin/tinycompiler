import argparse
import csv
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import triton

from bench_fusion import MODELS, RESULTS_DIR
from tinycompiler.capture import capture
from tinycompiler.codegen.triton_gen import run_fused
from tinycompiler.interpreter import run
from tinycompiler.passes.fusion import fuse_elementwise


def wall_us(fn, n_calls: int) -> float:
    '''What a user actually waits per call: Python, launches and GPU work back to back.
    No cache flush in between, so CPU overhead can't hide behind the GPU.'''
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    trials = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(n_calls):
            fn()
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - start) / n_calls * 1e6)
    return statistics.median(trials)


def bench_one(name: str, n: int) -> dict:
    model_class = MODELS[name][0]
    model = model_class().cuda()
    x = torch.randn(n, device="cuda")
    c = torch.randn(n, device="cuda")

    graph = fuse_elementwise(capture(model))
    fused = [node for node in graph.nodes if node.op == "fused"][0]

    torch._dynamo.reset()
    compiled = torch.compile(model)
    compiled(x, c)

    n_calls = max(10, min(1000, 2 ** 24 // n))  # about the same total time at every size

    eager_us = wall_us(lambda: model(x, c), n_calls)
    compile_us = wall_us(lambda: compiled(x, c), n_calls)
    ours_us = wall_us(lambda: run(graph, model, x, c, use_triton=True), n_calls)
    kernel_us = wall_us(lambda: run_fused(fused, [x, c]), n_calls)

    del x, c, model, compiled
    torch.cuda.empty_cache()
    return {
        "model": name,
        "log2_n": n.bit_length() - 1,
        "n": n,
        "eager_us": eager_us,
        "compile_us": compile_us,
        "ours_us": ours_us,
        "kernel_us": kernel_us,
    }


def plot(rows: list[dict]):
    labels = {
        "eager_us": "eager PyTorch",
        "compile_us": "torch.compile",
        "ours_us": "tinycompiler, end to end",
        "kernel_us": "tinycompiler, kernel only",
    }
    colours = {"eager_us": "C0", "compile_us": "C1", "ours_us": "C2", "kernel_us": "C3"}  # same colour in both panels

    for name in MODELS:
        model_rows = [row for row in rows if row["model"] == name]
        sizes = [row["log2_n"] for row in model_rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

        for key, label in labels.items():
            ax1.plot(sizes, [row[key] for row in model_rows], "o-", color=colours[key], label=label)
        ax1.set_yscale("log")
        ax1.set_xlabel("log2(n)")
        ax1.set_ylabel("wall-clock time per call (µs)")
        ax1.set_title(f"{name}: what a caller waits")
        ax1.legend()

        for key, label in labels.items():
            if key == "eager_us":
                continue
            ax2.plot(sizes, [row["eager_us"] / row[key] for row in model_rows], "o-", color=colours[key], label=label)
        ax2.axhline(1.0, color="black", linewidth=0.8)
        ax2.set_xlabel("log2(n)")
        ax2.set_ylabel("speedup over eager PyTorch")
        ax2.set_title(f"{name}: wall-clock speedup")
        ax2.legend()

        for ax in (ax1, ax2):
            ax.set_xticks(sizes)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, f"latency_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-log2", type=int, default=27)
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
                f"{name:11s} 2^{log2_n:<2d}  eager {row['eager_us']:9.1f} us  compile {row['compile_us']:9.1f} us  "
                f"ours {row['ours_us']:9.1f} us  kernel only {row['kernel_us']:9.1f} us"
            )

    csv_path = os.path.join(RESULTS_DIR, "latency.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")

    plot(rows)


if __name__ == "__main__":
    main()