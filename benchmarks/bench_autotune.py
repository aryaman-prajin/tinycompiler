import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import triton
import triton.testing

from bench_fusion import MODELS, RESULTS_DIR
from tinycompiler.capture import capture
from tinycompiler.codegen.autotune import best_config
from tinycompiler.codegen.triton_gen import DEFAULT_CONFIG, run_fused
from tinycompiler.passes.fusion import fuse_elementwise


def time_ms(fn) -> float:
    return triton.testing.do_bench(fn, return_mode="median")


def bench_one(name: str, n: int) -> dict:
    model = MODELS[name][0]().cuda()
    x = torch.randn(n, device="cuda")
    c = torch.randn(n, device="cuda")
    graph = fuse_elementwise(capture(model))
    fused = [node for node in graph.nodes if node.op == "fused"][0]

    tuned = best_config(fused, [x, c])
    torch.testing.assert_close(run_fused(fused, [x, c], tuned), model(x, c))

    fixed_ms = time_ms(lambda: run_fused(fused, [x, c], DEFAULT_CONFIG))
    tuned_ms = time_ms(lambda: run_fused(fused, [x, c], tuned))

    del x, c, model
    torch.cuda.empty_cache()
    return {
        "model": name,
        "log2_n": n.bit_length() - 1,
        "n": n,
        "fixed_ms": fixed_ms,
        "tuned_ms": tuned_ms,
        "speedup": fixed_ms / tuned_ms,
        "block_size": tuned[0],
        "num_warps": tuned[1],
    }


def plot(rows: list[dict]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for name in MODELS:
        model_rows = [row for row in rows if row["model"] == name]
        sizes = [row["log2_n"] for row in model_rows]
        ax1.plot(sizes, [row["speedup"] for row in model_rows], "o-", label=name)
        ax2.plot(sizes, [row["block_size"] for row in model_rows], "o-", label=name)
        for row in model_rows:
            ax2.annotate(f"{row['num_warps']}w", (row["log2_n"], row["block_size"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    ax1.axhline(1.0, color="black", linewidth=0.8)
    ax1.set_ylabel(f"speedup of autotuned over fixed {DEFAULT_CONFIG}")
    ax1.set_title("what autotuning buys")

    ax2.set_yscale("log", base=2)
    ax2.set_ylabel("chosen BLOCK_SIZE (label = num_warps)")
    ax2.set_title("what the autotuner picked")

    for ax in (ax1, ax2):
        ax.set_xlabel("log2(n)")
        ax.set_xticks(sizes)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "autotune.png")
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
                f"{name:11s} 2^{log2_n:<2d}  fixed {row['fixed_ms']:8.4f} ms  tuned {row['tuned_ms']:8.4f} ms  "
                f"speedup {row['speedup']:5.2f}x  picked BLOCK_SIZE {row['block_size']:4d}, {row['num_warps']} warps"
            )

    csv_path = os.path.join(RESULTS_DIR, "autotune.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")

    plot(rows)


if __name__ == "__main__":
    main()