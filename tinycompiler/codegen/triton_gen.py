import hashlib
import importlib.util
import operator
import os
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  #FMA explicitly on
import tempfile
import torch
import triton
from tinycompiler.graph import Node


#How each op is written in triton. {0} and {1} are its args, .format() fills em in
TRITON_OPS = {
    operator.add: "{0} + {1}",
    operator.sub: "{0} - {1}",
    operator.mul: "{0} * {1}",
    torch.add: "{0} + {1}",
    torch.sub: "{0} - {1}",
    torch.mul: "{0} * {1}",
    torch.relu: "tl.maximum({0}, 0.0)", # relu's just max(x, 0)
    torch.sigmoid: "tl.sigmoid({0})",
    torch.tanh: "2.0 * tl.sigmoid(2.0 * {0}) - 1.0", # tanh(x) = 2σ(2x) - 1 (for every version of triton)
    torch.exp: "tl.exp({0})",
    torch.neg: "-{0}",
}

DEFAULT_CONFIG = (1024, 4)  # (BLOCK_SIZE, num_warps) when not autotuning

KERNEL_DIR = tempfile.mkdtemp(prefix="tinycompiler_kernels_") # generated kernels end up here
compiled_kernels = {} # source hash -> kernel
kernel_for_node = {} # fused node -> kernel


def generate_source(fused: Node) -> str:
    input_ptrs = [f"in{i}_ptr" for i in range(len(fused.inputs))] # one per VRAM input

    # same for every kernel, only the params change
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        "@triton.jit",
        f"def kernel({', '.join(input_ptrs)}, out_ptr, n, BLOCK_SIZE: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offsets < n",
    ]

    '''Every value gets a variable name, v0, v1, etc
    The external inputs are loaded first, then one line per absorbed node'''
    var_names = {} # node -> its variable in the kernel
    for i, input_node in enumerate(fused.inputs):
        var_names[input_node] = f"v{i}"
        lines.append(f"    v{i} = tl.load({input_ptrs[i]} + offsets, mask=mask)") # don't forget the damn indentation

    for each_node in fused.target:
        if each_node.target not in TRITON_OPS:
            raise NotImplementedError(f"no Triton translation for '{each_node.name}'")

        arg_strings = []
        for arg in each_node.args:
            if isinstance(arg, Node):
                arg_strings.append(var_names[arg])
            else:
                arg_strings.append(repr(arg))  #constant

        var_names[each_node] = f"v{len(var_names)}" # next free name
        expression = TRITON_OPS[each_node.target].format(*arg_strings)
        lines.append(f"    {var_names[each_node]} = {expression}  # {each_node.name}")

    result = var_names[fused.target[-1]] # last node's value is what gets stored
    lines.append(f"    tl.store(out_ptr + offsets, {result}, mask=mask)")
    return "\n".join(lines) + "\n"


def compile_source(source: str):
    '''triton.jit reads a kernel's source from its file, so an execd string won't work.
    Write it to a file named after its hash and import it'''
    key = hashlib.sha256(source.encode()).hexdigest()[:16]
    if key in compiled_kernels: # same source, already done
        return compiled_kernels[key]

    path = os.path.join(KERNEL_DIR, f"kernel_{key}.py")
    with open(path, "w") as f:
        f.write(source)

    # import the file like any other module
    spec = importlib.util.spec_from_file_location(f"kernel_{key}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    compiled_kernels[key] = module.kernel
    return module.kernel


def prepare_inputs(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    first = inputs[0]
    for each_input in inputs:
        if each_input.shape != first.shape or each_input.dtype != first.dtype:
            raise NotImplementedError("fused kernels need inputs of the same shape and dtype, no broadcasting yet")

    return [each_input.contiguous() for each_input in inputs] # kernel assumes the elements sit side by side in memory


def get_kernel(fused: Node):
    # generate and compile once per fused node, not on every call
    if fused not in kernel_for_node:
        kernel_for_node[fused] = compile_source(generate_source(fused))

    return kernel_for_node[fused]


def launch(fused: Node, inputs: list[torch.Tensor], out: torch.Tensor, block_size: int, num_warps: int):
    n = out.numel()
    grid = (triton.cdiv(n, block_size),) # no. of blocks, rounded up
    get_kernel(fused)[grid](*inputs, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)


def run_fused(fused: Node, inputs: list[torch.Tensor], config: tuple[int, int] | None = None) -> torch.Tensor:
    inputs = prepare_inputs(inputs)
    out = torch.empty_like(inputs[0]) # empty, the kernel fills it
    block_size, num_warps = config if config is not None else DEFAULT_CONFIG
    launch(fused, inputs, out, block_size, num_warps)
    return out