# Grouped Sparse Attention — MASE Transform Pass
## ADLS Coursework Project Spec

---

## Project goal

Implement a grouped sparse attention CUDA/Triton kernel and integrate it as a
transform pass into the [MASE framework](https://github.com/DeepWok/mase).
The primary target model is **Chronos2**, which uses grouped self-attention
where tokens only attend within their group. The pass should be general enough
to work on any model that exposes a group structure.

Deliverable: a Pull Request into `DeepWok/mase` containing the transform pass,
kernels, benchmarks, and tests.

---

## Background

### What is grouped self-attention?

In Chronos2, the input sequence is divided into variable-size groups. Tokens
attend only to other tokens in the same group. The score matrix is therefore
block-diagonal — each diagonal block is a dense attention sub-matrix for one
group, and all off-diagonal entries are always zero.

This means the effective complexity is O(T · G) where G is the average group
size, rather than O(T²) for full attention.

### Why a custom kernel?

PyTorch's `scaled_dot_product_attention` with a boolean mask still allocates
and reads/writes the full T×T score matrix. A fused kernel operating at the
tile level can skip entire off-diagonal blocks entirely — never touching that
memory. The gain is in memory bandwidth, not just FLOPs.

### How MASE works (relevant parts)

MASE wraps a PyTorch model in a `MaseGraph` (a Torch FX graph). Optimisations
are applied as "passes" — functions that walk the graph and modify it. Passes
are chained inside a `CompressionPipeline`. The project adds a new pass:
`grouped_sparse_attn_transform_pass`.

```python
from chop.pipelines import CompressionPipeline
from chop import MaseGraph

mg = MaseGraph(model)
pipe = CompressionPipeline()
mg, _ = pipe(mg, pass_args={
    "grouped_sparse_attn_transform_pass": {
        "by": "type",
        "default": {"config": {"name": None}},
        "grouped_self_attention": {
            "config": {
                "kernel_variant": "auto",
                "window_size": None,
            }
        }
    }
})
```

---

## Repository structure

```
src/chop/passes/graph/transforms/attention/
├── __init__.py
├── grouped_sparse_attn_pass.py     # MASE transform pass entry point
├── model_analyser.py               # detect group structure from any model
├── kernel_dispatcher.py            # select kernel variant based on group sizes
└── kernels/
    ├── grouped_attn_small.cu       # warp-level, groups <= 32
    ├── grouped_attn_medium.cu      # shared-mem tiled, groups 33–256
    ├── grouped_attn_large.cu       # full tiled matmul, groups > 256
    ├── grouped_attn_triton.py      # Triton version (preferred integration path)
    └── setup.py

test/passes/graph/transforms/attention/
├── test_correctness.py             # output vs PyTorch reference
├── test_kernel_variants.py         # unit test each kernel
├── test_transform_pass.py          # end-to-end MASE pass test
└── test_benchmarks.py              # latency / memory benchmarks

docs/
└── grouped_sparse_attn.md          # explains the pass, usage, results
```

---

## Implementation plan

### Phase 1 — kernel (core contribution)

Write the grouped sparse attention CUDA kernel. The kernel takes:

- `Q, K, V` tensors of shape `[B, T, d]`
- `group_offsets` — start token index for each group, shape `[num_groups]`
- `group_sizes` — number of tokens per group, shape `[num_groups]`

And outputs `out` of shape `[B, T, d]`.

The key optimisation is a **tile-level early exit**: before doing any
computation, each CUDA block checks whether its tile falls entirely outside
every query's group. If so, all threads return immediately.

```cpp
// early exit: this tile is entirely outside all group windows
int tile_key_min   = blockIdx.x * BLOCK_SIZE;
int tile_key_max   = tile_key_min + BLOCK_SIZE - 1;
int tile_query_min = blockIdx.y * BLOCK_SIZE;
int tile_query_max = tile_query_min + BLOCK_SIZE - 1;

// load group bounds for the queries in this tile from constant memory
int g_start = group_start_for_query[tile_query_min / BLOCK_SIZE];
int g_end   = group_end_for_query[tile_query_max / BLOCK_SIZE];

if (tile_key_max < g_start || tile_key_min > g_end) return;
```

For tiles that do overlap, compute attention normally (tiled QKᵀ → scale →
softmax → weighted sum of V). Within a partially overlapping tile, mask
individual elements to -1e9 where `col` is outside the query's group.

**Three kernel variants by group size:**

| Variant | Condition | Key technique |
|---|---|---|
| `small` | max group size ≤ 32 | One warp per group. Softmax via `__shfl_down_sync`. No shared mem needed. |
| `medium` | max group size ≤ 256 | Shared-mem tiled QKᵀ. Standard `__syncthreads` barrier. |
| `large` | max group size > 256 | Full tiled matmul with double-buffered async loads (`cuda::memcpy_async`). |

**Alternative: write in Triton instead of CUDA.**

MASE already uses Triton for kernel fusion (Tutorial 9). A Triton kernel is
more maintainable and integrates more naturally. Triton also autotunes block
sizes for the target GPU automatically, removing manual tuning. Consider
implementing the medium/large variants in Triton and keeping the small variant
in CUDA for the warp-level intrinsics.

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64}, num_stages=3),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=4),
    ],
    key=["T", "d", "max_group_size"],
)
@triton.jit
def grouped_sparse_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    group_offsets_ptr, group_sizes_ptr,
    stride_b, stride_t, stride_d,
    B, T, d,
    BLOCK_SIZE: tl.constexpr,
):
    ...
```

### Phase 2 — model analyser

A module that inspects any PyTorch model and determines whether a grouped
attention kernel can be applied to each attention layer.

```python
@dataclass
class GroupAttentionInfo:
    layer_name: str
    layer: nn.Module
    group_sizes: list[int]
    seq_len: int
    head_dim: int
    can_optimise: bool
    reason: str

class ModelAnalyser:
    def analyse(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> dict[str, GroupAttentionInfo]:
        ...
```

Group sizes can come from two sources:

1. **Explicit `group_ids`** — a tensor of shape `[T]` where each element is
   the group index for that token. Compute sizes with `torch.unique(group_ids,
   return_counts=True)`.
2. **Inferred from model config** — for Chronos2, detect `patch_size` or
   similar attributes on the attention module and derive fixed group sizes.

### Phase 3 — kernel dispatcher

Select the right kernel variant based on the actual group size distribution
seen at runtime.

```python
class KernelDispatcher:
    SMALL_THRESHOLD  = 32
    MEDIUM_THRESHOLD = 256

    def select_variant(self, group_sizes: list[int]) -> KernelVariant:
        max_size = max(group_sizes)
        if max_size <= self.SMALL_THRESHOLD:
            return KernelVariant.SMALL
        elif max_size <= self.MEDIUM_THRESHOLD:
            return KernelVariant.MEDIUM
        else:
            return KernelVariant.LARGE

    def recommend_config(self, stats: dict) -> dict:
        """
        Auto-configure from MASE statistical profiler output.
        Integrates with chop.passes.graph.analysis.statistical_profiler.
        """
        group_sizes = stats.get("group_sizes", [])
        variant = self.select_variant(group_sizes)
        return {"kernel_variant": variant.value, "group_sizes": group_sizes}
```

### Phase 4 — MASE transform pass

The actual pass that MASE calls. It walks the `MaseGraph`, finds attention
layers the analyser says can be optimised, and replaces them with a wrapped
module that calls the selected kernel.

```python
def grouped_sparse_attn_transform_pass(
    mg: MaseGraph,
    pass_args: dict,
) -> tuple[MaseGraph, dict]:
    """
    MASE transform pass. Replaces eligible attention layers with the
    grouped sparse attention kernel.

    Usage in CompressionPipeline:
        pass_args = {
            "grouped_sparse_attn_transform_pass": {
                "by": "type",
                "default": {"config": {"name": None}},
                "grouped_self_attention": {
                    "config": {"kernel_variant": "auto"}
                }
            }
        }
    """
    config = pass_args.get("grouped_sparse_attn_transform_pass", {})
    analyser = ModelAnalyser()
    dispatcher = KernelDispatcher()

    for node in mg.fx_graph.nodes:
        if _node_is_eligible(node, config):
            layer_info = analyser.analyse_node(node)
            if layer_info.can_optimise:
                kernel_cfg = dispatcher.recommend_config({"group_sizes": layer_info.group_sizes})
                _replace_node_with_kernel(mg, node, kernel_cfg)

    mg.fx_graph.lint()
    mg.recompile()
    return mg, {}
```

### Phase 5 — optional: joint quantisation + sparsity search

MASE already has mixed-precision quantisation search (Tutorial 6). Extend it
to jointly search over both quantisation precision and attention sparsity
pattern per layer. The search space:

```python
search_space = {
    "attention_layers": {
        "quantization": ["fp16", "int8", "int4"],
        "sparsity_pattern": ["dense", "grouped", "local_window"],
        "kernel_variant": ["auto", "small", "medium", "large"],
    }
}
```

Plot the Pareto frontier: accuracy vs latency vs memory. Even a small
experiment is a strong result.

---

## GPU-specific notes (for RTX 3090 / sm_86)

- Compile with `-arch=sm_86`
- Default shared memory per block: 48 KB. Request up to 99 KB with
  `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024)`
- Warp size: 32 (always, on all current NVIDIA hardware)
- Max threads per block: 1024 → max square tile = 32×32
- Use `cuda::memcpy_async` for the large kernel variant (available on 8.6)
- Put `group_offsets` and `group_sizes` in `__constant__` memory if num_groups
  fits (max 64 KB): avoids repeated global memory reads by every thread block

Check device properties at runtime:

```python
import torch
p = torch.cuda.get_device_properties(0)
print(f"SM count:           {p.multi_processor_count}")
print(f"Shared mem / block: {p.shared_memory_per_block // 1024} KB")
print(f"Max threads/block:  {p.max_threads_per_block}")
print(f"L2 cache:           {p.l2_cache_size // (1024**2)} MB")
```

---

## Correctness testing

Every kernel variant must pass a numerical correctness check against PyTorch's
reference implementation before benchmarking.

```python
import torch
import torch.nn.functional as F

def test_correctness(kernel_fn, B=2, T=128, d=64, num_groups=8):
    torch.manual_seed(42)
    Q = torch.randn(B, T, d, device="cuda")
    K = torch.randn(B, T, d, device="cuda")
    V = torch.randn(B, T, d, device="cuda")

    # build group_ids: assign each token to a group
    group_ids = torch.randint(0, num_groups, (T,), device="cuda")
    group_ids, _ = torch.sort(group_ids)  # sort so groups are contiguous

    # reference: PyTorch dense attention with block-diagonal mask
    mask = build_group_mask(group_ids, T)  # True where attention is allowed
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

    # kernel output
    out = kernel_fn(Q, K, V, group_ids)

    assert torch.allclose(out, ref, atol=1e-4), \
        f"Max deviation: {(out - ref).abs().max().item():.6f}"
    print("Correctness check passed.")
```

---

## Benchmarks to run

All benchmarks should sweep over sequence length T ∈ {128, 256, 512, 1024,
2048} and group size G ∈ {8, 16, 32, 64, 128}.

**Latency** — wall-clock time per forward pass:

```python
import torch.utils.benchmark as benchmark

t = benchmark.Timer(
    stmt="kernel_fn(Q, K, V, group_ids)",
    globals={"kernel_fn": kernel_fn, "Q": Q, "K": K, "V": V, "group_ids": group_ids},
)
result = t.blocked_autorange(min_run_time=1.0)
print(f"Median latency: {result.median * 1e3:.3f} ms")
```

**Memory** — peak VRAM during forward:

```python
torch.cuda.reset_peak_memory_stats()
kernel_fn(Q, K, V, group_ids)
peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Peak memory: {peak_mb:.1f} MB")
```

**Accuracy** — on Chronos2's forecasting task. Report WQL (Weighted Quantile
Loss) and MASE (Mean Absolute Scaled Error) against the dense attention
baseline. A regression of more than 0.5% on either metric is a red flag.

**Comparison baselines:**

1. PyTorch `scaled_dot_product_attention` with block-diagonal boolean mask
2. Dense attention (no mask) — upper bound on compute
3. FlashAttention (if available) — state-of-the-art reference

---

## What makes this principled (for the writeup)

- The grouped self-attention pattern is directly from the Chronos2 paper.
  Cite: Amazon Chronos2 technical report.
- The tile-level early exit strategy is the same insight as FlashAttention
  (Dao et al., 2022) — work at tile granularity, not element granularity.
- The kernel size dispatch mirrors how cuBLAS internally selects GEMM
  algorithms based on matrix dimensions — a well-known heuristic in HPC.
- The use of MASE's statistical profiler to auto-configure dispatch makes it
  a principled automated system rather than a hardcoded heuristic.

---

## MASE contribution checklist (PR requirements)

- [ ] Transform pass registered in `chop/passes/graph/transforms/__init__.py`
- [ ] Pass follows existing MASE pass signature: `(mg, pass_args) -> (mg, dict)`
- [ ] Kernel compiles cleanly with `python setup.py build_ext --inplace`
- [ ] All tests pass under `pytest test/passes/graph/transforms/attention/`
- [ ] Benchmark script runs end-to-end and produces a results table
- [ ] README updated with usage example
- [ ] Docstrings on all public functions
- [ ] Code follows MASE Python style spec (Black formatter, type hints)

---

## Open questions / decisions to make

1. **CUDA vs Triton?** Triton integrates more naturally with MASE's existing
   fusion pass and autotunes block sizes. CUDA gives more control for the small
   warp-level variant. Recommend: Triton for medium/large, CUDA for small.

2. **Variable group sizes.** If groups are different sizes, some blocks waste
   threads on padding. Options: (a) pad all groups to the next power of 2
   — simple but wastes up to 50% of threads; (b) sort groups by size and
   assign blocks accordingly — complex but better utilisation.

3. **Multi-head.** Chronos2 uses multi-head attention. The kernel needs a head
   dimension. Simplest approach: fold heads into the batch dimension
   (`B_eff = B * num_heads`) and treat each head independently.

4. **Backward pass.** The coursework is focused on inference optimisation, so
   a forward-only kernel is fine. If a backward pass is needed for QAT, use
   `torch.autograd.Function` and implement a custom backward, or fall back to
   PyTorch autograd for the backward pass only.

---

## References

- Chronos2: Amazon Chronos technical report (cite when available)
- FlashAttention: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness", NeurIPS 2022
- MASE: Zhang et al., "MASE: An Efficient Representation for Software-Defined
  ML Hardware System Exploration", NeurIPS 2023 ML for Systems Workshop
- Longformer: Beltagy et al., "Longformer: The Long-Document Transformer",
  2020 (local window + strided attention patterns)
- Sparse Transformer: Child et al., "Generating Long Sequences with Sparse
  Transformers", 2019
