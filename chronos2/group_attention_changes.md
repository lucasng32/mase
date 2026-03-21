# FastGroupSelfAttention — Algorithm Documentation

## Background

Chronos-2 is a time-series foundation model built on a T5-style encoder. Each encoder block contains three sub-layers:

1. **TimeSelfAttention** — standard multi-head self-attention along the sequence (patch) dimension with RoPE
2. **GroupSelfAttention** — self-attention along the *batch* dimension, masked so each series only attends to others in the same group
3. **FeedForward** — two-layer MLP with residual connection

The optimisation targets **GroupSelfAttention** exclusively. The other two sub-layers are unchanged.

---

## What GroupSelfAttention Does

Given a batch of `B` time series and a `group_ids` tensor of length `B`, two series are allowed to attend to each other if and only if they share the same group ID.

```
group_ids = [0, 0, 1, 1, 1, 2]
              ↑───↑  ↑───↑───↑  ↑
            group 0  group 1  group 2
```

The baseline implementation achieves this by constructing a dense `(T, 1, B, B)` additive mask (−∞ for cross-group pairs, 0 for same-group pairs) and running a standard full `B×B` attention. This is **O(B²)** in compute and memory regardless of group size — a group of 2 still touches every other series in the batch.

---

## The Core Idea

The mask is block-diagonal. Series outside a group contribute zero to the softmax numerically. We exploit this by **never computing cross-group dot-products** — only intra-group pairs are touched.

This reduces compute from **O(B²)** to **O(B × max\_group\_size)**, which is linear in B for fixed group sizes.

---

## Phase 1 — Precomputation (at MASE pass time, once)

When the MASE transform pass is applied, `group_ids` is known. Everything that depends only on the group structure is computed once and stored as registered buffers on the module — never recomputed in a forward pass.

### GroupPartition

```python
GroupPartition.from_group_ids(group_ids)
```

Scans `group_ids` and extracts:

| Field | Description |
|---|---|
| `groups` | List of index tensors — one per group, containing the batch indices that belong to it |
| `num_groups` | Number of distinct groups |
| `max_group_size` | Size of the largest group |
| `all_univariate` | True if every group has exactly one member |

### KernelDispatcher

Selects the best execution path for the given partition and device:

```
all groups size 1?              → UNIVARIATE
CUDA ext available (opt-in)?    → CUDA
Triton available AND
  max_group_size > 8?           → TRITON
otherwise                       → PACKED_SPARSE
```

The crossover threshold of 8 is empirically derived — below it, Triton's tile-launch overhead exceeds the savings from sparsity; packed_sparse is cheaper.

---

## Phase 2 — Forward Pass

Input: `hidden_states (B, T, d_model)`, transposed internally to `(T, B, d_model)` so the batch axis aligns with the attention sequence dimension.

### Path 1 — UNIVARIATE

**Condition:** every series is its own group (all groups size 1).

softmax of a single logit is always 1, so the attention output is exactly V. Q and K projections are skipped entirely.

```
normed → V projection → O projection → residual
```

**Speedup source:** eliminates two large linear projections (Q and K) and the entire attention computation per block. At B=512, this gives **7×** speedup on the group attention layer.

---

### Path 2 — PACKED\_SPARSE

**Condition:** groups are small (max_group_size ≤ 8), or no GPU kernel is available.

Uses standard PyTorch batched MHA, but reshapes the problem so each "sequence" is one group rather than the full batch.

**Precomputed buffers:**
- `_group_indices_padded (G, M)` — for each group, the batch indices of its members, zero-padded to `max_group_size M`
- `_pad_mask (G, 1, M, M)` — additive mask with −∞ in padding slots
- `_real_flat_idx` — flat indices into the padded output that correspond to real (non-padded) slots
- `_scatter_b_idx` — where each real slot maps back to in the original batch

**Forward steps:**

```
1. Gather:   normed[:, group_indices_padded, :] → (T·G, M, d)
2. Attend:   batched MHA with pad_mask → (T·G, M, d)
3. Scatter:  pick real slots → (T, B, d)
```

The MHA now sees `T·G` sequences of length `M` (max group size) instead of one sequence of length `B`. For small groups, `G·M ≈ B` but each attention matrix is `M×M` instead of `B×B`, so compute drops from `B²` to `G·M²`.

**Limitation:** when group sizes vary widely (mixed batches), every group is padded to the largest group's size, wasting compute on padding slots.

---

### Path 3 — TRITON

**Condition:** CUDA device, `max_group_size > 8`, Triton available.

Avoids padding entirely. Sorts the batch so each group occupies a contiguous slice, then runs a custom tiled Triton kernel.

**Precomputed buffers:**
- `_sort_perm (B,)` — permutation that places group members contiguously
- `_cu_seqlens (G+1,)` — cumulative group sizes (start/end index for each group)
- `_unsort_perm (B,)` — inverse permutation to restore original batch order

**Forward steps:**

```
1. Project Q, K, V: (T, B, d) → (T, B, H, D)
2. Sort batch:      index with _sort_perm → (T, B_sorted, H, D)
3. Permute:         → (T, H, B_sorted, D)  [contiguous for kernel]
4. Triton kernel
5. Permute back:    → (T, B_sorted, H·D)
6. Unsort:          index with _unsort_perm → (T, B, H·D)
7. O projection
```

**Triton kernel** (`_tiled_grouped_attn_fwd`):

Grid: one program per `(timestep t, head h, group g, query-tile q_tile)`.

Each program:
1. Loads a `BLOCK_M × D` tile of queries for its group slice
2. Streams over K/V in `BLOCK_N`-sized tiles
3. Accumulates with **online softmax** (single pass — no double-read of K/V):

```
for each K/V tile:
    S = Q_tile @ K_tile^T * scale          # (BLOCK_M, BLOCK_N)
    m_new = max(m_i, row_max(S))
    alpha = exp(m_i - m_new)               # rescale factor
    p = exp(S - m_new)                     # unnormalised attention
    l_i = l_i * alpha + row_sum(p)
    acc = acc * alpha + p @ V_tile         # accumulate
    m_i = m_new
acc = acc / l_i                            # normalise
```

`tl.dot` maps tile matmuls onto tensor cores. No padding is allocated — groups with fewer than `BLOCK_M` queries simply return early.

---

### Path 4 — CUDA (opt-in)

**Condition:** CUDA device, extension compiled, `use_cuda_ext=True` explicitly passed.

Same sorted-batch layout as Triton but uses an AOT-compiled CUDA kernel (`grouped_attn.cu`). One CUDA thread per `(timestep, head, batch_index)`.

Each thread runs the full online softmax loop over its group's keys entirely in registers, with `#pragma unroll` on all head-dim loops. The `HEAD_DIM` template parameter (16/32/64/128) is resolved at compile time so the compiler fully unrolls inner loops.

**Compiled once** by nvcc via `torch.utils.cpp_extension.load()` and cached as a `.so`. No JIT overhead on subsequent runs.

Currently not auto-selected by default — benchmarks showed it performs similarly to Triton for the group sizes common in Chronos-2 deployments, and requires nvcc to be installed.

---

## Integration — MASE Transform Pass

```python
from chop.passes.graph.transforms.timeseries import fast_group_attention_transform_pass

mg, info = fast_group_attention_transform_pass(
    mg, pass_args={"group_ids": torch.arange(64)}
)
```

The pass:
1. Walks the FX graph to find all `GroupSelfAttention` nodes
2. Computes `GroupPartition` once (shared across all encoder layers)
3. Selects kernel variant via `KernelDispatcher`
4. Constructs `FastGroupSelfAttention`, transfers weights with `load_state_dict(strict=False)`
5. Swaps the module in-place and calls `mg.model.recompile()`

State-dict keys are identical to `GroupSelfAttention` so weights transfer without any remapping.

---

## Benchmark Results (Chronos-2-Small, d\_model=512, 8 heads, 6 layers, SDPA baseline)

### GroupSelfAttention layer in isolation

| Case | B | Variant | Speedup | Mem reduction |
|---|---|---|---|---|
| Univariate | 128 | univariate | 3.3× | 57% |
| Univariate | 512 | univariate | 7.2× | 79% |
| Pairs | 512 | packed\_sparse | 1.1× | 63% |
| Quads | 512 | packed\_sparse | 1.7× | 63% |
| Octets | 512 | packed\_sparse | 2.2× | 63% |
| Groups of 16 | 512 | triton | 2.8× | 67% |
| Dense (1 group) | 256 | triton | 1.5× | 50% |
| Mixed (~600) | 596 | packed\_sparse | 1.5× | 13% |

Speedup scales with B because baseline compute grows as O(B²) while ours grows as O(B × group\_size).

### Full Chronos2Model end-to-end

Gains are diluted by TimeSelfAttention and FFN (unchanged), but the trend holds at large B:

| Case | B | Speedup |
|---|---|---|
| Univariate | 500 | 1.54× |
| Pairs | 500 | 1.35× |
| Quads | 500 | 1.35× |
| Groups of 16 | 496 | 1.30× |

---

## Known Limitations

**Mixed-batch regression at small B:** when `max_group_size` is large (e.g. 8) but most groups are size 1–2, packed_sparse pads every small group up to size 8, wasting 7/8 of the compute for those groups. At small batch sizes this causes a regression (0.42× for B≈112 mixed). At large B (~600), the large-group members dominate and the result is positive (1.48×).

**Pairs at small B with SDPA baseline:** SDPA (FlashAttention) is very efficient for small sequences. For pairs at B=128, the baseline SDPA on a 128×128 matrix is faster than our packed_sparse overhead. The crossover is around B=512.

---

## Files

| File | Purpose |
|---|---|
| `src/chop/models/chronos2/optimized_layers.py` | `FastGroupSelfAttention`, `GroupPartition`, `KernelDispatcher` |
| `src/chop/models/chronos2/triton_grouped_attn.py` | Tiled Triton kernel |
| `src/chop/models/chronos2/cuda_grouped_attn.py` | CUDA extension loader and wrapper |
| `src/chop/models/chronos2/kernels/grouped_attn.cu` | AOT CUDA kernel source |
| `src/chop/passes/graph/transforms/timeseries/FastGroupAtten.py` | MASE transform pass |
| `test/passes/graph/transforms/timeseries/test_grouped_sparse_attn.py` | Correctness tests (28 tests) |
| `test/passes/graph/transforms/timeseries/bench_grouped_sparse_attn.py` | Layer-level benchmark vs SDPA baseline |
| `test/passes/graph/transforms/timeseries/bench_group_attn_layer.py` | Isolated layer benchmark |
| `test/passes/graph/transforms/timeseries/bench_chronos2_e2e.py` | End-to-end model benchmark |

---

## Next Steps

| Priority | Optimisation | Expected gain |
|---|---|---|
| High | Per-category dispatch for mixed batches (size-1 → univariate, rest → packed\_sparse) | Fixes mixed regression |
| High | Skip group-time mask construction after transform pass (currently O(B²) per forward) | Removes redundant alloc |
| Medium | Fused RMS LayerNorm (18 calls per forward, currently 3 separate kernels each) | ~10–15% |
| Medium | `torch.compile` on encoder | 5–15% general |
| Low | RoPE cos/sin cache for fixed sequence length | Minor |
