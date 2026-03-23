# Chronos2 Fast Group Attention — Implementation Notes

## Overview

The system replaces the standard `GroupSelfAttention.self_attention` (a generic `MHA`)
with a `GroupAwareMHA` that exploits the **block-diagonal sparsity** of the group
attention mask. The key insight: since each series only attends within its own group,
there is no point computing cross-group attention scores — they would be masked to
`-inf` anyway.

---

## Part 1: MASE Transform Pass (`FastGroupAtten.py`)

This is a **graph rewriting pass** — it modifies the `MaseGraph` (a `torch.fx` graph)
at **compile time** before any inference.

```python
fast_group_attention_transform_pass(mg, pass_args={"group_ids": ...})
```

**Steps:**

1. **Analyse** — `GroupAttentionAnalyser.analyse(mg)` walks every `call_module` node
   in the fx graph, finding all `GroupSelfAttention` instances whose `.self_attention`
   has not yet been replaced.
2. **Precompute** — `GroupPartition.from_group_ids(group_ids)` computes the group
   structure once on CPU: which batch indices belong to each group, how many groups,
   max group size, and whether all groups are size 1.
3. **Dispatch** — `KernelDispatcher.select(partition, device)` picks the best kernel
   variant. Decided **once at pass time** — no runtime overhead.
4. **Swap** — `module.self_attention = GroupAwareMHA(...)`. The outer
   `GroupSelfAttention` (layer norm, residual, dropout, transposes) is **never
   touched**.
5. **Metadata** — writes `group_ids`, `partition`, and `kernel_variant` into the
   node's MASE metadata for downstream passes.

---

## Part 2: GroupAwareMHA — Three Dispatch Paths (`optimized_layers.py`)

All indexing structures are precomputed at construction time; the `forward()` pass
contains **no Python branching on runtime tensor values**.

### Path 1: Univariate (all groups size 1)

```python
return self.o(self.v(x))  # Q and K projections skipped entirely
```

When every series is its own group, attention over a single element gives
`softmax([score]) = 1`, so `out = V`. Both the Q and K projections are bypassed.

### Path 2: Packed Sparse (CPU / small groups)

Precomputed buffers:

| Buffer | Shape | Purpose |
|---|---|---|
| `_group_indices_padded` | `(G, M)` | Batch indices per group, padded to `max_group_size` |
| `_pad_mask` | `(G, 1, M, M)` | Additive `-inf` mask for padding positions |
| `_real_flat_idx` | `(num_real,)` | Flat indices of real (non-padding) slots |
| `_scatter_b_idx` | `(num_real,)` | Original batch indices for scattering results back |

**Forward:**

```
x (T, B, d)
  → gather: x[:, group_indices_padded] → (T, G, M, d)
  → reshape → (T*G, M, d)         # each group treated as a "batch item"
  → QKV projections → (T*G, H, M, D)
  → standard eager attention with pad_mask
  → scatter real slots back → (T, B, d)
```

### Path 3: Triton Fused (CUDA, large groups)

Precomputed buffers:

| Buffer | Shape | Purpose |
|---|---|---|
| `_sort_perm` | `(B,)` | Permutation sorting batch by group |
| `_cu_seqlens` | `(G+1,)` int32 | Cumulative group sizes `[0, g0, g0+g1, ...]` |
| `_unsort_perm` | `(B,)` | Inverse permutation to restore original order |

**Forward:**

```
x (T, B, d)
  → QKV projections
  → sort by group → (T, H, B_sorted, D)
  → Triton kernel (no padding, no Python loops)
  → unsort → (T, B, d)
```

---

## Part 3: Triton Kernel (`triton_grouped_attn.py`)

**Grid:** `(T × H × G × TILES_PER_GROUP,)` — each GPU program handles one
`(timestep, head, group, query-tile)` combination.

**Algorithm — tiled online attention (FlashAttention-style):**

Each program:

1. Determines its group bounds from `cu_seqlens[g]` and `cu_seqlens[g+1]`
2. Loads its `BLOCK_M × D` query tile into registers
3. Streams over K/V in `BLOCK_N`-sized tiles, maintaining per-row running accumulators:
   - `m_i` — current row maximum (for numerical stability)
   - `l_i` — current normalizer (sum of exponentials)
   - `acc` — weighted accumulator for values
4. After all K/V tiles: `out = acc / l_i`, written back to global memory

This **online softmax** avoids materialising the full `(group_size × group_size)`
attention matrix. `tl.dot` maps tile matmuls onto GPU tensor cores.

---

## End-to-End Data Flow

```
MaseGraph at pass time:
  GroupSelfAttention.self_attention  →  MHA  (original)
          ↓  fast_group_attention_transform_pass
  GroupSelfAttention.self_attention  →  GroupAwareMHA  (kernel variant baked in)

At runtime, GroupSelfAttention shell does:
  x.transpose(0, 1)              # (batch, time, d) → (time, batch, d)
  normed_x = layer_norm(x)
  out = GroupAwareMHA.forward(normed_x, mask)   ← no runtime branching
    ├─ univariate:     o(v(x))
    ├─ packed_sparse:  gather → pad → attn → scatter
    └─ triton:         sort → fused kernel → unsort
  x = x + dropout(out)           # residual
  x.transpose(0, 1)              # (time, batch, d) → (batch, time, d)
```

---

## Complexity

The original `MHA` inside `GroupSelfAttention` builds a full `(B, B)` attention
matrix and masks off cross-group entries:

$$O(B^2)$$

`GroupAwareMHA` never computes those entries:

$$O\!\left(\sum_g |g|^2\right)$$

When groups are small (e.g., all univariate), this reduces to $O(B)$.

---

## Kernel Variant Selection (`KernelDispatcher`)

| Condition | Variant |
|---|---|
| All groups size 1 | `UNIVARIATE` |
| CUDA + Triton available + `max_group_size > 8` | `TRITON` |
| Otherwise (CPU, small groups, no Triton) | `PACKED_SPARSE` |

The crossover threshold (8) is configurable via `KernelDispatcher.TRITON_CROSSOVER`.
