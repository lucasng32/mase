# GroupAwareMHA — Algorithm Walkthrough

This document explains step-by-step what happens when
`GroupAwareMHA._forward_bucketed_triton` runs a forward pass.  It is written
for someone already familiar with multi-head attention but new to the
grouped-sparse-attention optimisation.

---

## Background: what is group attention?

Chronos2 processes **multivariate time-series** — multiple related signals
forecasted together (e.g. CPU %, memory %, network I/O on one host).  Related
series are assigned the same `group_id`.  Attention is computed only within
groups: a series in group 0 never attends to a series in group 1.

The attention mask is block-diagonal over the batch dimension:

```
group_ids = [0, 0, 1, 1, 1]

Mask (B × B):
  0 0 1 1 1     ← series 0 attends only to 0, 1
  0 0 1 1 1
  1 1 0 0 0     ← series 2, 3, 4 attend only to each other
  1 1 0 0 0
  1 1 0 0 0
```

(`0` = attend, `1` = -inf masked.)

The baseline `GroupSelfAttention` materialises this full B×B mask and runs
standard attention over all pairs — the vast majority of pairs cross group
boundaries and produce -inf logits that contribute nothing to the output.

---

## The three dispatch paths

`GroupAwareMHA` precomputes all index structures at MASE pass time and stores
them as registered buffers.  The `forward()` method has no Python conditionals
on runtime tensor values; dispatch is fixed at construction:

```
group_ids → GroupPartition → KernelDispatcher.select() → KernelVariant
```

| Variant | When selected | What it does |
|---|---|---|
| `UNIVARIATE` | All groups size 1 | `out = V`, skip Q and K entirely |
| `TRITON_BUCKETED` | CUDA + Triton available | Per-bucket Triton kernel launches |
| `TRITON` | Explicit override only | Single Triton launch, global `max_group_size` tile |
| `PACKED_SPARSE` | CPU or no Triton | Pad → batch matmul → scatter |

The rest of this document focuses on `TRITON_BUCKETED`.

---

## Step 0 — MASE pass (runs once, before inference)

```
fast_group_attention_transform_pass(mg, pass_args={"group_ids": group_ids})
```

1. **`GroupPartition.from_group_ids(group_ids)`** — find which batch indices
   belong to each group.
2. **`KernelDispatcher.select(partition, device)`** — pick `TRITON_BUCKETED`.
3. **Construct `GroupAwareMHA`** — copies weights from the existing `MHA`
   (`q`, `k`, `v`, `o` linear layers; `strict=True` load).
4. **`_precompute_bucketed_triton_buffers`** — build `BucketedPartition` from
   the partition, register `_bucketed_sort_perm` and `_bucketed_unsort_perm`
   as non-persistent buffers.
5. **Swap** `GroupSelfAttention.self_attention = group_aware_mha`.

The outer `GroupSelfAttention` (layer norm, residual, dropout, transposes) is
never touched.

---

## Step 1 — `BucketedPartition` construction (inside the pass)

Given `group_ids`, `BucketedPartition.from_group_ids` does:

### 1a. Assign each group to a power-of-2 bucket

```python
def _bucket_for_size(size: int) -> int:
    for b in [2, 4, 8, 16, 32, 64]:
        if size <= b:
            return b
    return next_power_of_2(size)
```

Example with `group_ids = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]`:

```
group 0: size 2  → bucket 2
group 1: size 4  → bucket 4
group 2: size 8  → bucket 8
```

### 1b. Sort groups by bucket, build global permutation

Groups are sorted by bucket size (stable within each bucket).  The batch
indices are concatenated in that order:

```
original:  [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
            idx 0 1  2 3 4 5  6 7 8 9 10 11 12 13

sorted:    bucket-2 first, then bucket-4, then bucket-8
           → global_sort_perm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
             (already sorted in this example)
```

`global_unsort_perm` is the inverse: `unsort[sort[i]] = i`.

### 1c. Build per-bucket `BucketData`

For each occupied bucket:

```
BucketData(
    bucket_size = 2,
    num_groups  = 1,
    cu_seqlens  = [0, 2],        # actual sizes, not padded
    batch_offset = 0,            # starts at position 0 in the sorted batch
)
BucketData(
    bucket_size = 4,
    num_groups  = 1,
    cu_seqlens  = [0, 4],
    batch_offset = 2,            # after the 2 elements of bucket-2
)
BucketData(
    bucket_size = 8,
    num_groups  = 1,
    cu_seqlens  = [0, 8],
    batch_offset = 6,            # after 2 + 4 elements
)
```

`cu_seqlens` records *actual* group sizes, not padded.  A group of size 3 in
bucket 4 has `cu_seqlens = [..., prev, prev+3, ...]`.  The kernel handles the
partial final tile via masking.

---

## Step 2 — Forward pass (runs at every inference call)

Input shape from `GroupSelfAttention`: `(B, T, d_model)`.
`GroupSelfAttention` transposes this to `(T, B, d_model)` before calling
`self_attention.forward(hidden_states, mask)`.  So `x` arriving in
`_forward_bucketed_triton` has shape `(T, B, d_model)`.

### 2a. Project Q, K, V

```python
q = self.q(x).view(T, B, H, D)   # (T, B, H, D)
k = self.k(x).view(T, B, H, D)
v = self.v(x).view(T, B, H, D)
```

One linear projection per head-set, over the full batch.  This is identical to
the standard MHA projection.

### 2b. Reorder batch into bucket order

```python
q_s = q[:, sort_perm, :, :].permute(0, 2, 1, 3).contiguous()  # (T, H, B, D)
k_s = k[:, sort_perm, :, :]...
v_s = v[:, sort_perm, :, :]...
```

After the permute, the batch axis (dim 2) is sorted so that all elements of
bucket-2 groups come first, then bucket-4 groups, then bucket-8 groups.

### 2c. Per-bucket kernel launches

```python
for bucket_size, bucket in partition.buckets.items():
    block_m = max(min(bucket_size, 32), 16)   # tensor-core floor at 16
    block_n = max(min(bucket_size, 32), 16)
    tiles_per_group = ceil(bucket_size / block_m)

    offset = bucket.batch_offset
    total  = cu_seqlens[-1]                   # total actual elements in this bucket

    Q_b = q_s[:, :, offset : offset + total, :]   # slice — no copy, just a view
    K_b = k_s[:, :, offset : offset + total, :]
    V_b = v_s[:, :, offset : offset + total, :]
    Out_b = Out[:, :, offset : offset + total, :]

    _tiled_grouped_attn_fwd[grid](..., BLOCK_M=block_m, BLOCK_N=block_n, ...)
```

Each slice is a **non-owning view** into the sorted tensor — no memory copy.
Triton receives the adjusted base pointer and the original strides.  Writes
from one bucket's kernel land in a disjoint slice of `Out`; there is no
synchronisation required between launches.

### 2d. Inside `_tiled_grouped_attn_fwd`

The kernel grid is `(T × H × G_bucket × TILES_PER_GROUP,)`.  Each program
handles one `(timestep, head, group, query-tile)` tuple.

```
pid → (t, h, g, q_tile)

g_start = cu_seqlens[g]
g_end   = cu_seqlens[g + 1]
g_size  = g_end - g_start

q_begin = q_tile * BLOCK_M
if q_begin >= g_size: return   # no-op for partial-tile overshoot
```

Then it:
1. Loads a `(BLOCK_M, D)` slice of Q for this tile, masking rows beyond
   `g_size`.
2. Streams over `(BLOCK_N, D)` tiles of K and V within the same group.
3. Accumulates the attention output with **online softmax** (single pass over
   K/V — no double-read, no materialised attention matrix).
4. Stores the result into `Out_b`.

Because `cu_seqlens` is local to each bucket and indexes into the bucket's
contiguous slice, each program only ever touches memory belonging to one group
in one bucket.  Programs from different bucket launches never access the same
memory.

### 2e. Restore original batch order and project output

```python
out = Out.permute(0, 2, 1, 3).reshape(T, B, -1)   # (T, B, inner_dim)
return self.o(out[:, unsort_perm, :])              # (T, B, d_model)
```

`unsort_perm` is the precomputed inverse of `sort_perm`.  The O projection is
a single linear layer over the full batch.

---

## Memory layout summary

```
Sorted batch tensor (T, H, B_sorted, D):

  ┌─────────────────────────────────────────────────────┐
  │ bucket-2 slice        │ bucket-4 slice   │ bucket-8  │
  │ [offset=0, total=2]   │ [offset=2, t=4]  │ [off=6,8] │
  └─────────────────────────────────────────────────────┘
  ↑                       ↑                  ↑
  launch 1                launch 2           launch 3
  BLOCK_M=16              BLOCK_M=16         BLOCK_M=16
  (clamped from 2)        (clamped from 4)   (clamped from 8)
```

Each kernel launch reads and writes a disjoint contiguous slice.  No atomic
operations, no cross-launch data dependencies.

---

## What the outer `GroupSelfAttention` still does

`GroupAwareMHA` only replaces the `self_attention` attribute.  The outer shell
is untouched:

```python
class GroupSelfAttention(nn.Module):
    def forward(self, hidden_states, mask):
        # 1. transpose: (B, T, d) → (T, B, d)
        x = hidden_states.transpose(0, 1)

        # 2. layer norm
        normed = self.layer_norm(x)

        # 3. attention  ← GroupAwareMHA runs here
        attn_out = self.self_attention(normed, mask).hidden_states

        # 4. dropout + residual
        x = x + self.dropout(attn_out)

        # 5. transpose back: (T, B, d) → (B, T, d)
        return AttentionOutput(hidden_states=x.transpose(0, 1), ...)
```

The `mask` argument is accepted by `GroupAwareMHA.forward` (to match the `MHA`
signature) but is not used — the group structure is fully baked into the
precomputed buffers.

---

## Files

| File | Purpose |
|---|---|
| `src/chop/models/chronos2/triton_bucketed_attn.py` | `BucketData`, `BucketedPartition`, `triton_bucketed_attention` |
| `src/chop/models/chronos2/optimized_layers.py` | `GroupAwareMHA`, `GroupPartition`, `KernelDispatcher`, `KernelVariant` |
| `src/chop/models/chronos2/triton_grouped_attn.py` | `_tiled_grouped_attn_fwd` Triton kernel, `triton_grouped_attention` |
| `src/chop/passes/graph/transforms/timeseries/FastGroupAtten.py` | MASE pass entry point |
| `test/.../test_bucketed_attn.py` | Correctness tests for `TRITON_BUCKETED` |
| `test/.../test_grouped_sparse_attn.py` | Correctness tests for all variants |
| `test/.../bench_bucketed_attn.py` | Performance benchmark |
