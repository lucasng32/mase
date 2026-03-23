# Idea: Triton Group Concatenation for Small Groups

## Problem with the Current Triton Kernel

The current tiled Triton kernel (`_tiled_grouped_attn_fwd`) launches one program per `(timestep, head, group, query_tile)`. For small groups:

- A group of size 2 with `BLOCK_M=32` means 2 real queries and 30 padding rows per tile
- Each program does 32×32 = 1024 score computations, but only 2×2 = 4 are non-masked
- **Utilisation: 0.4%** — the kernel is almost entirely wasted work
- Worse, the kernel launch overhead itself (grid scheduling, SM allocation) is paid per program

For a batch of 128 pairs (`G=64` groups of 2), the grid is `T × H × 64 × 1` programs — 64 Triton program launches per timestep per head, each doing nearly no real work.

---

## The Idea

**Concatenate multiple small groups into a single virtual tile** so that the real queries fill the tile, then mask out cross-group attention within the virtual tile using an intra-tile block-diagonal mask.

Instead of:
```
Group 0: [q0, q1]          → 1 Triton program, BLOCK_M=32, 2/32 rows real
Group 1: [q2, q3]          → 1 Triton program, BLOCK_M=32, 2/32 rows real
...
Group 15: [q30, q31]       → 1 Triton program, BLOCK_M=32, 2/32 rows real
```

Do:
```
Virtual group [0..15]: [q0,q1, q2,q3, ..., q30,q31] → 1 Triton program, BLOCK_M=32, 32/32 rows real
  (intra-tile mask blocks q0→q2, q0→q4, etc.)
```

16 Triton programs become 1. Utilisation goes from 0.4% to near 100%.

---

## How the Masking Works

Within a virtual tile of `N_real = bin_count × group_size` queries, the attention mask is block-diagonal at the sub-group level:

```
Virtual tile (4 groups of 2, BLOCK_M=8):

Q\K  q0 q1 q2 q3 q4 q5 q6 q7
q0   ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗
q1   ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗
q2   ✗  ✗  ✓  ✓  ✗  ✗  ✗  ✗
q3   ✗  ✗  ✓  ✓  ✗  ✗  ✗  ✗
...
```

Inside the kernel, this mask is computed cheaply: a query at position `q` within the virtual tile belongs to sub-group `q // group_size`; a key at position `k` belongs to sub-group `k // group_size`. They may attend iff `q // group_size == k // group_size`.

```triton
q_subgroup = q_offs // GROUP_SIZE   # (BLOCK_M,)
k_subgroup = k_offs // GROUP_SIZE   # (BLOCK_N,)
intra_mask = q_subgroup[:, None] == k_subgroup[None, :]  # (BLOCK_M, BLOCK_N)
s = tl.where(intra_mask, s, float("-inf"))
```

This is a single integer comparison per score — negligible cost.

---

## Binning Strategy

Groups are binned by size into categories. Within each category, groups are concatenated until the virtual tile is full:

```
category size=2:  concatenate floor(BLOCK_M / 2) = 16 groups per virtual tile
category size=4:  concatenate floor(BLOCK_M / 4) = 8  groups per virtual tile
category size=8:  concatenate floor(BLOCK_M / 8) = 4  groups per virtual tile
category size=16: concatenate floor(BLOCK_M / 16) = 2 groups per virtual tile
category size=32: 1 group per tile (fills exactly)
category size>32: use existing tiled kernel (multi-tile, no change)
```

For non-power-of-2 group sizes, round up to the next power of 2 within the tile (wastes a few rows but keeps the intra-tile mask simple).

**Remainder handling:** if `num_groups % bin_count != 0`, the last virtual tile is partially filled. These leftover rows get the same early-exit treatment as padding in the current kernel (`if q_begin >= g_size: return`), extended to check real slot count within the virtual tile.

---

## Precomputed Buffers

At MASE pass time, groups are sorted into bins and a virtual-group layout is computed:

```python
@dataclass
class ConcatBin:
    group_size: int           # size of real groups in this bin
    bin_count: int            # groups per virtual tile = BLOCK_M // group_size
    virtual_cu_seqlens: Tensor  # (num_virtual_groups + 1,) cumulative virtual sizes
    sort_perm: Tensor           # (B_bin,) — batch indices for groups in this bin, sorted
    unsort_perm: Tensor         # inverse
    num_real_per_virtual: Tensor  # (num_virtual_groups,) — how many real groups in each vtile
```

The forward pass processes each bin independently and scatters results back.

---

## Forward Pass Sketch

```python
def _forward_triton_concat(self, normed):
    T, B, d = normed.shape
    q = mha.q(normed).view(T, B, H, D)
    k = mha.k(normed).view(T, B, H, D)
    v = mha.v(normed).view(T, B, H, D)
    out = torch.empty_like(q)

    for bin in self._concat_bins:
        # Extract the slice of the batch belonging to this bin
        q_bin = q[:, bin.sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
        k_bin = k[:, bin.sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
        v_bin = v[:, bin.sort_perm, :, :].permute(0, 2, 1, 3).contiguous()

        out_bin = triton_concat_grouped_attention(
            q_bin, k_bin, v_bin,
            virtual_cu_seqlens=bin.virtual_cu_seqlens,
            group_size=bin.group_size,        # for intra-tile mask
            num_virtual_groups=len(bin.virtual_cu_seqlens) - 1,
        )
        # Scatter back
        out[:, bin.unsort_perm, :, :] = out_bin.permute(0, 2, 1, 3)

    return mha.o(out.reshape(T, B, -1))
```

---

## Kernel Changes

The existing `_tiled_grouped_attn_fwd` needs two modifications:

1. **Grid:** `(T × H × num_virtual_groups × TILES_PER_VIRTUAL,)` instead of `(T × H × G × TILES_PER_GROUP,)`. Since virtual groups fill a tile, `TILES_PER_VIRTUAL = ceil(BLOCK_M / BLOCK_M) = 1` for groups that fit within one tile.

2. **Intra-tile mask:** after computing the score tile `S = Q @ K^T`, apply:
   ```triton
   q_sub = (q_offs - vg_start) // GROUP_SIZE
   k_sub = (k_offs - vg_start) // GROUP_SIZE
   same_sub = q_sub[:, None] == k_sub[None, :]
   s = tl.where(same_sub & k_valid[None, :] & q_valid[:, None], s, float("-inf"))
   ```

Everything else (online softmax, accumulation, store) is identical.

---

## Expected Improvement

For the **mixed-batch regression** case (100 pairs + 1 group of 16, B≈212):

| Approach | Programs launched | Real scores | Total scores | Utilisation |
|---|---|---|---|---|
| Current packed_sparse | 1 PyTorch MHA | 416 | 43,264 | 1.0% |
| Current Triton (skipped — max_size=16 > 8 crossover) | 101 programs | 416 | 51,200 | 0.8% |
| **Concat Triton (pairs bin)** | **7 programs** | **400** | **1,024** | **39%** |
| + **1 program for g16** | +1 | 256 | 256 | 100% |
| **Combined** | **8 programs** | **656** | **1,280** | **51%** |

vs packed_sparse at 1.0%: **51× better utilisation**, and 8 kernel programs vs equivalent of thousands of tiny MHA calls.

---

## Relationship to Existing Paths

This would be a new `KernelVariant.TRITON_CONCAT` path:

```
KernelDispatcher:
  all_univariate                         → UNIVARIATE
  cuda_bucketed available                → CUDA_BUCKETED
  triton, mixed/small groups             → TRITON_CONCAT  ← new
  triton, max_group_size > BLOCK_M       → TRITON (existing, handles very large groups)
  else                                   → PACKED_SPARSE
```

The existing Triton kernel remains for very large groups where a single group spans multiple query tiles — the concat approach only applies when a group fits within one tile.

---

## Open Questions

1. **Bin launch overhead vs utilisation tradeoff** — for very small batches (B < 32), even 8 kernel launches may be expensive relative to one PyTorch MHA call. A minimum-B threshold per bin makes sense.

2. **Non-uniform group sizes within a bin** — groups of size 3 in a bin targeted at size-4 waste 1/4 of each virtual tile's K dimension. Rounding group sizes up to the next power of 2 before binning simplifies the intra-tile mask but adds some waste. Exact-size bins (separate bins for size 3 and size 4) eliminate this but add more kernel variants.

3. **Interaction with the bucketed CUDA idea** — both ideas solve the same mixed-batch problem from different angles. CUDA bucketed is better for small groups (fully unrolled inner loop, no tile overhead); Triton concat is better for medium groups (BLOCK_M=32 fits well, tensor core utilisation). The two could coexist: CUDA bucketed for group_size ≤ 8, Triton concat for 8 < group_size ≤ BLOCK_M.

4. **Multi-tile virtual groups** — if a virtual group (after concatenation) is larger than BLOCK_M, it needs multiple query tiles. This is the same as the existing Triton kernel's tiling, so the concat approach naturally degrades to the existing behaviour for large virtual groups.
