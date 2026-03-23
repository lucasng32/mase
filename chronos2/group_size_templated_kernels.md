# Group-Size-Bucketed Triton Kernels

## The Core Problem: Two Compounding Issues with `max_group_size`

### Issue 1 — Tile waste from global `TILES_PER_GROUP`

The current Triton kernel bakes `TILES_PER_GROUP` as a `tl.constexpr` computed from the
**largest group in the batch**:

```python
max_group_size = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
tiles_per_group = triton.cdiv(max_group_size, BLOCK_M)   # e.g. ceil(64/32) = 2
grid = (T * H * G * tiles_per_group,)
```

Because `TILES_PER_GROUP` is a compile-time constant shared by every group in the grid,
**every group gets the same number of tile-programs launched regardless of its actual size.**
A group of size 1 gets exactly as many programs launched as a group of size 64. The extra
programs immediately hit this check and return:

```python
if q_begin >= g_size:
    return  # padded tile — nothing to do
```

These programs are not free — they are scheduled, warps are allocated, and then the GPU
immediately discards them. This is silent wasted parallelism.

Beyond the tile count, `BLOCK_M` and `BLOCK_N` are hardcoded at 32. A group of size 2
loads a `(32, D)` Q tile with only 2 valid rows, and the `tl.dot` operates on a `32×32`
score matrix where only a `2×2` subblock is real. For a pair-group:

```
effective utilisation = (2 × 2) / (32 × 32) = 0.4%
```

### The realistic Chronos2 workload

The interesting case is genuinely mixed multivariate groups — batches like:

```
[2, 2, 3, 4, 4, 16]
```

Every group requires real attention computation, but group sizes vary widely. Under the
current kernel, all of them are padded to size 16 and run through `BLOCK_M=32` tiles.
The size-2 and size-3 groups are doing real work but wasting the vast majority of their
allocated compute on masked-out padding.

### Worked example

Batch: 50 pairs (size 2) + 50 quads (size 4) + 1 group of 32.

- `max_group_size = 32` → `TILES_PER_GROUP = 1`, `BLOCK_M = 32`, `BLOCK_N = 32`
- Every group — regardless of size — runs a `32×32` score matrix

| Path | Score matrix operations | Real scores | Utilisation |
|---|---|---|---|
| Current Triton | 101 × (32×32) | 50×(2×2) + 50×(4×4) + 32×32 | ~4% for pairs, ~16% for quads |
| Packed sparse | 101 × (32×32) | same | same |
| **Bucketed Triton** *(planned)* | 50×(2×2) + 50×(4×4) + 1×(32×32) | same | **~100% for all** |

---

## Current Implementation

### Architecture: `GroupAwareMHA` replaces the inner MHA

The MASE pass swaps **only `GroupSelfAttention.self_attention`** (the `MHA` instance)
with a `GroupAwareMHA` module.  The outer `GroupSelfAttention` shell — layer norm,
residual connection, dropout, and axis transposes — is left completely untouched.

```
Before:   GroupSelfAttention
            └─ self_attention: MHA          ← standard attention

After:    GroupSelfAttention
            └─ self_attention: GroupAwareMHA  ← group-aware dispatch
```

Weight keys (`q`/`k`/`v`/`o`) in `GroupAwareMHA` are identical to those in `MHA`, so
`load_state_dict(mha.state_dict(), strict=True)` transfers weights with no key renaming.

Index buffers (sort permutations, `cu_seqlens`, scatter indices) are registered with
`persistent=False` so they do not appear in `state_dict()` — they are always recomputed
from the group partition at pass application time.

### Data structures

```python
@dataclass
class GroupPartition:
    """Pre-computed group partition, derived once from a group_ids tensor."""
    groups:          list[torch.Tensor]  # per-group batch indices (1-D long tensors)
    num_groups:      int
    max_group_size:  int
    all_univariate:  bool

    @classmethod
    def from_group_ids(cls, group_ids: torch.Tensor) -> GroupPartition: ...
```

`GroupPartition.from_group_ids` is the single construction path.  It is called once
inside `fast_group_attention_transform_pass` and the result is stored in node metadata
and inside each `GroupAwareMHA`.

### Kernel dispatch

```python
class KernelVariant(Enum):
    UNIVARIATE    = auto()   # all groups size 1 — skip Q/K entirely
    TRITON        = auto()   # fused Triton tiled kernel
    PACKED_SPARSE = auto()   # pure PyTorch fallback (CPU or small groups)

class KernelDispatcher:
    TRITON_CROSSOVER = 8  # min max_group_size at which Triton beats packed-sparse

    @staticmethod
    def select(partition: GroupPartition, device: torch.device) -> KernelVariant:
        if partition.all_univariate:
            return KernelVariant.UNIVARIATE
        if device.type == "cuda" and is_triton_available() \
                and partition.max_group_size > KernelDispatcher.TRITON_CROSSOVER:
            return KernelVariant.TRITON
        return KernelVariant.PACKED_SPARSE
```

The variant is chosen once at pass time and baked into `GroupAwareMHA._variant`.  The
`forward()` method has no Python control flow that depends on runtime tensor values.

The `kernel_variant` pass argument overrides auto-selection:

```python
mg, info = fast_group_attention_transform_pass(
    mg, pass_args={"group_ids": group_ids, "kernel_variant": "triton"}
)
```

### Three dispatch paths

**1. Univariate** (`all_univariate = True`):

softmax of a single score is 1, so `attn_out = V`.  Q and K projections are skipped
entirely.

```python
def _forward_univariate(self, x):
    return self.o(self.v(x))
```

**2. Triton fused** (CUDA, `max_group_size > 8`):

```python
def _forward_triton(self, x):
    T, B, d = x.shape
    H, D = self.n_heads, self.kv_proj_dim

    q = self.q(x).view(T, B, H, D)[:, self._sort_perm, :, :]
    k = self.k(x).view(T, B, H, D)[:, self._sort_perm, :, :]
    v = self.v(x).view(T, B, H, D)[:, self._sort_perm, :, :]

    q, k, v = [t.permute(0, 2, 1, 3).contiguous() for t in (q, k, v)]

    out = triton_grouped_attention(q, k, v, self._cu_seqlens,
                                   num_groups=self._partition.num_groups, scale=1.0)

    out = out.permute(0, 2, 1, 3).reshape(T, B, -1)
    return self.o(out[:, self._unsort_perm, :])
```

The batch is sorted by group into `_sort_perm` order so each group occupies a contiguous
slice, described by `_cu_seqlens`.  A single Triton kernel call handles all groups.
After the kernel, `_unsort_perm` restores the original batch order.

The kernel currently uses `BLOCK_M = BLOCK_N = 32` globally and computes
`TILES_PER_GROUP` from the global `max_group_size` — the tile-waste problem described
above is still present.  Per-bucket dispatch is the planned next step (see below).

**3. Packed sparse** (CPU or `max_group_size ≤ 8`):

Groups are padded to `max_group_size`, stacked into one batched tensor, a single
batched `torch.matmul` attention call runs, and real slots are scattered back.  Works
on any device.

### MASE pass

```python
# fast_group_attention_transform_pass (simplified)
partition = GroupPartition.from_group_ids(group_ids)
variant   = KernelDispatcher.select(partition, device)

group_aware_mha = GroupAwareMHA(config=mha.config, partition=partition, variant=variant)
group_aware_mha.load_state_dict(mha.state_dict())   # strict=True, keys match exactly
group_aware_mha.to(device)

module.self_attention = group_aware_mha             # outer GroupSelfAttention untouched
```

`GroupAttentionAnalyser` detects already-optimised nodes by checking
`isinstance(module.self_attention, GroupAwareMHA)` — a second pass application is a
no-op for those nodes.

---

## Planned Enhancement: Per-Bucket Kernel Specialisation

Instead of one kernel compiled with the global `max_group_size`, compile **separate kernel
instances per group-size bucket**. Each bucket has its own `TILES_PER_GROUP`, `BLOCK_M`,
and `BLOCK_N` constexprs baked in. Groups are sorted into buckets at MASE pass application
time; at inference, one kernel launch per occupied bucket.

```
size 2–2        → BLOCK_M=2,  BLOCK_N=2,  TILES_PER_GROUP=1
size 3–4        → BLOCK_M=4,  BLOCK_N=4,  TILES_PER_GROUP=1
size 5–8        → BLOCK_M=8,  BLOCK_N=8,  TILES_PER_GROUP=1
size 9–16       → BLOCK_M=16, BLOCK_N=16, TILES_PER_GROUP=1
size 17–32      → BLOCK_M=32, BLOCK_N=32, TILES_PER_GROUP=1
size 33–64      → BLOCK_M=32, BLOCK_N=32, TILES_PER_GROUP=2
size > 64       → BLOCK_M=32, BLOCK_N=32, TILES_PER_GROUP=ceil(size/32)
```

A group of actual size 3 uses the size-4 bucket: 1 padding slot per query, not padded to
the global maximum. A group of size 1 never enters the Triton kernel at all.

The same `_tiled_grouped_attn_fwd` kernel function is used for every non-univariate bucket.
Triton's JIT naturally produces a **separate compiled binary per unique combination of
`tl.constexpr` values** — no new kernel code is needed, only a new dispatch layer.

### Why Triton JIT specialises naturally here

When a `@triton.jit` function is called with different `tl.constexpr` arguments, Triton
compiles and caches a separate GPU binary for each unique combination. Calling with
`BLOCK_M=4` and `BLOCK_M=32` produces two distinct binaries — the first with a 4×4 tile
matmul, the second with a 32×32 matmul. There is no runtime branching between them.

This means bucketed dispatch is just calling the same kernel function multiple times with
different constexpr arguments — one call per occupied bucket. Triton handles the
specialisation automatically.

### Proposed data structures for bucketed dispatch

```python
@dataclass
class BucketData:
    bucket_size: int
    cu_seqlens:  torch.Tensor   # (num_groups_in_bucket + 1,) int32
    num_groups:  int

@dataclass
class BucketedPartition:
    buckets:            dict[int, BucketData]  # keyed by bucket_size
    global_sort_perm:   torch.Tensor           # (B,) reorders batch so groups are contiguous
    global_unsort_perm: torch.Tensor           # inverse of global_sort_perm
```

**`BucketedPartition.from_group_ids(group_ids)` construction:**

1. Compute size of each unique group from `group_ids`
2. Assign each group to bucket `next_power_of_2(size)`, min bucket = 2
3. Sort batch so groups are contiguous, and within that, groups of the same bucket are contiguous
4. For each occupied bucket, build its `cu_seqlens` from positions in the globally-sorted batch
5. Store one `BucketData` per occupied bucket

### Proposed forward pass

```python
def _forward_bucketed_triton(self, normed: torch.Tensor) -> torch.Tensor:
    T, B, d = normed.shape
    H, D = self.n_heads, self.kv_proj_dim
    BLOCK_D = triton.next_power_of_2(D)

    # Project Q, K, V once for the entire batch
    q = self.q(normed).view(T, B, H, D)
    k = self.k(normed).view(T, B, H, D)
    v = self.v(normed).view(T, B, H, D)

    # Sort full batch by group (univariate groups + each bucket contiguous)
    q_s = q[:, self._global_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
    k_s = k[:, self._global_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
    v_s = v[:, self._global_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
    out = torch.empty_like(q_s)

    # One kernel launch per occupied bucket
    for bucket_size, bucket in self._buckets.items():
        block_m = min(bucket_size, 32)
        block_n = min(bucket_size, 32)
        tiles_per_group = triton.cdiv(bucket_size, block_m)

        grid = (T * H * bucket.num_groups * tiles_per_group,)
        _tiled_grouped_attn_fwd[grid](
            q_s, k_s, v_s, out,
            bucket.cu_seqlens, 1.0,
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            k_s.stride(0), k_s.stride(1), k_s.stride(2), k_s.stride(3),
            v_s.stride(0), v_s.stride(1), v_s.stride(2), v_s.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            H=H, G=bucket.num_groups, D=D,
            TILES_PER_GROUP=tiles_per_group,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=BLOCK_D,
        )

    # Unsort back to original batch order
    out = out.permute(0, 2, 1, 3).reshape(T, B, -1)
    return self.o(out[:, self._global_unsort_perm, :])
```

Q/K/V projections happen once for the full batch. Each bucket kernel reads and writes only
its own contiguous slice of the sorted batch — no overlap, no synchronisation needed
between launches.

This is implemented as `KernelVariant.TRITON_BUCKETED` sitting above `TRITON` in the
`KernelDispatcher` order. The existing `TRITON` variant is kept as an explicit override
(via `kernel_variant="triton"`) for comparison.

---

## AOT Compilation: Eliminating JIT Overhead

### The JIT problem

Triton compiles a new GPU binary the first time a kernel is called with a previously-unseen
combination of `tl.constexpr` values. Each bucket variant takes 2–8 seconds to compile.
Triton caches compiled binaries to `~/.triton/cache/` persistently on disk — the cost is
paid **once ever** per unique configuration, not once per session. But the first run after
a Triton version bump or constexpr change will stall.

### Option 1 (recommended): Kernel warmup at MASE pass application time

The MASE pass already knows all constexpr values when it runs — `H`, `D`, and the set of
occupied bucket sizes are all determined from `group_ids` before any inference. Trigger
compilation there:

```python
# At the end of fast_group_attention_transform_pass, after module replacement
for bucket_size, bucket in partition.buckets.items():
    block_m = min(bucket_size, 32)
    block_n = min(bucket_size, 32)
    tiles_per_group = triton.cdiv(bucket_size, block_m)
    block_d = triton.next_power_of_2(D)

    dummy_q   = torch.zeros(1, H, bucket.num_groups, D, device=device, dtype=dtype)
    dummy_out = torch.empty_like(dummy_q)
    dummy_cu  = bucket.cu_seqlens.to(device)

    _tiled_grouped_attn_fwd.warmup(
        dummy_q, dummy_q, dummy_q, dummy_out,
        dummy_cu, 1.0,
        *[s for s in dummy_q.stride() for _ in range(4)],  # Q/K/V/Out strides
        H=H, G=bucket.num_groups, D=D,
        TILES_PER_GROUP=tiles_per_group,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        grid=(1,),
    )
```

Warmup runs at model setup time, not inference time. The first real forward pass hits the
disk cache and incurs zero compilation overhead.

### Option 2: Full AOT via `triton.compile()`

Triton 2.1+ exposes `triton.compile()` which produces a `.cubin` binary entirely ahead of
time, with no warmup needed at model load:

```python
from triton.compiler import compile as triton_compile

compiled = triton_compile(
    _tiled_grouped_attn_fwd,
    signature={
        "Q": "*fp16", "K": "*fp16", "V": "*fp16", "Out": "*fp16",
        "cu_seqlens": "*i32", "scale": "fp32",
        "stride_qt": "i32", "stride_qh": "i32", "stride_qb": "i32", "stride_qd": "i32",
        # ... remaining stride args
    },
    constants={
        "H": H, "G": num_groups, "D": D,
        "TILES_PER_GROUP": tiles_per_group,
        "BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_D": block_d,
    },
    num_warps=4,
    num_stages=2,
)
```

**Limitations:**
- `.cubin` is GPU-architecture-specific (SM80 for A100, SM89 for RTX 4090) — one binary per target GPU
- `triton.compiler` API has shifted across Triton 2.x → 3.x and is less stable
- All constexpr values and tensor dtypes must be fixed at build time

**Recommendation:** Use Option 1 (warmup at pass time) for all development and standard
deployment. Only reach for full AOT if deploying to an environment where disk writes are
forbidden or you need hermetic, reproducible binaries.

---

## Running the Benchmark

```bash
# From the repo root
uv run python test/passes/graph/transforms/timeseries/bench_bucketed_attn.py
```

Requires CUDA. The benchmark prints a table comparing baseline `GroupSelfAttention`,
`PACKED_SPARSE`, `TRITON` (single launch), and `TRITON_BUCKETED` across uniform and
mixed group-size workloads. The correctness test suite runs with:

```bash
uv run pytest test/passes/graph/transforms/timeseries/test_bucketed_attn.py -v
```

---

## Benchmark Results

Measured on a single GPU (`d_model=512, n_heads=8, d_kv=64, T=64`).
Times are median milliseconds over 100 iterations after 20 warmup runs.

```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Case                    Description                        Baseline       Packed       Triton     Bucketed  vs Packed  vs Triton
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
uniform g=2 B=128       64 groups of 2                        8.708        6.778        9.903        8.303      0.82x      1.19x
uniform g=2 B=256       128 groups of 2                      24.309       13.537       19.605       16.540      0.82x      1.19x
uniform g=4 B=128       32 groups of 4                        8.777        6.669        8.571        7.849      0.85x      1.09x
uniform g=4 B=256       64 groups of 4                       24.431       13.159       16.811       15.261      0.86x      1.10x
uniform g=8 B=128       16 groups of 8                        8.810        6.671        7.885        7.412      0.90x      1.06x
uniform g=8 B=256       32 groups of 8                       24.499       13.284       15.385       14.626      0.91x      1.05x
mix2+4 B=128            32×2 + 16×4                           8.833        9.339        9.298        8.287      1.13x      1.12x
mix2+4 B=256            64×2 + 32×4                          24.492       18.513       18.251       16.148      1.15x      1.13x
mix2+4+8 B=112          20×2 + 10×4 + 4×8                    7.164       12.594        7.801        7.269      1.73x      1.07x
mix2+4+8 B=224          40×2 + 20×4 + 8×8                   20.433       25.171       15.384       14.066      1.79x      1.09x
realistic B=226         50×1 + 50×2 + 15×4 + 2×8            20.780       41.856       17.718       15.357      2.73x      1.15x
realistic B=452         100×1 + 100×2 + 30×4 + 4×8          61.622       84.243       35.401       29.986      2.81x      1.18x
mix2+4+8 T=16           3-bucket mix, T=16                    1.857        3.428        2.203        2.366      1.45x      0.93x
mix2+4+8 T=32           3-bucket mix, T=32                    3.787        6.513        4.133        4.039      1.61x      1.02x
mix2+4+8 T=64           3-bucket mix, T=64                    7.266       12.929        7.907        7.352      1.76x      1.08x
mix2+4+8 T=128          3-bucket mix, T=128                  14.631       25.774       15.610       14.320      1.80x      1.09x
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Observations

**Uniform batches (one occupied bucket):**
Bucketed is 8–19% faster than the old single-launch `TRITON` (which uses `BLOCK_M=32`
even for pairs), but 8–18% slower than `PACKED_SPARSE`. With only one bucket, bucketed
reduces to a single launch with the same multi-launch overhead but no advantage. For
purely uniform batches, `PACKED_SPARSE` is the right choice; `KernelDispatcher` could
detect this case and fall back. Bucketed still beats the `TRITON` variant in all
uniform cases.

**2-bucket mixed batches:**
Bucketed beats both alternatives: 13–15% faster than packed-sparse and 12–13% faster
than the single-launch Triton. The batched padding waste in packed-sparse (pads all
groups to the global max) starts to show.

**3-bucket mixed batches:**
1.73–1.79× faster than packed-sparse, 7–9% faster than single-launch Triton.

**Realistic 4-bucket workload (univariate + pairs + quads + octets):**
The strongest case: **2.73–2.81× faster than packed-sparse** and **15–18% faster than
single-launch Triton**. The packed-sparse path is catastrophic here because it must pad
every group (including size-1 and size-2) to the global max of 8.

**T sweep:**
Speedup vs packed-sparse grows with T (from 1.45× at T=16 to 1.80× at T=128). At
T=16, bucketed is marginally slower than single-launch Triton (0.93×) — the multi-launch
overhead dominates at short sequences where per-tile compute is very cheap. At T≥32
bucketed is faster.

### The BLOCK_M floor constraint

Triton's `tl.dot` requires M, N, K ≥ 16 (tensor-core minimum). The current
implementation clamps `BLOCK_M = max(min(bucket_size, 32), 16)`, so the bucket-2, 4,
and 8 tile sizes are all 16×16 rather than the ideal 2×2, 4×4, or 8×8. This caps the
score-matrix utilisation improvement for the smallest groups but the overall benefit is
still real — the reduction in cross-group tile-count waste dominates.

A future optimisation for groups with size < 16 is to use a scalar accumulation path
(register loops with `tl.sum`) instead of `tl.dot`, which would recover full utilisation
for those buckets.

---

## Benefits Summary

| | Baseline | Packed Sparse | Triton (single) | **Bucketed Triton** |
|---|---|---|---|---|
| Univariate series skip Q/K | No | No | No | Yes (bucket ≠ needed) |
| Padding per group | To global max | To global max | To global max | To next power of 2 |
| Effective BLOCK_M for pairs | 32 | N/A | 32 | 16 (floor) |
| Speedup vs packed, realistic mix | — | 1× | 0.42× | **2.8×** |
| Speedup vs Triton, realistic mix | — | 0.42× | 1× | **1.18×** |
| Kernel launches per forward | 1 | 1 | 1 | 1 per occupied bucket (≤ 7) |
| New kernel code required | — | — | — | No — same kernel, new dispatch |
| JIT compilation variants | 1 | 0 | 1 | 1 per bucket (≤ 7, cached to disk) |

---

## Open Questions

1. **`tl.dot` floor at 16** — `BLOCK_M` is clamped to `max(bucket_size, 16)` to satisfy
   the tensor-core minimum. Groups of size 2–8 run a 16×16 tile with heavy masking rather
   than the ideal 2×2/4×4/8×8. A scalar-accumulation kernel path for groups < 16 would
   eliminate this waste. Worth implementing before declaring the bucketed path fully
   optimal for small-group workloads.

2. **Uniform-batch regression** — for a perfectly uniform batch (one occupied bucket),
   bucketed loses to packed-sparse by ~15% due to Triton launch overhead. The dispatcher
   could detect a single-bucket partition and fall back to `PACKED_SPARSE` for the
   `max_group_size ≤ TRITON_CROSSOVER` range. The crossover point appears to be around
   `max_group_size = 8–16` from the benchmark.

3. **Short-sequence launch overhead** — at T=16, the 3-bucket mix is marginally slower
   than single-launch Triton (0.93×). The per-launch scheduling cost dominates when
   per-tile compute is tiny. A minimum-group-count threshold per bucket (skip the kernel
   launch if fewer than N groups occupy a bucket, fall back to packed-sparse for those
   groups) would cap this.

4. **Register pressure for large buckets** — for `BLOCK_M=32, D=64` the accumulator
   `acc[BLOCK_M, BLOCK_D]` is `32×64×4 = 8 KB` per program. For `D=128` it doubles to
   16 KB, reducing SM occupancy. The `BLOCK_M=32` cap for large buckets is the right
   call; the tiling loop handles groups larger than 32 without register blowup.

5. **Bucket granularity** — powers of 2 (2, 4, 8, 16, 32, 64) minimise JIT variants.
   Non-power-of-2 buckets would reduce padding waste for groups of size 3, 5, 6, 7 but
   multiply warmup time and compiled variants combinatorially. Powers of 2 are the right
   default; revisit only if profiling shows significant waste in the 3–7 range.
