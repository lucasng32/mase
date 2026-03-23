# Triton Grouped Attention Kernel Walkthrough

A line-by-line explanation of `triton_grouped_attn.py`.

---

## Triton Fundamentals

Before reading the code, there are a few concepts unique to Triton:

- **Program**: The unit of parallel work. Every program instance runs the same kernel code but with a different `program_id`. Many programs run in parallel on the GPU. There is no concept of a single thread — you always think in tiles.
- **Tile**: A block of rows/columns that a single program owns and operates on. Tile sizes must be powers of 2.
- **`tl.constexpr`**: A value known at compile time. Triton JIT-compiles a separate GPU binary for each unique combination of constexprs. This is how it specialises tile sizes without runtime branching.
- **Pointer arithmetic**: Inside a Triton kernel, tensors are raw pointers. You navigate to elements manually using strides (bytes-per-step in each dimension). There is no `.shape` or `.stride()` available — those are passed in as arguments.

---

## Python Entry Point: `triton_grouped_attention`

```python
T, H, B, D = Q.shape
```
Unpack shape dimensions: T = timesteps, H = attention heads, B = batch size (pre-sorted by group), D = per-head feature dimension.

```python
Out = torch.empty_like(Q)
```
Allocate the output tensor with the same shape and dtype as Q. The Triton kernel writes directly into this — it is not returned from the kernel, it is mutated in-place.

```python
BLOCK_D = triton.next_power_of_2(D)
```
Round D up to the nearest power of 2. `tl.dot` (tensor core matmul) requires power-of-2 tile dimensions. If D=64 → BLOCK_D=64. If D=96 → BLOCK_D=128. The extra columns are masked off during loads.

```python
max_group_size = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
```
`cu_seqlens` is a cumulative array e.g. `[0, 3, 5, 8]`. Differencing it gives per-group sizes `[3, 2, 3]`. Taking the max gives the size of the largest group. This is used to determine how many query tiles are needed.

```python
tiles_per_group = triton.cdiv(max_group_size, BLOCK_M_DEFAULT)
```
`triton.cdiv` is ceiling division. How many `BLOCK_M=32`-row tiles does the largest group need? e.g. max_group_size=64 → 2 tiles. This becomes a single constant applied to **all** groups — the root of the mixed-size inefficiency, because small groups get the same tile count as the largest one.

```python
grid = (T * H * num_groups * tiles_per_group,)
```
The total number of programs to launch, as a flat 1D grid. Each program will independently decode which `(timestep, head, group, query-tile)` it owns from its flat ID.

```python
_tiled_grouped_attn_fwd[grid](
    Q, K, V, Out, cu_seqlens, scale,
    Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
    K.stride(0), K.stride(1), K.stride(2), K.stride(3),
    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
    Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
    H=H, G=num_groups, D=D,
    TILES_PER_GROUP=tiles_per_group,
    BLOCK_M=BLOCK_M_DEFAULT,
    BLOCK_N=BLOCK_N_DEFAULT,
    BLOCK_D=BLOCK_D,
)
```
Launch the kernel. `[grid]` sets how many programs run. Strides are passed explicitly because inside the kernel you cannot call `.stride()` on a pointer — you must navigate memory using pre-computed step sizes. For a contiguous `(T, H, B, D)` tensor: `stride(0)=H*B*D`, `stride(1)=B*D`, `stride(2)=D`, `stride(3)=1`.

---

## Kernel: `_tiled_grouped_attn_fwd`

### Signature

```python
@triton.jit
def _tiled_grouped_attn_fwd(
    Q, K, V, Out, cu_seqlens, scale,
    stride_qt, stride_qh, stride_qb, stride_qd,
    stride_kt, stride_kh, stride_kb, stride_kd,
    stride_vt, stride_vh, stride_vb, stride_vd,
    stride_ot, stride_oh, stride_ob, stride_od,
    H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    TILES_PER_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
```

`@triton.jit` marks this function as a GPU kernel — Triton compiles it to PTX (GPU assembly). `Q, K, V, Out` are raw memory pointers, not tensors. The `tl.constexpr` arguments are baked into the compiled binary, allowing the compiler to unroll loops and optimise tile sizes statically.

---

### Decode Program ID

```python
pid = tl.program_id(0)
```
Returns the integer ID of this program instance on grid axis 0. Ranges from 0 to `grid[0] - 1`. This is the only piece of information that differs between parallel program instances.

```python
q_tile = pid % TILES_PER_GROUP
tmp    = pid // TILES_PER_GROUP
g      = tmp % G
tmp    = tmp // G
h      = tmp % H
t      = tmp // H
```
Unpack the flat `pid` into a 4-tuple `(t, h, g, q_tile)` using modulo/divide — the same as converting a flat array index to multi-dimensional indices. After this, each program knows exactly which timestep, head, group, and query-tile chunk it is responsible for.

---

### Group Bounds

```python
g_start = tl.load(cu_seqlens + g)
g_end   = tl.load(cu_seqlens + g + 1)
g_size  = g_end - g_start
```
`tl.load(ptr)` reads a single scalar value from GPU memory at the given pointer address. `cu_seqlens` is e.g. `[0, 3, 5, 8]` — loading index `g` and `g+1` gives the start and end positions of this group's slice in the sorted batch dimension. `g_size` is the number of sequences in this group.

```python
q_begin = q_tile * BLOCK_M
if q_begin >= g_size:
    return
```
This tile's starting row within the group. If the group is smaller than the tile offset (e.g. group has 3 elements but this is tile index 1 which starts at row 32), this program has no real work to do and exits immediately. **This early return is the wasted-work path for small groups in a mixed-size batch.**

---

### Build Q Tile Pointers

```python
q_offs = g_start + q_begin + tl.arange(0, BLOCK_M)
```
`tl.arange(0, BLOCK_M)` produces the vector `[0, 1, 2, ..., BLOCK_M-1]` in registers. Adding `g_start + q_begin` shifts it to the actual batch indices this program handles — e.g. if `g_start=3`, `q_begin=0`: `q_offs = [3, 4, 5, ..., 34]`.

```python
q_valid = tl.arange(0, BLOCK_M) < (g_size - q_begin)
```
Boolean mask vector of length `BLOCK_M`. Marks which rows in this tile are real data vs padding. If the group has 5 elements and `q_begin=0`: `q_valid = [T, T, T, T, T, F, F, ..., F]`.

```python
d_range = tl.arange(0, BLOCK_D)
d_valid = d_range < D
```
Same idea for the feature dimension. `BLOCK_D` is rounded up to a power of 2, so `d_valid` masks off the extra columns that don't correspond to actual data.

```python
q_ptrs = (
    Q
    + t * stride_qt
    + h * stride_qh
    + q_offs[:, None] * stride_qb
    + d_range[None, :] * stride_qd
)
```
Compute a 2D array of pointers of shape `(BLOCK_M, BLOCK_D)`. `q_offs[:, None]` is a column vector `(BLOCK_M, 1)` and `d_range[None, :]` is a row vector `(1, BLOCK_D)` — broadcasting gives every `(row, col)` combination. Multiplying by strides converts logical indices into byte offsets from the base pointer `Q`. This gives the exact memory address of each element in the tile.

```python
q_tile_data = tl.load(
    q_ptrs, mask=q_valid[:, None] & d_valid[None, :], other=0.0
).to(tl.float32)
```
Load the entire `(BLOCK_M, BLOCK_D)` tile from GPU memory in one operation. `mask` is a 2D boolean array — positions where the mask is False are not read from memory and are filled with `other=0.0` instead (prevents out-of-bounds access). `.to(tl.float32)` upcasts to float32 for numerical stability during the softmax computation.

---

### Online Softmax Accumulators

```python
m_i  = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
```
Initialise the three state variables for **online (Flash Attention) softmax** — one scalar per query row for `m_i` and `l_i`, one vector per query row for `acc`:

- `m_i` — running maximum of attention scores seen so far. Starts at `-inf`. Used for numerical stability: instead of computing `exp(score)`, we compute `exp(score - max)`.
- `l_i` — running sum of `exp(score - max)` across all K/V tiles seen so far. This is the softmax denominator.
- `acc` — running weighted sum of V values. This is the softmax numerator (the actual output).

These three together allow computing the full softmax attention in a **single left-to-right pass** over K/V tiles without ever storing all scores simultaneously. This is the Flash Attention algorithm.

---

### Loop Over K/V Tiles

```python
for k_begin in range(0, g_size, BLOCK_N):
```
Iterate over this group's keys and values in chunks of `BLOCK_N` rows. This is a Python `range` written inside a `@triton.jit` function — Triton compiles it into a GPU loop (or unrolls it if the count is small enough).

```python
k_offs  = g_start + k_begin + tl.arange(0, BLOCK_N)
k_valid = tl.arange(0, BLOCK_N) < (g_size - k_begin)
```
Same offset/mask pattern as for Q but for the current K tile. `k_valid` marks which columns correspond to real keys vs padding at the end of the group.

```python
k_ptrs = (
    K
    + t * stride_kt + h * stride_kh
    + k_offs[:, None] * stride_kb
    + d_range[None, :] * stride_kd
)
k_tile_data = tl.load(k_ptrs, mask=k_valid[:, None] & d_valid[None, :], other=0.0).to(tl.float32)
```
Compute pointer array and load K tile `(BLOCK_N, BLOCK_D)` from memory, same approach as Q.

```python
s = tl.dot(q_tile_data, tl.trans(k_tile_data)) * scale
```
`tl.dot` performs a matrix multiply that dispatches to GPU tensor cores: `(BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N)` → `(BLOCK_M, BLOCK_N)`. `tl.trans` transposes K in-register (no memory movement). The result `s` is the raw attention score matrix for this Q-tile vs this K-tile. `scale` is 1.0 here — Chronos2 does not use the standard `1/sqrt(D)` scaling.

```python
s = tl.where(k_valid[None, :], s, float("-inf"))
s = tl.where(q_valid[:, None], s, float("-inf"))
```
`tl.where(condition, true_val, false_val)` — element-wise conditional, like `torch.where`. First line: set scores for padded key positions to `-inf` so softmax gives them zero weight. Second line: set scores for padded query rows to `-inf` so they don't corrupt the running maximum.

---

### Online Softmax Update (Flash Attention Recurrence)

```python
m_new = tl.maximum(m_i, tl.max(s, axis=1))
```
`tl.max(s, axis=1)` reduces each query row of `s` to its maximum score value — shape `(BLOCK_M,)`. `tl.maximum` takes the element-wise maximum with the previous running max `m_i`. `m_new` is the updated running max after seeing this K tile.

```python
alpha = tl.exp(m_i - m_new)
```
Correction factor for the existing accumulator. If the running max increased (new, larger scores were found), all previously accumulated values were computed relative to the old max and need to be rescaled down. `alpha <= 1.0` always.

```python
p = tl.exp(s - m_new[:, None])
```
Unnormalised softmax probabilities for this K tile, stabilised by subtracting the new running max. Shape `(BLOCK_M, BLOCK_N)`.

```python
l_i = l_i * alpha + tl.sum(p, axis=1)
```
Update the running denominator: rescale old sum by `alpha` (correction for new max), then add the sum of new probabilities across this K tile.

```python
acc = acc * alpha[:, None]
```
Rescale the existing output accumulator by `alpha`. Same correction — the old weighted V sum was computed relative to the old max.

```python
acc += tl.dot(p.to(tl.float32), v_tile_data)
```
Load V tile and accumulate: `(BLOCK_M, BLOCK_N) @ (BLOCK_N, BLOCK_D)` → `(BLOCK_M, BLOCK_D)`. Adds the contribution of this K/V tile to the running output. Note V is loaded here (after `p` is computed) to avoid holding it in registers during the Q@K matmul.

```python
m_i = m_new
```
Advance the running max for the next K/V tile iteration.

---

### Normalise and Store

```python
safe_l = tl.where(l_i > 0.0, l_i, 1.0)
acc    = acc / safe_l[:, None]
```
After all K/V tiles are processed, divide each row of `acc` by its softmax denominator `l_i` to produce the final normalised output. `tl.where` replaces any zero denominators (fully-masked query rows) with 1.0 to avoid divide-by-zero — those rows will output zeros anyway since `acc` was never updated for them.

```python
o_ptrs = (
    Out
    + t * stride_ot + h * stride_oh
    + q_offs[:, None] * stride_ob
    + d_range[None, :] * stride_od
)
tl.store(
    o_ptrs,
    acc.to(Out.dtype.element_ty),
    mask=q_valid[:, None] & d_valid[None, :],
)
```
Compute output pointer array (same shape and logic as Q pointers). `tl.store` writes the tile back to GPU memory. `.to(Out.dtype.element_ty)` casts from float32 back to the original dtype of `Out` (e.g. float16). The mask prevents writing to padded positions.

---

## Triton Instruction Reference

| Instruction | What it does |
|---|---|
| `tl.program_id(axis)` | Returns the integer ID of this program instance on the given grid axis |
| `tl.arange(start, end)` | Returns a vector `[start, start+1, ..., end-1]` in registers — like `torch.arange` but inside the kernel |
| `tl.full([N], val, dtype)` | Creates a vector of length N filled with `val` |
| `tl.zeros([M, N], dtype)` | Creates an M×N tile of zeros |
| `tl.load(ptr, mask, other)` | Loads a tile from GPU memory; positions where `mask=False` return `other` instead of reading memory |
| `tl.store(ptr, val, mask)` | Writes a tile to GPU memory; positions where `mask=False` are not written |
| `tl.dot(A, B)` | Tile matrix multiply dispatched to tensor cores: `(M,K) @ (K,N) → (M,N)` |
| `tl.trans(A)` | Transposes a tile in-register with no memory movement |
| `tl.where(cond, a, b)` | Element-wise conditional — like `torch.where` |
| `tl.max(x, axis)` | Reduce a tile along an axis, returning the maximum |
| `tl.sum(x, axis)` | Reduce a tile along an axis, returning the sum |
| `tl.exp(x)` | Element-wise exponential |
| `tl.maximum(a, b)` | Element-wise maximum of two vectors |
| `tl.constexpr` | Value baked into the compiled binary — enables static tile-size specialisation |
| `ptr + offset` | Manual pointer arithmetic to navigate to a specific memory address |
| `x[:, None]` | Add a size-1 axis (column vector) — same semantics as NumPy/PyTorch |
| `x[None, :]` | Add a size-1 axis (row vector) — used with broadcasting in pointer arithmetic |
| `Out.dtype.element_ty` | The Triton dtype of the output tensor's elements (e.g. `tl.float16`) |
| `triton.next_power_of_2(n)` | Round n up to the nearest power of 2 — used to set BLOCK_D |
| `triton.cdiv(a, b)` | Ceiling integer division — used to compute number of tiles |
