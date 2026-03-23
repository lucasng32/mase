# `triton_rope_attn.py` — Explained

## Module-level setup

```python
from __future__ import annotations
import logging
import torch
```
Standard imports. `annotations` defers type evaluation so `"torch.Tensor"` string annotations work before Triton is imported.

```python
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
```
Triton is optional. The whole kernel block is guarded by `if _TRITON_AVAILABLE`, so the file imports cleanly on CPU-only machines.

```python
_SMALL_S_THRESHOLD = 256
```
Runtime dispatch boundary. Sequences shorter than this use the inline-RoPE kernel; longer sequences use pre-rotation. Exposed as a module constant so callers can override it.

---

## Small-S kernel constants

```python
_SMALL_BLOCK_M = 32
_SMALL_BLOCK_N = 32
```
Fixed tile sizes for the small-S kernel — no autotuning. At tiny grids (e.g. B=1, H=8, S=64 → 16 CTAs), autotuning overhead would take longer than the actual computation.

---

## `_fused_rope_attn_small` — fully fused single kernel

### Signature
```python
@triton.jit
def _fused_rope_attn_small(
    Q, K, V, Out, Mask, InvFreq,
    stride_qb, stride_qh, stride_qs, stride_qd,
    ...
    H, S, D,
    HALF_D: tl.constexpr, BLOCK_HD: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
```
All tensors are passed as raw pointers. Strides are passed explicitly because Triton kernels cannot introspect tensor metadata. `tl.constexpr` parameters are compile-time constants — Triton generates specialised PTX for each unique combination.

### Grid decode
```python
pid         = tl.program_id(0)          # which CTA am I?
num_q_tiles = tl.cdiv(S, BLOCK_M)       # how many Q-tiles total?
q_tile      = pid % num_q_tiles          # which Q-tile does this CTA own?
tmp         = pid // num_q_tiles
h           = tmp % H                    # which head?
b           = tmp // H                   # which batch item?
```
The 1-D grid is logically `(batch, head, q_tile)` flattened. Each CTA is responsible for exactly one `(b, h, q_tile)` triple. This keeps the grid launch simple while covering all parallelism dimensions.

### Tile index vectors
```python
q_range = q_start + tl.arange(0, BLOCK_M)    # absolute seq positions for this Q-tile
q_valid = q_range < S                          # out-of-bounds mask for padding
hd      = tl.arange(0, BLOCK_HD)              # [0 .. HALF_D), padded to power-of-2
d_all   = tl.arange(0, BLOCK_D)               # [0 .. D), padded to power-of-2
```
`BLOCK_HD` and `BLOCK_D` are powers of 2 so Triton can emit aligned vector loads. The `_valid` masks prevent loading garbage from out-of-bounds positions.

### Base pointers
```python
Q_base = Q   + b * stride_qb + h * stride_qh
K_base = K   + b * stride_kb + h * stride_kh
...
```
Precomputed once per CTA. All subsequent loads add only the seq/dim offsets on top, keeping pointer arithmetic cheap.

### Q load and RoPE (outside the inner loop)
```python
inv_freq = tl.load(InvFreq + hd, mask=hd_valid, other=0.0).to(tl.float32)

q_fst = tl.load(Q_base + q_range[:,None]*stride_qs + hd[None,:]*stride_qd, ...)   # dims [0..HALF_D)
q_snd = tl.load(Q_base + q_range[:,None]*stride_qs + (hd[None,:]+HALF_D)*stride_qd, ...)  # dims [HALF_D..D)
```
Q is split into two `(BLOCK_M, BLOCK_HD)` half-dim strips because Triton has no runtime slice syntax — you can't do `q[:, :HALF_D]` on a register tile. Loading as two separate strips lets RoPE be applied with element-wise ops only.

```python
q_pos     = q_range[:,None].to(tl.float32) * inv_freq[None,:]  # (BLOCK_M, BLOCK_HD) angle matrix
q_cos     = tl.cos(q_pos)
q_sin     = tl.sin(q_pos)
q_rot_fst = q_fst * q_cos - q_snd * q_sin   # RoPE first half
q_rot_snd = q_snd * q_cos + q_fst * q_sin   # RoPE second half
```
Standard RoPE rotation formula. `tl.cos/sin` operate element-wise on the register tile — the rotated Q values live entirely in registers throughout the kernel.

### Online softmax accumulators
```python
m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # running row-max
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running normaliser
acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)         # running weighted-V sum
```
These three register arrays implement the [Flash Attention](https://arxiv.org/abs/2205.14135) online softmax — the full attention matrix is never materialised in SRAM or GMEM.

### Inner K/V streaming loop
```python
for k_start in range(0, S, BLOCK_N):
    k_range = k_start + tl.arange(0, BLOCK_N)
    k_valid = k_range < S
```
Iterates over every K-tile. This is the loop that makes the kernel O(S²) in compute but O(S) in peak memory.

```python
    k_fst = tl.load(K_base + k_range[:,None]*stride_ks + hd[None,:]*stride_kd, ...)
    k_snd = tl.load(K_base + k_range[:,None]*stride_ks + (hd[None,:]+HALF_D)*stride_kd, ...)
    k_pos     = k_range[:,None].to(tl.float32) * inv_freq[None,:]
    k_rot_fst = k_fst * k_cos - k_snd * k_sin
    k_rot_snd = k_snd * k_cos + k_fst * k_sin
```
**Small-S path specific:** K RoPE is computed inline here, once per K-tile per Q-tile. At S=64 with BLOCK_M=32, each K position is rotated only `S/BLOCK_M = 2` times total — the redundancy is tiny. The benefit is no `K_rot` allocation in GMEM.

```python
    s = tl.dot(q_rot_fst, tl.trans(k_rot_fst)) + tl.dot(q_rot_snd, tl.trans(k_rot_snd))
```
Attention score matrix `(BLOCK_M, BLOCK_N)` computed as the sum of two half-dim matmuls. This is mathematically identical to a single full-dim `q_rot @ k_rot^T` — split by the half-dim trick needed for RoPE.

```python
    s = s + mask_tile
    s = tl.where(k_valid[None,:], s, float("-inf"))
    s = tl.where(q_valid[:,None], s, float("-inf"))
```
Adds the additive attention mask (causal or padding, pre-supplied as `-inf` for blocked positions), then forces out-of-bounds K and Q positions to `-inf` so they contribute zero after softmax.

```python
    m_new = tl.maximum(m_i, tl.max(s, axis=1))   # new row-wise max
    alpha = tl.exp(m_i - m_new)                   # rescale factor for previous acc
    p     = tl.exp(s - m_new[:,None])             # softmax numerators (not yet normalised)
    l_i   = l_i * alpha + tl.sum(p, axis=1)       # update normaliser
    acc   = acc * alpha[:,None]                    # rescale previous accumulator
    m_i   = m_new
```
Flash Attention online softmax update. Each tile rescales the running accumulator so the final result is numerically equivalent to computing softmax over the full row at once — without storing the full row.

```python
    v = tl.load(V_base + k_range[:,None]*stride_vs + d_all[None,:]*stride_vd, ...)
    acc += tl.dot(p, v)
```
V is loaded as a single full-D `(BLOCK_N, D)` tile (no half-dim split needed — V is not RoPE-rotated). `tl.dot(p, v)` accumulates the weighted value sum into `acc`.

### Normalise and store
```python
safe_l = tl.where(l_i > 0.0, l_i, 1.0)    # guard against all-masked rows
acc    = acc / safe_l[:,None]               # divide by softmax denominator
tl.store(O_base + q_range[:,None]*stride_os + d_all[None,:]*stride_od, acc, ...)
```
Single full-D store. The output is the correctly normalised attention-weighted sum of V, with RoPE applied to Q and K.

---

## `_prerotate_k` — K pre-rotation kernel (large-S path only)

```python
@triton.jit
def _prerotate_k(K, Krot, InvFreq, ..., BLOCK_S: tl.constexpr):
```
Takes raw `K` and writes RoPE-rotated `Krot`. Grid is `(B * H * cdiv(S, BLOCK_S),)` — purely parallel over the sequence dimension.

```python
    s_range = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    ...
    k_rot_fst = k_fst * cos_p - k_snd * sin_p
    k_rot_snd = k_snd * cos_p + k_fst * sin_p
    tl.store(Ko_base + ..., k_rot_fst, ...)
    tl.store(Ko_base + ... + HALF_D, k_rot_snd, ...)
```
Identical RoPE formula, but written to GMEM instead of accumulating further. The two stores write the two half-dim strips back into a layout-identical output buffer. Cost: one read + one write of the full K tensor — O(S) per head.

---

## `_fused_rope_attn_fwd` — autotuned attention kernel (large-S path only)

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        ...
    ],
    key=["S", "HALF_D"],
)
```
Triton will benchmark all 6 configs on the first call for each unique `(S, HALF_D)` pair and cache the winner. `num_stages=2` enables software pipelining — the next K-tile load is issued while the current tile's matmul executes, hiding memory latency. At large S this recovers the L2/DRAM bandwidth the pre-rotation write costs.

The kernel body is structurally identical to `_fused_rope_attn_small`, with two differences:
- `Krot` is passed instead of `K` — **no trig in the inner loop**, just two loads and two matmuls per K-tile.
- `BLOCK_M` and `BLOCK_N` are autotuned `constexpr` parameters rather than fixed constants.

---

## `triton_fused_rope_attention` — Python entry point

```python
if S < _SMALL_S_THRESHOLD:
    grid = (B * H * triton.cdiv(S, _SMALL_BLOCK_M),)
    _fused_rope_attn_small[grid](Q, K, V, Out, mask, inv_freq, ..., num_warps=4)
else:
    K_rot = torch.empty_like(K)
    _prerotate_k[pre_grid](K, K_rot, ...)
    _fused_rope_attn_fwd[grid](Q, K_rot, V, Out, ...)
```
Pure Python dispatch — no CUDA synchronisation needed between the two conditions. For the large path, `K_rot` is allocated as a temporary and freed when the function returns; the autotuned kernel's `grid` is a lambda so Triton can substitute the winning `BLOCK_M` at call time.
