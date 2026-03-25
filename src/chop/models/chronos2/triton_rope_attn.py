"""
Triton kernel for fused RoPE + multi-head self-attention.

Two dispatch paths, chosen at runtime based on sequence length:

Small-S path  (S < ``_SMALL_S_THRESHOLD``, default 128)
---------------------------------------------------------
``_fused_rope_attn_small`` — fully register-fused single kernel.
K-side RoPE is recomputed inside the inner K-tile loop (the O(S²) trig
cost is negligible when S is small — at S=64 with BLOCK_M=32 each K
position is rotated only twice).  The benefit is zero GMEM allocation for
a rotated-K buffer, which matters when kernel-launch overhead dominates.
Fixed tile sizes (BLOCK_M=32, BLOCK_N=32, num_warps=4) — no autotune
overhead, which would dwarf the tiny computation at small S.
V is still loaded as a single full-D ``(BLOCK_N, D)`` tile (improvement
over v1's two half-dim strips, regardless of S).

Large-S path  (S ≥ ``_SMALL_S_THRESHOLD``)
-------------------------------------------
Two-phase implementation:

1. ``_prerotate_k`` — lightweight kernel that rotates the full K tensor
   once (O(S) cost) into a ``K_rot`` scratch buffer.  This eliminates the
   O(S/BLOCK_M) redundant trig passes that hurt v1 at large S.

2. ``_fused_rope_attn_fwd`` — autotuned FlashAttention-style attention
   loop.  ``@triton.autotune`` sweeps BLOCK_M ∈ {32,64,128},
   BLOCK_N ∈ {64,128}, num_warps ∈ {4,8}, num_stages ∈ {1,2}.
   The first call for a given (B, H, S, HALF_D) shape benchmarks 6 configs;
   subsequent calls use the cached best.  B and H are included in the key
   so that large-batch shapes (where bigger BLOCK_M amortises per-CTA setup
   cost) get a different cached config from small-batch shapes (where smaller
   BLOCK_M improves SM occupancy).

Common design
-------------
RoPE on a head vector ``x`` of dimension ``D = 2 * HALF_D``:

    x_rot[i]           = x[i]         * cos(pos * inv_freq[i])
                         - x[i+HALF_D] * sin(pos * inv_freq[i])
    x_rot[i + HALF_D]  = x[i+HALF_D]  * cos(pos * inv_freq[i])
                         + x[i]        * sin(pos * inv_freq[i])

Q is loaded in two half-dim strips (required for element-wise rotation
without runtime slicing in Triton).  Attention scores are the sum of two
``(BLOCK_M, BLOCK_N)`` matmuls over the two half-dims.  No scaling
(``scale = 1.0``) to match the original Chronos-2 implementation.

Requires: ``triton`` (ships with PyTorch on CUDA).
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


# Sequence length below which the fully-fused single-kernel path is used.
# Below this threshold the K_rot GMEM roundtrip cost of pre-rotation exceeds
# the (small) O(S²) trig savings.  At exactly the threshold, both paths are
# comparable; we defer to the pre-rotation path for safety.
_SMALL_S_THRESHOLD = 256

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    # ── Small-S kernel: fully fused, K RoPE inline, fixed tiles ─────────────

    _SMALL_BLOCK_M = 32
    _SMALL_BLOCK_N = 32

    @triton.jit
    def _fused_rope_attn_small(
        Q,        # (B, H, S, D)
        K,        # (B, H, S, D) — raw (unrotated) keys
        V,        # (B, H, S, D)
        Out,      # (B, H, S, D)
        Mask,     # (B, 1, S, S) — additive float mask
        InvFreq,  # (HALF_D,)
        # Q strides
        stride_qb, stride_qh, stride_qs, stride_qd,
        # K strides
        stride_kb, stride_kh, stride_ks, stride_kd,
        # V strides
        stride_vb, stride_vh, stride_vs, stride_vd,
        # Out strides
        stride_ob, stride_oh, stride_os, stride_od,
        # Mask strides: (B, 1, S, S) — skip the head-1 stride
        stride_mb, stride_mq, stride_mk,
        # Dimensions
        H, S, D,
        HALF_D:   tl.constexpr,
        BLOCK_HD: tl.constexpr,   # next_power_of_2(HALF_D)
        BLOCK_D:  tl.constexpr,   # 2 * BLOCK_HD
        BLOCK_M:  tl.constexpr,   # fixed: _SMALL_BLOCK_M
        BLOCK_N:  tl.constexpr,   # fixed: _SMALL_BLOCK_N
    ):
        """
        Fully register-fused kernel for small sequences.

        Grid: (B * H * cdiv(S, BLOCK_M),)

        K-side RoPE is computed inline per K-tile — no pre-rotation buffer.
        At S < 128 the number of Q-tiles is tiny (≤ 4 with BLOCK_M=32), so
        the redundant trig cost is negligible compared to eliminating the
        K_rot GMEM allocation and extra kernel launch.

        V is loaded as a single full-D (BLOCK_N, D) tile (no split).
        """
        pid         = tl.program_id(0)
        num_q_tiles = tl.cdiv(S, BLOCK_M)
        q_tile      = pid % num_q_tiles
        tmp         = pid // num_q_tiles
        h = tmp % H
        b = tmp // H

        q_start = q_tile * BLOCK_M
        qm      = tl.arange(0, BLOCK_M)
        q_range = q_start + qm
        q_valid = q_range < S

        hd       = tl.arange(0, BLOCK_HD)
        hd_valid = hd < HALF_D
        d_all    = tl.arange(0, BLOCK_D)
        d_valid  = d_all < D

        Q_base = Q   + b * stride_qb + h * stride_qh
        K_base = K   + b * stride_kb + h * stride_kh
        V_base = V   + b * stride_vb + h * stride_vh
        O_base = Out + b * stride_ob + h * stride_oh
        M_base = Mask + b * stride_mb

        # Q: load + RoPE (done once per program, outside the inner loop)
        inv_freq = tl.load(InvFreq + hd, mask=hd_valid, other=0.0).to(tl.float32)

        q_fst = tl.load(
            Q_base + q_range[:, None] * stride_qs + hd[None, :] * stride_qd,
            mask=q_valid[:, None] & hd_valid[None, :], other=0.0,
        ).to(tl.float32)
        q_snd = tl.load(
            Q_base + q_range[:, None] * stride_qs + (hd[None, :] + HALF_D) * stride_qd,
            mask=q_valid[:, None] & hd_valid[None, :], other=0.0,
        ).to(tl.float32)

        q_pos     = q_range[:, None].to(tl.float32) * inv_freq[None, :]
        q_cos     = tl.cos(q_pos)
        q_sin     = tl.sin(q_pos)
        q_rot_fst = q_fst * q_cos - q_snd * q_sin
        q_rot_snd = q_snd * q_cos + q_fst * q_sin

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for k_start in range(0, S, BLOCK_N):
            kn      = tl.arange(0, BLOCK_N)
            k_range = k_start + kn
            k_valid = k_range < S

            # K: load two half-dim strips and apply RoPE inline
            k_fst = tl.load(
                K_base + k_range[:, None] * stride_ks + hd[None, :] * stride_kd,
                mask=k_valid[:, None] & hd_valid[None, :], other=0.0,
            ).to(tl.float32)
            k_snd = tl.load(
                K_base + k_range[:, None] * stride_ks + (hd[None, :] + HALF_D) * stride_kd,
                mask=k_valid[:, None] & hd_valid[None, :], other=0.0,
            ).to(tl.float32)

            k_pos     = k_range[:, None].to(tl.float32) * inv_freq[None, :]
            k_cos     = tl.cos(k_pos)
            k_sin     = tl.sin(k_pos)
            k_rot_fst = k_fst * k_cos - k_snd * k_sin
            k_rot_snd = k_snd * k_cos + k_fst * k_sin

            s = (
                tl.dot(q_rot_fst, tl.trans(k_rot_fst))
                + tl.dot(q_rot_snd, tl.trans(k_rot_snd))
            )

            mask_tile = tl.load(
                M_base + q_range[:, None] * stride_mq + k_range[None, :] * stride_mk,
                mask=q_valid[:, None] & k_valid[None, :], other=0.0,
            ).to(tl.float32)
            s = s + mask_tile
            s = tl.where(k_valid[None, :], s, float("-inf"))
            s = tl.where(q_valid[:, None], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p     = tl.exp(s - m_new[:, None])
            l_i   = l_i * alpha + tl.sum(p, axis=1)
            acc   = acc * alpha[:, None]
            m_i   = m_new

            # V: single full-D load
            v = tl.load(
                V_base + k_range[:, None] * stride_vs + d_all[None, :] * stride_vd,
                mask=k_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot(p, v)

        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / safe_l[:, None]

        tl.store(
            O_base + q_range[:, None] * stride_os + d_all[None, :] * stride_od,
            acc.to(Out.dtype.element_ty),
            mask=q_valid[:, None] & d_valid[None, :],
        )

    # ── Large-S kernels: pre-rotation + autotuned attention ─────────────────

    # Seq-tile width for the K pre-rotation pass.  128 saturates GMEM bandwidth
    # for typical head-dims without over-occupying shared memory.
    _PREROTATE_BLOCK_S = 128

    @triton.jit
    def _prerotate_k(
        K,        # (B, H, S, D) — raw projected keys
        Krot,     # (B, H, S, D) — output: RoPE-rotated keys
        InvFreq,  # (HALF_D,)
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_ob, stride_oh, stride_os, stride_od,
        H, S,
        HALF_D:   tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_S:  tl.constexpr,
    ):
        """
        Grid: (B * H * cdiv(S, BLOCK_S),)

        Rotates one (batch, head, seq-tile) slab of K and writes to Krot.
        This runs once per forward call, keeping K-RoPE cost at O(S) total.
        """
        pid    = tl.program_id(0)
        num_s  = tl.cdiv(S, BLOCK_S)
        s_tile = pid % num_s
        tmp    = pid // num_s
        h = tmp % H
        b = tmp // H

        s_range  = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
        s_valid  = s_range < S
        hd       = tl.arange(0, BLOCK_HD)
        hd_valid = hd < HALF_D

        K_base  = K    + b * stride_kb + h * stride_kh
        Ko_base = Krot + b * stride_ob + h * stride_oh

        inv_freq = tl.load(InvFreq + hd, mask=hd_valid, other=0.0).to(tl.float32)

        k_fst = tl.load(
            K_base + s_range[:, None] * stride_ks + hd[None, :] * stride_kd,
            mask=s_valid[:, None] & hd_valid[None, :], other=0.0,
        ).to(tl.float32)
        k_snd = tl.load(
            K_base + s_range[:, None] * stride_ks + (hd[None, :] + HALF_D) * stride_kd,
            mask=s_valid[:, None] & hd_valid[None, :], other=0.0,
        ).to(tl.float32)

        pos       = s_range[:, None].to(tl.float32) * inv_freq[None, :]
        cos_p     = tl.cos(pos)
        sin_p     = tl.sin(pos)
        k_rot_fst = k_fst * cos_p - k_snd * sin_p
        k_rot_snd = k_snd * cos_p + k_fst * sin_p

        tl.store(
            Ko_base + s_range[:, None] * stride_os + hd[None, :] * stride_od,
            k_rot_fst.to(Krot.dtype.element_ty),
            mask=s_valid[:, None] & hd_valid[None, :],
        )
        tl.store(
            Ko_base + s_range[:, None] * stride_os + (hd[None, :] + HALF_D) * stride_od,
            k_rot_snd.to(Krot.dtype.element_ty),
            mask=s_valid[:, None] & hd_valid[None, :],
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=4, num_stages=1),
        ],
        key=["B", "H", "S", "HALF_D"],
    )
    @triton.jit
    def _fused_rope_attn_fwd(
        Q,        # (B, H, S, D)
        Krot,     # (B, H, S, D) — pre-rotated keys
        V,        # (B, H, S, D)
        Out,      # (B, H, S, D)
        Mask,     # (B, 1, S, S) — additive float mask
        InvFreq,  # (HALF_D,)   — RoPE inv_freq (used only for Q)
        # Q strides
        stride_qb, stride_qh, stride_qs, stride_qd,
        # Krot strides
        stride_kb, stride_kh, stride_ks, stride_kd,
        # V strides
        stride_vb, stride_vh, stride_vs, stride_vd,
        # Out strides
        stride_ob, stride_oh, stride_os, stride_od,
        # Mask strides: (B, 1, S, S) — skip the head-1 stride
        stride_mb, stride_mq, stride_mk,
        # Dimensions
        B, H, S, D,
        HALF_D:   tl.constexpr,
        BLOCK_M:  tl.constexpr,    # autotuned
        BLOCK_N:  tl.constexpr,    # autotuned
        BLOCK_HD: tl.constexpr,    # next_power_of_2(HALF_D)
        BLOCK_D:  tl.constexpr,    # 2 * BLOCK_HD — full head-dim tile width
    ):
        """
        Grid: (B * H * cdiv(S, BLOCK_M),)

        Each program handles one (batch, head, query-tile).
        - Q RoPE is applied once per program, outside the K/V streaming loop.
        - Krot is already RoPE-rotated; no trig inside the inner loop.
        - V is loaded as a full (BLOCK_N, D) tile; output written as (BLOCK_M, D).
        """
        # ── decode program id ─────────────────────────────────────────
        pid         = tl.program_id(0)
        num_q_tiles = tl.cdiv(S, BLOCK_M)
        q_tile      = pid % num_q_tiles
        tmp         = pid // num_q_tiles
        h = tmp % H
        b = tmp // H

        q_start = q_tile * BLOCK_M
        qm      = tl.arange(0, BLOCK_M)
        q_range = q_start + qm
        q_valid = q_range < S

        hd       = tl.arange(0, BLOCK_HD)
        hd_valid = hd < HALF_D
        d_all    = tl.arange(0, BLOCK_D)   # full head-dim indices
        d_valid  = d_all < D

        # ── base pointers ─────────────────────────────────────────────
        Q_base = Q    + b * stride_qb + h * stride_qh
        K_base = Krot + b * stride_kb + h * stride_kh
        V_base = V    + b * stride_vb + h * stride_vh
        O_base = Out  + b * stride_ob + h * stride_oh
        M_base = Mask + b * stride_mb

        # ── Q: load two half-dim strips and apply RoPE once ──────────
        inv_freq = tl.load(InvFreq + hd, mask=hd_valid, other=0.0).to(tl.float32)

        q_fst = tl.load(
            Q_base + q_range[:, None] * stride_qs + hd[None, :] * stride_qd,
            mask=q_valid[:, None] & hd_valid[None, :], other=0.0,
        ).to(tl.float32)
        q_snd = tl.load(
            Q_base + q_range[:, None] * stride_qs + (hd[None, :] + HALF_D) * stride_qd,
            mask=q_valid[:, None] & hd_valid[None, :], other=0.0,
        ).to(tl.float32)

        q_pos     = q_range[:, None].to(tl.float32) * inv_freq[None, :]
        q_cos     = tl.cos(q_pos)
        q_sin     = tl.sin(q_pos)
        q_rot_fst = q_fst * q_cos - q_snd * q_sin   # (BLOCK_M, BLOCK_HD)
        q_rot_snd = q_snd * q_cos + q_fst * q_sin   # (BLOCK_M, BLOCK_HD)

        # ── online softmax accumulators ───────────────────────────────
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # full-D V output

        # ── stream over K/V tiles ─────────────────────────────────────
        for k_start in range(0, S, BLOCK_N):
            kn      = tl.arange(0, BLOCK_N)
            k_range = k_start + kn
            k_valid = k_range < S

            # Krot: two half-dim strips — already rotated, no trig needed
            kr_fst = tl.load(
                K_base + k_range[:, None] * stride_ks + hd[None, :] * stride_kd,
                mask=k_valid[:, None] & hd_valid[None, :], other=0.0,
            ).to(tl.float32)
            kr_snd = tl.load(
                K_base + k_range[:, None] * stride_ks + (hd[None, :] + HALF_D) * stride_kd,
                mask=k_valid[:, None] & hd_valid[None, :], other=0.0,
            ).to(tl.float32)

            # Attention scores: q_rot @ k_rot^T split over two half-dim matmuls
            s = (
                tl.dot(q_rot_fst, tl.trans(kr_fst))
                + tl.dot(q_rot_snd, tl.trans(kr_snd))
            )

            # Additive attention mask
            mask_tile = tl.load(
                M_base + q_range[:, None] * stride_mq + k_range[None, :] * stride_mk,
                mask=q_valid[:, None] & k_valid[None, :], other=0.0,
            ).to(tl.float32)
            s = s + mask_tile
            s = tl.where(k_valid[None, :], s, float("-inf"))
            s = tl.where(q_valid[:, None], s, float("-inf"))

            # Online softmax
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p     = tl.exp(s - m_new[:, None])          # (BLOCK_M, BLOCK_N)
            l_i   = l_i * alpha + tl.sum(p, axis=1)
            acc   = acc * alpha[:, None]
            m_i   = m_new

            # V: single full-D load → one matmul → one accumulation
            v = tl.load(
                V_base + k_range[:, None] * stride_vs + d_all[None, :] * stride_vd,
                mask=k_valid[:, None] & d_valid[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.dot(p, v)

        # ── normalise ─────────────────────────────────────────────────
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / safe_l[:, None]

        # ── store: single full-D write ────────────────────────────────
        tl.store(
            O_base + q_range[:, None] * stride_os + d_all[None, :] * stride_od,
            acc.to(Out.dtype.element_ty),
            mask=q_valid[:, None] & d_valid[None, :],
        )


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------

def triton_fused_rope_attention(
    Q: "torch.Tensor",
    K: "torch.Tensor",
    V: "torch.Tensor",
    mask: "torch.Tensor",
    inv_freq: "torch.Tensor",
) -> "torch.Tensor":
    """Compute fused RoPE + multi-head attention using the Triton kernel.

    Dispatches to one of two paths based on sequence length:

    * **S < _SMALL_S_THRESHOLD** (default 128): ``_fused_rope_attn_small`` —
      single kernel, K RoPE computed inline, no K_rot allocation, fixed
      32×32 tiles.  Optimal when kernel-launch overhead dominates.

    * **S ≥ _SMALL_S_THRESHOLD**: ``_prerotate_k`` + ``_fused_rope_attn_fwd``
      — K rotated once into a scratch buffer (O(S) cost), followed by an
      autotuned attention loop with no trig in the inner K-tile loop.
      Optimal when the O(S²) trig savings outweigh the K_rot roundtrip.

    Args:
        Q, K, V:   ``(B, H, S, D)`` — projected head tensors (pre-RoPE),
                   contiguous, on CUDA.
        mask:      ``(B, 1, S, S)`` — additive float mask (0 for valid,
                   -inf for masked), contiguous.
        inv_freq:  ``(D//2,)``      — RoPE inverse frequencies, same device.

    Returns:
        Output tensor of shape ``(B, H, S, D)``.
    """
    assert _TRITON_AVAILABLE, "Triton is not available"
    B, H, S, D = Q.shape
    assert D % 2 == 0, "head_dim D must be even for RoPE"

    HALF_D   = D // 2
    BLOCK_HD = triton.next_power_of_2(HALF_D)
    BLOCK_D  = 2 * BLOCK_HD
    Out      = torch.empty_like(Q)

    if S < _SMALL_S_THRESHOLD:
        # ── Fully-fused single kernel (small S) ───────────────────────────
        grid = (B * H * triton.cdiv(S, _SMALL_BLOCK_M),)
        _fused_rope_attn_small[grid](
            Q, K, V, Out, mask, inv_freq,
            Q.stride(0),   Q.stride(1),   Q.stride(2),   Q.stride(3),
            K.stride(0),   K.stride(1),   K.stride(2),   K.stride(3),
            V.stride(0),   V.stride(1),   V.stride(2),   V.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            mask.stride(0), mask.stride(2), mask.stride(3),
            H=H, S=S, D=D,
            HALF_D=HALF_D,
            BLOCK_HD=BLOCK_HD,
            BLOCK_D=BLOCK_D,
            BLOCK_M=_SMALL_BLOCK_M,
            BLOCK_N=_SMALL_BLOCK_N,
            num_warps=4,
        )
    else:
        # ── Pre-rotation + autotuned attention (large S) ──────────────────
        K_rot    = torch.empty_like(K)
        pre_grid = (B * H * triton.cdiv(S, _PREROTATE_BLOCK_S),)
        _prerotate_k[pre_grid](
            K, K_rot, inv_freq,
            K.stride(0),     K.stride(1),     K.stride(2),     K.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            H=H, S=S,
            HALF_D=HALF_D, BLOCK_HD=BLOCK_HD,
            BLOCK_S=_PREROTATE_BLOCK_S,
            num_warps=4,
        )
        grid = lambda meta: (B * H * triton.cdiv(S, meta["BLOCK_M"]),)
        _fused_rope_attn_fwd[grid](
            Q, K_rot, V, Out, mask, inv_freq,
            Q.stride(0),     Q.stride(1),     Q.stride(2),     Q.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            V.stride(0),     V.stride(1),     V.stride(2),     V.stride(3),
            Out.stride(0),   Out.stride(1),   Out.stride(2),   Out.stride(3),
            mask.stride(0),  mask.stride(2),  mask.stride(3),
            B=B, H=H, S=S, D=D,
            HALF_D=HALF_D,
            BLOCK_HD=BLOCK_HD,
            BLOCK_D=BLOCK_D,
        )

    return Out

