"""
Triton kernel for batch-axis grouped attention (tiled).

Computes multi-head self-attention within pre-defined groups along the batch
dimension — the pattern used by Chronos2's GroupSelfAttention.  The batch is
pre-sorted so that each group occupies a contiguous slice, described by
cumulative offsets (``cu_seqlens``).

The kernel uses **tiled parallelism**:

* One program per ``(timestep, head, group, query-tile)``.
* Each program loads a ``BLOCK_M × D`` tile of queries and streams over
  ``BLOCK_N``-sized tiles of keys/values, accumulating the result with an
  **online softmax** (single pass over K/V — no double-read).
* ``tl.dot`` maps the tile matmuls onto the GPU's tensor cores.

This replaces the earlier naive kernel which processed one query at a time
inside a single program per group.

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


# ---------------------------------------------------------------------------
# Tiled kernel
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.jit
    def _tiled_grouped_attn_fwd(
        Q,
        K,
        V,
        Out,
        cu_seqlens,
        scale,
        # Q strides
        stride_qt,
        stride_qh,
        stride_qb,
        stride_qd,
        # K strides
        stride_kt,
        stride_kh,
        stride_kb,
        stride_kd,
        # V strides
        stride_vt,
        stride_vh,
        stride_vb,
        stride_vd,
        # Out strides
        stride_ot,
        stride_oh,
        stride_ob,
        stride_od,
        # Compile-time constants
        H: tl.constexpr,
        G: tl.constexpr,
        D: tl.constexpr,
        TILES_PER_GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        One program per (timestep, head, group, query-tile).

        Grid: ``(T * H * G * TILES_PER_GROUP,)``

        Q, K, V layout: ``(T, H, B_sorted, D)``
        cu_seqlens:      ``(G + 1,)`` int32 cumulative group sizes.
        """
        # ── decode program id ─────────────────────────────────────────
        pid = tl.program_id(0)
        q_tile = pid % TILES_PER_GROUP
        tmp = pid // TILES_PER_GROUP
        g = tmp % G
        tmp = tmp // G
        h = tmp % H
        t = tmp // H

        # ── group bounds ──────────────────────────────────────────────
        g_start = tl.load(cu_seqlens + g)
        g_end = tl.load(cu_seqlens + g + 1)
        g_size = g_end - g_start

        # This tile's first query offset within the group
        q_begin = q_tile * BLOCK_M
        if q_begin >= g_size:
            return  # padded tile — nothing to do

        # ── load Q tile: (BLOCK_M, BLOCK_D) ──────────────────────────
        q_offs = g_start + q_begin + tl.arange(0, BLOCK_M)
        q_valid = tl.arange(0, BLOCK_M) < (g_size - q_begin)
        d_range = tl.arange(0, BLOCK_D)
        d_valid = d_range < D

        q_ptrs = (
            Q
            + t * stride_qt
            + h * stride_qh
            + q_offs[:, None] * stride_qb
            + d_range[None, :] * stride_qd
        )
        q_tile_data = tl.load(
            q_ptrs, mask=q_valid[:, None] & d_valid[None, :], other=0.0
        ).to(tl.float32)

        # ── online softmax accumulators (per query row) ───────────────
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # ── stream over K/V tiles ─────────────────────────────────────
        for k_begin in range(0, g_size, BLOCK_N):
            k_offs = g_start + k_begin + tl.arange(0, BLOCK_N)
            k_valid = tl.arange(0, BLOCK_N) < (g_size - k_begin)

            # K tile: (BLOCK_N, BLOCK_D)
            k_ptrs = (
                K
                + t * stride_kt
                + h * stride_kh
                + k_offs[:, None] * stride_kb
                + d_range[None, :] * stride_kd
            )
            k_tile_data = tl.load(
                k_ptrs, mask=k_valid[:, None] & d_valid[None, :], other=0.0
            ).to(tl.float32)

            # S = Q @ K^T : (BLOCK_M, BLOCK_N)
            s = tl.dot(q_tile_data, tl.trans(k_tile_data)) * scale
            # Mask invalid keys (padding beyond group boundary)
            s = tl.where(k_valid[None, :], s, float("-inf"))
            # Mask invalid queries so they don't affect m_new
            s = tl.where(q_valid[:, None], s, float("-inf"))

            # ── online softmax update ─────────────────────────────────
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # V tile: (BLOCK_N, BLOCK_D)
            v_ptrs = (
                V
                + t * stride_vt
                + h * stride_vh
                + k_offs[:, None] * stride_vb
                + d_range[None, :] * stride_vd
            )
            v_tile_data = tl.load(
                v_ptrs, mask=k_valid[:, None] & d_valid[None, :], other=0.0
            ).to(tl.float32)

            acc += tl.dot(p.to(tl.float32), v_tile_data)
            m_i = m_new

        # ── normalise and store ───────────────────────────────────────
        # Guard against l_i == 0 for fully-masked query rows
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / safe_l[:, None]

        o_ptrs = (
            Out
            + t * stride_ot
            + h * stride_oh
            + q_offs[:, None] * stride_ob
            + d_range[None, :] * stride_od
        )
        tl.store(
            o_ptrs,
            acc.to(Out.dtype.element_ty),
            mask=q_valid[:, None] & d_valid[None, :],
        )


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------
BLOCK_M_DEFAULT = 32
BLOCK_N_DEFAULT = 32


def triton_grouped_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_groups: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute grouped attention using the tiled Triton kernel.

    Args:
        Q, K, V: ``(T, H, B_sorted, D)`` — projected and head-split tensors
            with the batch dimension sorted by group.
        cu_seqlens: ``(G + 1,)`` int32 tensor of cumulative group sizes.
        num_groups: number of groups ``G``.
        scale: attention score scale (1.0 for Chronos2).

    Returns:
        Output tensor of the same shape as Q.
    """
    assert _TRITON_AVAILABLE, "Triton is not available"
    T, H, B, D = Q.shape

    Out = torch.empty_like(Q)

    BLOCK_D = triton.next_power_of_2(D)

    # Max group size → how many query tiles per group
    max_group_size = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    tiles_per_group = triton.cdiv(max_group_size, BLOCK_M_DEFAULT)

    grid = (T * H * num_groups * tiles_per_group,)

    _tiled_grouped_attn_fwd[grid](
        Q,
        K,
        V,
        Out,
        cu_seqlens,
        scale,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        H=H,
        G=num_groups,
        D=D,
        TILES_PER_GROUP=tiles_per_group,
        BLOCK_M=BLOCK_M_DEFAULT,
        BLOCK_N=BLOCK_N_DEFAULT,
        BLOCK_D=BLOCK_D,
    )
    return Out
