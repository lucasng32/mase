"""
Per-group specialized Triton kernels where G_SIZE is tl.constexpr.

For large groups (size >= 16), groups are sorted by exact size and dispatched
in runs of same-size groups. Each unique G_SIZE value triggers a separate
Triton JIT compilation, baking the group size into the binary. This allows
the compiler to:

  1. Eliminate cu_seqlens HBM loads — g_start = g * G_SIZE, pure arithmetic
  2. Unroll the K/V streaming loop when G_SIZE is small (fully unrolled when
     G_SIZE <= BLOCK_N, i.e. single-iteration groups)
  3. Eliminate validity masking for exact-fit tiles when G_SIZE % BLOCK_M == 0
     (no masked stores or loads in the common case)

Small groups (size < 16) are handled by the existing stitched tile kernel,
which remains unchanged — bin-packing is still optimal there.

Dispatch model
--------------
  Regime 1  small groups (< 16)  →  stitched 16-lane tiles (FFD bin-packing)
  Regime 2  large groups (>= 16) →  sorted by exact size, same-size groups
                                     batched into a single _uniform_groups_attn_fwd
                                     call per unique G_SIZE value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .triton_grouped_attn import is_triton_available
from .triton_stitched_attn import (
    StitchedTileData,
    _SMALL_GROUP_THRESHOLD,
    _ffd_pack_small_groups,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpecializedGroupRun:
    """A contiguous run of same-size groups in the globally-sorted batch."""

    abs_start: int   # first position in the globally-sorted batch tensor
    n_groups: int    # number of groups in this run (all have the same size)
    g_size: int      # exact size of every group
    block_m: int     # BLOCK_M = max(min(g_size, 32), 16)
    tiles: int       # TILES_PER_GROUP = ceil(g_size / block_m)


@dataclass
class SpecializedPartition:
    """Partition for per-G_SIZE specialized kernel dispatch.

    Created once at MASE pass application time, never in a forward pass.
    """

    stitched: Optional[StitchedTileData]
    """Small-group bin-packed tile metadata (size < 16), or None."""

    large_runs: list[SpecializedGroupRun]
    """One entry per run of same-size large groups, sorted ascending by g_size."""

    global_sort_perm: torch.Tensor    # (B,) long
    global_unsort_perm: torch.Tensor  # (B,) long

    @classmethod
    def from_group_ids(cls, group_ids: torch.Tensor) -> SpecializedPartition:
        ids = group_ids.cpu()
        unique_ids = ids.unique()
        group_members = [(ids == g).nonzero(as_tuple=True)[0] for g in unique_ids]
        group_sizes = [m.numel() for m in group_members]

        small_idxs = [i for i, s in enumerate(group_sizes) if s < _SMALL_GROUP_THRESHOLD]
        large_idxs = [i for i, s in enumerate(group_sizes) if s >= _SMALL_GROUP_THRESHOLD]

        # Small groups: FFD bin-packing into 16-lane tiles (same as stitched)
        stitched, small_parts = _ffd_pack_small_groups(
            [group_members[i] for i in small_idxs],
            [group_sizes[i] for i in small_idxs],
        )

        # Large groups: sort by EXACT size so same-size groups are contiguous.
        # This enables same-G_SIZE groups to share a single kernel launch while
        # still getting a unique compiled binary per unique g_size.
        sorted_large = sorted(large_idxs, key=lambda i: group_sizes[i])
        large_parts = [group_members[i] for i in sorted_large]

        all_parts = small_parts + large_parts
        global_sort_perm = (
            torch.cat(all_parts).long()
            if all_parts
            else torch.empty(0, dtype=torch.long)
        )

        B = len(ids)
        global_unsort_perm = torch.empty(B, dtype=torch.long)
        global_unsort_perm[global_sort_perm] = torch.arange(B, dtype=torch.long)

        # Build runs of consecutive same-size large groups
        stitched_total = stitched.total_seqs if stitched is not None else 0
        large_sizes = [group_sizes[gi] for gi in sorted_large]
        large_runs: list[SpecializedGroupRun] = []
        abs_pos = stitched_total
        i = 0
        while i < len(sorted_large):
            g_size = large_sizes[i]
            run_abs_start = abs_pos
            n = 0
            while i + n < len(sorted_large) and large_sizes[i + n] == g_size:
                abs_pos += g_size
                n += 1
            block_m = max(min(g_size, 32), 16)
            tiles = (g_size + block_m - 1) // block_m
            large_runs.append(SpecializedGroupRun(
                abs_start=run_abs_start,
                n_groups=n,
                g_size=g_size,
                block_m=block_m,
                tiles=tiles,
            ))
            i += n

        return cls(
            stitched=stitched,
            large_runs=large_runs,
            global_sort_perm=global_sort_perm,
            global_unsort_perm=global_unsort_perm,
        )


# ---------------------------------------------------------------------------
# Specialized kernel: uniform group size
# ---------------------------------------------------------------------------
if is_triton_available():
    import triton
    import triton.language as tl
    from .triton_stitched_attn import _stitched_grouped_attn_fwd  # reused for small groups

    @triton.jit
    def _uniform_groups_attn_fwd(
        Q,
        K,
        V,
        Out,
        scale,
        stride_qt, stride_qh, stride_qb, stride_qd,
        stride_kt, stride_kh, stride_kb, stride_kd,
        stride_vt, stride_vh, stride_vb, stride_vd,
        stride_ot, stride_oh, stride_ob, stride_od,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        G: tl.constexpr,       # number of groups (all identical size)
        G_SIZE: tl.constexpr,  # exact size of every group — baked into binary
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        TILES: tl.constexpr,   # = ceil(G_SIZE / BLOCK_M)
    ):
        """One program per (timestep, head, group, query-tile).

        All groups have identical size G_SIZE.  Group start offsets are
        computed as ``g * G_SIZE`` — a compile-time multiply, with no
        cu_seqlens pointer and no HBM metadata loads.

        Grid: ``(T * H * G * TILES,)``
        Q, K, V layout: ``(T, H, G * G_SIZE, D)``
        """
        pid = tl.program_id(0)
        q_tile = pid % TILES
        tmp = pid // TILES
        g = tmp % G
        tmp = tmp // G
        h = tmp % H
        t = tmp // H

        # Group start: pure compile-time arithmetic — no HBM load
        g_start = g * G_SIZE
        q_begin = q_tile * BLOCK_M

        if q_begin >= G_SIZE:
            return  # out-of-bounds tile — compile-time dead for most configs

        q_offs = g_start + q_begin + tl.arange(0, BLOCK_M)
        # G_SIZE is constexpr → Triton can evaluate this mask at compile time
        q_valid = tl.arange(0, BLOCK_M) < (G_SIZE - q_begin)
        d_range = tl.arange(0, BLOCK_D)
        d_valid = d_range < D

        q_ptrs = (
            Q
            + t * stride_qt
            + h * stride_qh
            + q_offs[:, None] * stride_qb
            + d_range[None, :] * stride_qd
        )
        q_data = tl.load(
            q_ptrs, mask=q_valid[:, None] & d_valid[None, :], other=0.0
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # K/V streaming loop — G_SIZE constexpr → iteration count known at compile
        # time; Triton fully unrolls when G_SIZE <= BLOCK_N (single-tile groups)
        for k_begin in range(0, G_SIZE, BLOCK_N):
            k_offs = g_start + k_begin + tl.arange(0, BLOCK_N)
            k_valid = tl.arange(0, BLOCK_N) < (G_SIZE - k_begin)  # constexpr!

            k_ptrs = (
                K
                + t * stride_kt
                + h * stride_kh
                + k_offs[:, None] * stride_kb
                + d_range[None, :] * stride_kd
            )
            k_data = tl.load(
                k_ptrs, mask=k_valid[:, None] & d_valid[None, :], other=0.0
            ).to(tl.float32)

            s = tl.dot(q_data, tl.trans(k_data)) * scale
            s = tl.where(k_valid[None, :], s, float("-inf"))
            s = tl.where(q_valid[:, None], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_ptrs = (
                V
                + t * stride_vt
                + h * stride_vh
                + k_offs[:, None] * stride_vb
                + d_range[None, :] * stride_vd
            )
            v_data = tl.load(
                v_ptrs, mask=k_valid[:, None] & d_valid[None, :], other=0.0
            ).to(tl.float32)
            acc += tl.dot(p.to(tl.float32), v_data)
            m_i = m_new

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
# Dispatch
# ---------------------------------------------------------------------------

def triton_specialized_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    partition: SpecializedPartition,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute grouped attention with G_SIZE-specialized kernel dispatch.

    Args:
        Q, K, V: ``(T, H, B_sorted, D)`` sorted by ``partition.global_sort_perm``.
        partition: ``SpecializedPartition`` (stitched prefix + large-group runs).
        scale: attention score scale (1.0 for Chronos2).
    """
    import triton

    T, H, B, D = Q.shape
    Out = torch.empty_like(Q)
    BLOCK_D = triton.next_power_of_2(D)

    # Regime 1: stitched small-group tiles (identical to TRITON_STITCHED path)
    if partition.stitched is not None:
        st = partition.stitched
        total = st.total_seqs
        Q_s = Q[:, :, :total, :]
        K_s = K[:, :, :total, :]
        V_s = V[:, :, :total, :]
        Out_s = Out[:, :, :total, :]

        grid = (T * H * st.num_tiles,)
        _stitched_grouped_attn_fwd[grid](
            Q_s, K_s, V_s, Out_s,
            st.tile_seq_starts.to(Q.device),
            st.tile_seq_counts.to(Q.device),
            st.tile_group_labels.to(Q.device),
            scale,
            Q_s.stride(0), Q_s.stride(1), Q_s.stride(2), Q_s.stride(3),
            K_s.stride(0), K_s.stride(1), K_s.stride(2), K_s.stride(3),
            V_s.stride(0), V_s.stride(1), V_s.stride(2), V_s.stride(3),
            Out_s.stride(0), Out_s.stride(1), Out_s.stride(2), Out_s.stride(3),
            H=H, D=D, NUM_TILES=st.num_tiles, BLOCK_D=BLOCK_D,
        )

    # Regime 2: one specialized launch per run of same-size large groups
    for run in partition.large_runs:
        total = run.n_groups * run.g_size
        s, e = run.abs_start, run.abs_start + total
        Q_r = Q[:, :, s:e, :]
        K_r = K[:, :, s:e, :]
        V_r = V[:, :, s:e, :]
        Out_r = Out[:, :, s:e, :]

        grid = (T * H * run.n_groups * run.tiles,)
        _uniform_groups_attn_fwd[grid](
            Q_r, K_r, V_r, Out_r, scale,
            Q_r.stride(0), Q_r.stride(1), Q_r.stride(2), Q_r.stride(3),
            K_r.stride(0), K_r.stride(1), K_r.stride(2), K_r.stride(3),
            V_r.stride(0), V_r.stride(1), V_r.stride(2), V_r.stride(3),
            Out_r.stride(0), Out_r.stride(1), Out_r.stride(2), Out_r.stride(3),
            H=H, D=D, BLOCK_D=BLOCK_D,
            G=run.n_groups,
            G_SIZE=run.g_size,   # constexpr: unique binary per unique g_size
            BLOCK_M=run.block_m,
            BLOCK_N=run.block_m,
            TILES=run.tiles,
        )

    return Out
