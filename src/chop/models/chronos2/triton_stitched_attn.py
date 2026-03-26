"""
Stitched-tile Triton kernel dispatch for grouped attention (small groups).

For groups with size < 16, bin-packing multiple groups into a single 16-lane
tile avoids the 87.5% wasted tensor-core compute that occurs when a 2-element
group runs inside a 16-lane tile alone.

Two regimes, split at ``_SMALL_GROUP_THRESHOLD = 16``:

**Regime 1 — small groups (size < 16): stitched tiles**
  First Fit (ascending size) bin-packs small groups into 16-lane tiles.  One
  Triton program per (timestep, head, tile) computes attention for all
  sub-groups in the tile using a block-diagonal mask.  Programs are reduced by
  up to 7× vs the old bucketed dispatch for the same batch.

**Regime 2 — large groups (size >= 16): bucketed kernel (unchanged)**
  Inherited from ``triton_bucketed_attn``.  One Triton kernel launch per
  occupied power-of-2 bucket.

The stitched prefix occupies the first positions in the globally-sorted batch
tensor; large-group buckets follow.  A single global unsort permutation
recombines both outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .triton_grouped_attn import is_triton_available

_SMALL_GROUP_THRESHOLD = 16

_LARGE_BUCKET_THRESHOLDS = [16, 32, 64, 128]


def _bucket_for_large(size: int) -> int:
    for b in _LARGE_BUCKET_THRESHOLDS:
        if size <= b:
            return b
    p = 128
    while p < size:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Stitched-tile kernel
# ---------------------------------------------------------------------------
if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _stitched_grouped_attn_fwd(
        Q,
        K,
        V,
        Out,
        tile_seq_starts,    # (num_tiles,) int32
        tile_seq_counts,    # (num_tiles,) int32
        tile_group_labels,  # (num_tiles * 16,) int32
        scale,
        stride_qt, stride_qh, stride_qb, stride_qd,
        stride_kt, stride_kh, stride_kb, stride_kd,
        stride_vt, stride_vh, stride_vb, stride_vd,
        stride_ot, stride_oh, stride_ob, stride_od,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_TILES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per (timestep, head, tile).  Tile is always 16 lanes wide.

        Grid: ``(T * H * NUM_TILES,)``
        """
        pid = tl.program_id(0)
        tile_id = pid % NUM_TILES
        tmp = pid // NUM_TILES
        h = tmp % H
        t = tmp // H

        seq_start = tl.load(tile_seq_starts + tile_id)
        seq_count = tl.load(tile_seq_counts + tile_id)

        lane = tl.arange(0, 16)
        valid = lane < seq_count
        d_range = tl.arange(0, BLOCK_D)
        d_valid = d_range < D

        q_label = tl.load(
            tile_group_labels + tile_id * 16 + lane, mask=valid, other=-1
        )
        q_offs = seq_start + lane

        q_ptrs = (
            Q + t * stride_qt + h * stride_qh
            + q_offs[:, None] * stride_qb + d_range[None, :] * stride_qd
        )
        q_data = tl.load(
            q_ptrs, mask=valid[:, None] & d_valid[None, :], other=0.0
        ).to(tl.float32)

        k_ptrs = (
            K + t * stride_kt + h * stride_kh
            + q_offs[:, None] * stride_kb + d_range[None, :] * stride_kd
        )
        k_data = tl.load(
            k_ptrs, mask=valid[:, None] & d_valid[None, :], other=0.0
        ).to(tl.float32)

        v_ptrs = (
            V + t * stride_vt + h * stride_vh
            + q_offs[:, None] * stride_vb + d_range[None, :] * stride_vd
        )
        v_data = tl.load(
            v_ptrs, mask=valid[:, None] & d_valid[None, :], other=0.0
        ).to(tl.float32)

        # S = Q @ K^T : (16, 16)
        s = tl.dot(q_data, tl.trans(k_data)) * scale

        # Block-diagonal mask: isolate groups within the tile
        cross_group = q_label[:, None] != q_label[None, :]
        s = tl.where(cross_group, float("-inf"), s)
        s = tl.where(valid[None, :], s, float("-inf"))
        s = tl.where(valid[:, None], s, float("-inf"))

        # Softmax — single step: all K fits in one 16-lane tile
        m_i = tl.max(s, axis=1)
        p = tl.exp(s - m_i[:, None])
        p = tl.where(valid[:, None], p, 0.0)
        l_i = tl.sum(p, axis=1)
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        acc = tl.dot(p, v_data) / safe_l[:, None]

        o_ptrs = (
            Out + t * stride_ot + h * stride_oh
            + q_offs[:, None] * stride_ob + d_range[None, :] * stride_od
        )
        tl.store(
            o_ptrs,
            acc.to(Out.dtype.element_ty),
            mask=valid[:, None] & d_valid[None, :],
        )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StitchedTileData:
    """Metadata for the stitched-tile regime (groups with size < 16)."""

    num_tiles: int
    total_seqs: int
    tile_seq_starts: torch.Tensor   # (num_tiles,) int32
    tile_seq_counts: torch.Tensor   # (num_tiles,) int32
    tile_group_labels: torch.Tensor # (num_tiles * 16,) int32; -1 for padding


@dataclass
class BucketData:
    """Metadata for one large-group bucket (size >= 16)."""

    bucket_size: int
    num_groups: int
    cu_seqlens: torch.Tensor  # (num_groups + 1,) int32
    batch_offset: int


@dataclass
class StitchedPartition:
    """Pre-computed stitched + bucketed partition.

    Created once at MASE pass application time, never in a forward pass.
    """

    stitched: Optional[StitchedTileData]
    """Small-group (size < 16) stitched tile metadata, or None."""

    buckets: Dict[int, BucketData]
    """Large-group (size >= 16) per-bucket metadata."""

    global_sort_perm: torch.Tensor   # (B,) long
    global_unsort_perm: torch.Tensor # (B,) long

    @classmethod
    def from_group_ids(cls, group_ids: torch.Tensor) -> StitchedPartition:
        ids = group_ids.cpu()
        unique_ids = ids.unique()

        group_members = [(ids == g).nonzero(as_tuple=True)[0] for g in unique_ids]
        group_sizes = [m.numel() for m in group_members]

        small_idxs = [i for i, s in enumerate(group_sizes) if s < _SMALL_GROUP_THRESHOLD]
        large_idxs = [i for i, s in enumerate(group_sizes) if s >= _SMALL_GROUP_THRESHOLD]

        stitched, small_parts = _ffd_pack_small_groups(
            [group_members[i] for i in small_idxs],
            [group_sizes[i] for i in small_idxs],
        )

        large_buckets = [_bucket_for_large(group_sizes[i]) for i in large_idxs]
        sorted_large = sorted(range(len(large_idxs)), key=lambda j: large_buckets[j])
        large_parts = [group_members[large_idxs[j]] for j in sorted_large]

        all_parts = small_parts + large_parts
        global_sort_perm = torch.cat(all_parts).long() if all_parts else torch.empty(0, dtype=torch.long)

        B = len(ids)
        global_unsort_perm = torch.empty(B, dtype=torch.long)
        global_unsort_perm[global_sort_perm] = torch.arange(B, dtype=torch.long)

        stitched_total = stitched.total_seqs if stitched is not None else 0
        bucket_group_map: dict[int, list[int]] = {}
        for j in sorted_large:
            bucket_group_map.setdefault(large_buckets[j], []).append(large_idxs[j])

        buckets: dict[int, BucketData] = {}
        batch_offset = stitched_total
        for bsize in sorted(bucket_group_map.keys()):
            gis = bucket_group_map[bsize]
            sizes = [group_sizes[gi] for gi in gis]
            cu: list[int] = [0]
            for s in sizes:
                cu.append(cu[-1] + s)
            buckets[bsize] = BucketData(
                bucket_size=bsize,
                num_groups=len(gis),
                cu_seqlens=torch.tensor(cu, dtype=torch.int32),
                batch_offset=batch_offset,
            )
            batch_offset += cu[-1]

        return cls(
            stitched=stitched,
            buckets=buckets,
            global_sort_perm=global_sort_perm,
            global_unsort_perm=global_unsort_perm,
        )


# ---------------------------------------------------------------------------
# FFD bin-packing
# ---------------------------------------------------------------------------

def _ffd_pack_small_groups(
    group_members: list[torch.Tensor],
    group_sizes: list[int],
    tile_size: int = 16,
) -> tuple[Optional[StitchedTileData], list[torch.Tensor]]:
    """First Fit (ascending size) bin-packing into ``tile_size``-lane tiles."""
    if not group_members:
        return None, []

    sorted_order = sorted(range(len(group_members)), key=lambda i: group_sizes[i])

    tiles: list[list[tuple[torch.Tensor, int]]] = []
    tile_used: list[int] = []

    for i in sorted_order:
        members = group_members[i]
        size = group_sizes[i]
        placed = False
        for ti in range(len(tiles)):
            if tile_used[ti] + size <= tile_size:
                tiles[ti].append((members, len(tiles[ti])))
                tile_used[ti] += size
                placed = True
                break
        if not placed:
            tiles.append([(members, 0)])
            tile_used.append(size)

    num_tiles = len(tiles)
    seq_starts: list[int] = []
    seq_counts: list[int] = []
    labels_flat: list[int] = []
    sort_parts: list[torch.Tensor] = []

    batch_offset = 0
    for tile_groups in tiles:
        total = sum(m.numel() for m, _ in tile_groups)
        seq_starts.append(batch_offset)
        seq_counts.append(total)

        labels = [-1] * tile_size
        lane = 0
        for members, label in tile_groups:
            for _ in range(members.numel()):
                labels[lane] = label
                lane += 1
        labels_flat.extend(labels)

        for members, _ in tile_groups:
            sort_parts.append(members)
        batch_offset += total

    return StitchedTileData(
        num_tiles=num_tiles,
        total_seqs=batch_offset,
        tile_seq_starts=torch.tensor(seq_starts, dtype=torch.int32),
        tile_seq_counts=torch.tensor(seq_counts, dtype=torch.int32),
        tile_group_labels=torch.tensor(labels_flat, dtype=torch.int32),
    ), sort_parts


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def triton_stitched_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    partition: StitchedPartition,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute grouped attention: stitched kernel for small groups, bucketed for large.

    Args:
        Q, K, V: ``(T, H, B_sorted, D)`` sorted by ``partition.global_sort_perm``.
        partition: ``StitchedPartition`` (stitched prefix first, large buckets after).
        scale: attention score scale (1.0 for Chronos2).
    """
    import triton
    from .triton_grouped_attn import _tiled_grouped_attn_fwd

    T, H, B, D = Q.shape
    Out = torch.empty_like(Q)
    BLOCK_D = triton.next_power_of_2(D)

    # Regime 1: stitched small-group tiles
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

    # Regime 2: per-bucket launches for large groups
    for bucket_size, bucket in partition.buckets.items():
        block_m = max(min(bucket_size, 32), 16)
        block_n = block_m
        tiles_per_group = (bucket_size + block_m - 1) // block_m

        offset = bucket.batch_offset
        total = int(bucket.cu_seqlens[-1].item())
        Q_b = Q[:, :, offset:offset + total, :]
        K_b = K[:, :, offset:offset + total, :]
        V_b = V[:, :, offset:offset + total, :]
        Out_b = Out[:, :, offset:offset + total, :]

        cu = bucket.cu_seqlens.to(Q.device)
        grid = (T * H * bucket.num_groups * tiles_per_group,)
        _tiled_grouped_attn_fwd[grid](
            Q_b, K_b, V_b, Out_b, cu, scale,
            Q_b.stride(0), Q_b.stride(1), Q_b.stride(2), Q_b.stride(3),
            K_b.stride(0), K_b.stride(1), K_b.stride(2), K_b.stride(3),
            V_b.stride(0), V_b.stride(1), V_b.stride(2), V_b.stride(3),
            Out_b.stride(0), Out_b.stride(1), Out_b.stride(2), Out_b.stride(3),
            H=H, G=bucket.num_groups, D=D,
            TILES_PER_GROUP=tiles_per_group,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=BLOCK_D,
        )

    return Out
