"""
Per-bucket Triton kernel dispatch for grouped attention.

``BucketedPartition`` sorts groups into power-of-2 size buckets and launches
one Triton kernel call per occupied bucket.  Each bucket uses ``BLOCK_M`` and
``BLOCK_N`` equal to its bucket size (capped at 32), so small groups avoid
running oversized tile matmuls against the global maximum group size.

This is the backend for ``KernelVariant.TRITON_BUCKETED`` in
``GroupAwareMHA``.  It reuses the same ``_tiled_grouped_attn_fwd`` kernel
from ``triton_grouped_attn`` — Triton's JIT naturally compiles a separate
binary per unique ``(BLOCK_M, BLOCK_N, TILES_PER_GROUP)`` combination.

Bucket thresholds (power-of-2 tile sizes):

    size 1       → bucket  2  (1 wasted slot; handled via masking)
    size 2       → bucket  2  (BLOCK_M = 2,  TILES_PER_GROUP = 1)
    size 3–4     → bucket  4  (BLOCK_M = 4,  TILES_PER_GROUP = 1)
    size 5–8     → bucket  8  (BLOCK_M = 8,  TILES_PER_GROUP = 1)
    size 9–16    → bucket 16  (BLOCK_M = 16, TILES_PER_GROUP = 1)
    size 17–32   → bucket 32  (BLOCK_M = 32, TILES_PER_GROUP = 1)
    size 33–64   → bucket 64  (BLOCK_M = 32, TILES_PER_GROUP = 2)
    size > 64    → bucket = next_power_of_2(size), TILES_PER_GROUP = ceil/32
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .triton_grouped_attn import is_triton_available

# Power-of-2 tile-size boundaries.  Groups with actual size ≤ threshold go
# into that bucket; BLOCK_M = min(threshold, 32).
_BUCKET_THRESHOLDS = [2, 4, 8, 16, 32, 64]


def _bucket_for_size(size: int) -> int:
    """Return the smallest bucket threshold >= *size*."""
    for b in _BUCKET_THRESHOLDS:
        if size <= b:
            return b
    # size > 64: next power of 2
    p = 128
    while p < size:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BucketData:
    """Metadata for one group-size bucket."""

    bucket_size: int
    """Tile size for this bucket (BLOCK_M = BLOCK_N = min(bucket_size, 32))."""

    num_groups: int
    """Number of groups in this bucket."""

    cu_seqlens: torch.Tensor
    """``(num_groups + 1,)`` int32 — cumulative actual group sizes, relative to
    the start of this bucket's slice in the globally-sorted batch."""

    batch_offset: int
    """First position in the globally-sorted batch tensor belonging to this
    bucket."""


@dataclass
class BucketedPartition:
    """Pre-computed bucketed group partition derived from a ``group_ids`` tensor.

    Created once at MASE pass application time, never in a forward pass.
    """

    buckets: Dict[int, BucketData]
    """Per-bucket metadata, keyed by ``bucket_size``, sorted ascending."""

    global_sort_perm: torch.Tensor
    """``(B,)`` long — reorders the batch so that groups are contiguous and
    groups of the same bucket are adjacent."""

    global_unsort_perm: torch.Tensor
    """``(B,)`` long — inverse of ``global_sort_perm``."""

    @classmethod
    def from_group_ids(cls, group_ids: torch.Tensor) -> BucketedPartition:
        """Build a ``BucketedPartition`` from a ``group_ids`` tensor.

        Args:
            group_ids: ``(B,)`` long tensor of group labels.

        Returns:
            ``BucketedPartition`` with ``buckets`` sorted ascending by
            ``bucket_size``.
        """
        ids = group_ids.cpu()
        unique_ids = ids.unique()

        # Members (batch indices) and sizes for each unique group label
        group_members = [(ids == g).nonzero(as_tuple=True)[0] for g in unique_ids]
        group_sizes = [g.numel() for g in group_members]
        group_buckets = [_bucket_for_size(s) for s in group_sizes]

        # Stable sort of groups by bucket size
        sorted_order = sorted(range(len(group_members)), key=lambda i: group_buckets[i])

        # Global sort permutation: concatenate batch indices in bucket order
        global_sort_perm = torch.cat([group_members[gi] for gi in sorted_order]).long()

        # Inverse permutation
        B = len(ids)
        global_unsort_perm = torch.empty(B, dtype=torch.long)
        global_unsort_perm[global_sort_perm] = torch.arange(B, dtype=torch.long)

        # Build per-bucket BucketData
        bucket_group_map: dict[int, list[int]] = {}  # bucket_size -> [group indices]
        for gi in sorted_order:
            bucket_group_map.setdefault(group_buckets[gi], []).append(gi)

        buckets: dict[int, BucketData] = {}
        batch_offset = 0
        for bsize in sorted(bucket_group_map.keys()):
            member_gi_list = bucket_group_map[bsize]
            num_groups = len(member_gi_list)

            # Actual group sizes (not padded to bucket_size)
            actual_sizes = [group_sizes[gi] for gi in member_gi_list]
            cu: list[int] = [0]
            for s in actual_sizes:
                cu.append(cu[-1] + s)

            buckets[bsize] = BucketData(
                bucket_size=bsize,
                num_groups=num_groups,
                cu_seqlens=torch.tensor(cu, dtype=torch.int32),
                batch_offset=batch_offset,
            )
            batch_offset += cu[-1]

        return cls(
            buckets=buckets,
            global_sort_perm=global_sort_perm,
            global_unsort_perm=global_unsort_perm,
        )


# ---------------------------------------------------------------------------
# Multi-bucket kernel dispatch
# ---------------------------------------------------------------------------

def triton_bucketed_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    partition: BucketedPartition,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute grouped attention using one Triton kernel launch per bucket.

    Args:
        Q, K, V: ``(T, H, B_sorted, D)`` — projected, head-split tensors
            with the batch dimension sorted in ``partition.global_sort_perm``
            order.
        partition: ``BucketedPartition`` with metadata on the same device as
            ``Q``.
        scale: attention score scale (1.0 for Chronos2).

    Returns:
        Output tensor of the same shape as ``Q``.
    """
    assert is_triton_available(), "Triton is not available"

    import triton

    from .triton_grouped_attn import _tiled_grouped_attn_fwd

    T, H, B, D = Q.shape
    Out = torch.empty_like(Q)
    BLOCK_D = triton.next_power_of_2(D)

    for bucket_size, bucket in partition.buckets.items():
        # tl.dot requires M, N, K >= 16 (tensor-core minimum).
        # For small buckets (size < 16) we clamp up to 16; the kernel's
        # q_valid / k_valid masks handle the padding correctly.
        block_m = max(min(bucket_size, 32), 16)
        block_n = max(min(bucket_size, 32), 16)
        tiles_per_group = (bucket_size + block_m - 1) // block_m

        offset = bucket.batch_offset
        total = int(bucket.cu_seqlens[-1].item())

        # Slice to this bucket's contiguous range in the sorted batch.
        # These are non-owning views; the Triton kernel writes through them
        # directly into Out via the advanced base pointer.
        Q_b = Q[:, :, offset : offset + total, :]
        K_b = K[:, :, offset : offset + total, :]
        V_b = V[:, :, offset : offset + total, :]
        Out_b = Out[:, :, offset : offset + total, :]

        cu = bucket.cu_seqlens.to(Q.device)
        grid = (T * H * bucket.num_groups * tiles_per_group,)

        _tiled_grouped_attn_fwd[grid](
            Q_b,
            K_b,
            V_b,
            Out_b,
            cu,
            scale,
            Q_b.stride(0),
            Q_b.stride(1),
            Q_b.stride(2),
            Q_b.stride(3),
            K_b.stride(0),
            K_b.stride(1),
            K_b.stride(2),
            K_b.stride(3),
            V_b.stride(0),
            V_b.stride(1),
            V_b.stride(2),
            V_b.stride(3),
            Out_b.stride(0),
            Out_b.stride(1),
            Out_b.stride(2),
            Out_b.stride(3),
            H=H,
            G=bucket.num_groups,
            D=D,
            TILES_PER_GROUP=tiles_per_group,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=BLOCK_D,
        )

    return Out
