"""
Optimized drop-in replacement for the MHA module inside GroupSelfAttention.

``GroupAwareMHA`` replaces ``GroupSelfAttention.self_attention`` (an ``MHA``
instance) with a group-aware attention module that exploits the block-diagonal
sparsity of the group attention mask.  The outer ``GroupSelfAttention`` shell
(layer norm, residual, dropout) is left completely untouched.

Three dispatch paths, chosen at construction time from the precomputed group
partition — the forward pass contains no Python control flow that depends on
runtime tensor values:

  1. **Univariate** (all groups size 1):
       Skip Q, K and the attention matmul entirely.
       softmax([single logit]) = 1, so attn_out = V.

  2. **Triton fused** (CUDA, groups of any size):
       Sort the batch by group, run Q/K/V projections, call a fused Triton
       kernel that iterates only over intra-group pairs — no padding.

  3. **Packed sparse** (CPU or Triton unavailable):
       Pad each group to max_group_size, stack into one tensor, run a
       single batched attention call, scatter real slots back.

All indexing structures are precomputed once at MASE pass application time
and stored as registered buffers.

Weight keys (q/k/v/o) match ``MHA`` exactly, so
``load_state_dict(mha.state_dict())`` works with ``strict=True``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_chronos2 import Chronos2CoreConfig
from .layers import AttentionOutput
from .triton_grouped_attn import is_triton_available, triton_grouped_attention
from .triton_rope_attn import triton_fused_rope_attention
from .triton_bucketed_attn import BucketedPartition, triton_bucketed_attention
from .triton_stitched_attn import StitchedPartition, triton_stitched_attention

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Group partition
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class GroupPartition:
    """Pre-computed group partition derived from a ``group_ids`` tensor.

    Created once at MASE pass application time, never in a forward pass.
    """

    groups: list[torch.Tensor]
    """Per-group batch indices (each a 1-D long tensor on CPU)."""
    num_groups: int
    max_group_size: int
    all_univariate: bool

    @classmethod
    def from_group_ids(cls, group_ids: torch.Tensor) -> GroupPartition:
        ids = group_ids
        groups = [(ids == g).nonzero(as_tuple=True)[0] for g in ids.unique()]
        return cls(
            groups=groups,
            num_groups=len(groups),
            max_group_size=max(g.numel() for g in groups),
            all_univariate=all(g.numel() == 1 for g in groups),
        )


def compute_groups(group_ids: torch.Tensor) -> list[torch.Tensor]:
    """Legacy helper — prefer ``GroupPartition.from_group_ids``."""
    return GroupPartition.from_group_ids(group_ids).groups


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel variant selection
# ═══════════════════════════════════════════════════════════════════════════════
class KernelVariant(Enum):
    TRITON_STITCHED = auto()   # stitched tiles for small groups + bucketed for large
    TRITON_BUCKETED = auto()   # original: one bucket per power-of-2 size (no stitching)
    TRITON = auto()
    PACKED_SPARSE = auto()


class KernelDispatcher:
    @staticmethod
    def select(
        partition: GroupPartition,
        device: torch.device,
    ) -> KernelVariant:
        """Select the best variant.

        Args:
            partition: pre-computed group partition.
            device: target device.
        """
        if device.type == "cuda" and is_triton_available():
            return KernelVariant.TRITON_STITCHED
        return KernelVariant.PACKED_SPARSE


# ═══════════════════════════════════════════════════════════════════════════════
# UnivariateGroupAwareMHA
# ═══════════════════════════════════════════════════════════════════════════════
class UnivariateGroupAwareMHA(nn.Module):
    """Replaces ``MHA`` when the MASE pass determines all groups are size 1.

    For a group of size 1, ``softmax([q·k]) = 1``, so the attention output is
    always ``V``.  Q and K projections are never computed.

    Only ``v`` and ``o`` weights are kept; ``q`` and ``k`` are discarded.
    This is instantiated by the pass, never by ``KernelDispatcher``.
    """

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        inner_dim = config.num_heads * config.d_kv
        self.v = nn.Linear(config.d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, config.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        return AttentionOutput(
            hidden_states=self.o(self.v(hidden_states)),
            attn_weights=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GroupAwareMHA
# ═══════════════════════════════════════════════════════════════════════════════
class GroupAwareMHA(nn.Module):
    """Drop-in replacement for ``MHA`` inside ``GroupSelfAttention``.

    Weight keys (q/k/v/o) are identical to ``MHA`` so weights transfer with
    ``load_state_dict(mha.state_dict(), strict=True)``.

    The MASE pass sets ``group_self_attn.self_attention = GroupAwareMHA(...)``
    leaving the outer ``GroupSelfAttention`` (layer norm, residual, dropout)
    completely untouched.
    """

    def __init__(
        self,
        config: Chronos2CoreConfig,
        partition: GroupPartition,
        variant: KernelVariant | None = None,
    ):
        super().__init__()

        # ── weights matching MHA exactly ──────────────────────────────
        self.d_model: int = config.d_model
        self.kv_proj_dim: int = config.d_kv
        self.n_heads: int = config.num_heads
        self.dropout: float = config.dropout_rate
        self.inner_dim: int = self.n_heads * self.kv_proj_dim
        self.config = config

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # ── dispatch ──────────────────────────────────────────────────
        self._partition = partition
        self._variant = variant or KernelVariant.PACKED_SPARSE

        if self._variant == KernelVariant.TRITON_STITCHED:
            self._precompute_stitched_triton_buffers(partition)
        elif self._variant == KernelVariant.TRITON_BUCKETED:
            self._precompute_bucketed_triton_buffers(partition)
        elif self._variant == KernelVariant.TRITON:
            self._precompute_triton_buffers(partition)
        else:
            self._precompute_packed_buffers(partition)

    # ------------------------------------------------------------------
    # Buffer pre-computation
    # ------------------------------------------------------------------
    def _precompute_triton_buffers(self, partition: GroupPartition) -> None:
        """Build sort permutation and cumulative-seqlens for the Triton path.

        Registered as non-persistent so they don't appear in state_dict() —
        they are always recomputed from the group partition.
        """
        perm_parts: list[torch.Tensor] = []
        cu: list[int] = [0]
        for g in partition.groups:
            perm_parts.append(g)
            cu.append(cu[-1] + g.numel())

        self.register_buffer("_sort_perm", torch.cat(perm_parts), persistent=False)
        self.register_buffer(
            "_cu_seqlens", torch.tensor(cu, dtype=torch.int32), persistent=False
        )

        inv = torch.empty_like(self._sort_perm)
        inv[self._sort_perm] = torch.arange(
            len(self._sort_perm), device=self._sort_perm.device
        )
        self.register_buffer("_unsort_perm", inv, persistent=False)

    @staticmethod
    def _group_ids_from_partition(partition: GroupPartition) -> torch.Tensor:
        """Reconstruct a flat group_ids tensor from a GroupPartition."""
        B = sum(g.numel() for g in partition.groups)
        group_ids = torch.empty(B, dtype=torch.long)
        for label, members in enumerate(partition.groups):
            group_ids[members] = label
        return group_ids

    def _precompute_bucketed_triton_buffers(self, partition: GroupPartition) -> None:
        """Build the BucketedPartition and register sort/unsort permutations.

        Registered as non-persistent so they don't appear in state_dict().
        The BucketedPartition itself is a plain attribute (not a buffer) since
        it contains non-tensor metadata.
        """
        group_ids = self._group_ids_from_partition(partition)
        bp = BucketedPartition.from_group_ids(group_ids)
        self._bucketed_partition = bp
        self.register_buffer(
            "_bucketed_sort_perm", bp.global_sort_perm, persistent=False
        )
        self.register_buffer(
            "_bucketed_unsort_perm", bp.global_unsort_perm, persistent=False
        )

    def _precompute_stitched_triton_buffers(self, partition: GroupPartition) -> None:
        """Build the StitchedPartition and register sort/unsort permutations."""
        group_ids = self._group_ids_from_partition(partition)
        sp = StitchedPartition.from_group_ids(group_ids)
        self._stitched_partition = sp
        self.register_buffer(
            "_stitched_sort_perm", sp.global_sort_perm, persistent=False
        )
        self.register_buffer(
            "_stitched_unsort_perm", sp.global_unsort_perm, persistent=False
        )

    def _precompute_packed_buffers(self, partition: GroupPartition) -> None:
        """Build padded gather/scatter indices for the packed-sparse path.

        Registered as non-persistent so they don't appear in state_dict() —
        they are always recomputed from the group partition.
        """
        G = partition.num_groups
        M = partition.max_group_size
        fmin = torch.finfo(torch.float32).min

        group_indices_padded = torch.zeros(G, M, dtype=torch.long)
        pad_mask = torch.zeros(G, 1, M, M)
        real_flat: list[int] = []
        scatter_b: list[int] = []

        for i, g in enumerate(partition.groups):
            k = g.numel()
            group_indices_padded[i, :k] = g
            if k < M:
                pad_mask[i, 0, :, k:] = fmin
            for j in range(k):
                real_flat.append(i * M + j)
                scatter_b.append(g[j].item())

        self.register_buffer("_group_indices_padded", group_indices_padded, persistent=False)
        self.register_buffer("_pad_mask", pad_mask, persistent=False)
        self.register_buffer(
            "_real_flat_idx", torch.tensor(real_flat, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_scatter_b_idx", torch.tensor(scatter_b, dtype=torch.long), persistent=False
        )
        self._num_groups = G
        self._max_size = M

    # ------------------------------------------------------------------
    # Forward — identical signature to MHA.forward()
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """Group-aware forward pass.

        Args:
            hidden_states: ``(T, B, d_model)`` — already transposed by
                ``GroupSelfAttention`` so the batch axis is the seq dim.
            mask: group-time additive mask from the model; not used at runtime
                by the optimised paths (group structure is baked into buffers).
        """
        if self._variant == KernelVariant.TRITON_STITCHED:
            attn_out = self._forward_stitched_triton(hidden_states)
        elif self._variant == KernelVariant.TRITON_BUCKETED:
            attn_out = self._forward_bucketed_triton(hidden_states)
        elif self._variant == KernelVariant.TRITON:
            attn_out = self._forward_triton(hidden_states)
        else:
            attn_out = self._forward_packed_sparse(hidden_states)

        return AttentionOutput(hidden_states=attn_out, attn_weights=None)

    # ------------------------------------------------------------------
    # Dispatch paths
    # ------------------------------------------------------------------
    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Fused Triton kernel — no padding, no Python loops."""
        T, B, d = x.shape
        H, D = self.n_heads, self.kv_proj_dim

        q = self.q(x).view(T, B, H, D)
        k = self.k(x).view(T, B, H, D)
        v = self.v(x).view(T, B, H, D)

        q = q[:, self._sort_perm, :, :]
        k = k[:, self._sort_perm, :, :]
        v = v[:, self._sort_perm, :, :]

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        out = triton_grouped_attention(
            q,
            k,
            v,
            cu_seqlens=self._cu_seqlens,
            num_groups=self._partition.num_groups,
            scale=1.0,  # Chronos2 uses no sqrt scaling
        )

        out = out.permute(0, 2, 1, 3).reshape(T, B, -1)
        out = out[:, self._unsort_perm, :]

        return self.o(out)

    def _forward_stitched_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Stitched-tile Triton dispatch — small groups bin-packed, large groups bucketed."""
        T, B, d = x.shape
        H, D = self.n_heads, self.kv_proj_dim

        q = self.q(x).view(T, B, H, D)
        k = self.k(x).view(T, B, H, D)
        v = self.v(x).view(T, B, H, D)

        q_s = q[:, self._stitched_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
        k_s = k[:, self._stitched_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
        v_s = v[:, self._stitched_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()

        # Lazy device migration (once)
        sp = self._stitched_partition
        for bucket in sp.buckets.values():
            bucket.cu_seqlens = bucket.cu_seqlens.to(x.device)
        if sp.stitched is not None:
            st = sp.stitched
            st.tile_seq_starts = st.tile_seq_starts.to(x.device)
            st.tile_seq_counts = st.tile_seq_counts.to(x.device)
            st.tile_group_labels = st.tile_group_labels.to(x.device)

        out = triton_stitched_attention(q_s, k_s, v_s, sp, scale=1.0)

        out = out.permute(0, 2, 1, 3).reshape(T, B, -1)
        return self.o(out[:, self._stitched_unsort_perm, :])

    def _forward_bucketed_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Per-bucket Triton dispatch — one kernel launch per occupied bucket."""
        T, B, d = x.shape
        H, D = self.n_heads, self.kv_proj_dim

        q = self.q(x).view(T, B, H, D)
        k = self.k(x).view(T, B, H, D)
        v = self.v(x).view(T, B, H, D)

        # Sort batch into bucket order
        q_s = q[:, self._bucketed_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
        k_s = k[:, self._bucketed_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()
        v_s = v[:, self._bucketed_sort_perm, :, :].permute(0, 2, 1, 3).contiguous()

        # Move metadata tensors to the right device (done lazily once)
        bp = self._bucketed_partition
        for bucket in bp.buckets.values():
            bucket.cu_seqlens = bucket.cu_seqlens.to(x.device)

        out = triton_bucketed_attention(q_s, k_s, v_s, bp, scale=1.0)

        out = out.permute(0, 2, 1, 3).reshape(T, B, -1)
        return self.o(out[:, self._bucketed_unsort_perm, :])

    def _forward_packed_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """Gather → pad → group attention → scatter. Works on any device."""
        T, B, d = x.shape
        G, M = self._num_groups, self._max_size
        H, D = self.n_heads, self.kv_proj_dim

        # Gather and pad: (T, G, M, d) → (T*G, M, d)
        packed = x[:, self._group_indices_padded, :].reshape(T * G, M, d)

        # Project QKV and reshape to (T*G, H, M, D)
        def _shape(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(T * G, M, H, D).transpose(1, 2)

        q = _shape(self.q(packed))
        k = _shape(self.k(packed))
        v = _shape(self.v(packed))

        # Padding mask: (G, 1, M, M) → broadcast over T and H
        batched_mask = (
            self._pad_mask.to(dtype=x.dtype)
            .unsqueeze(0)
            .expand(T, -1, -1, -1, -1)
            .reshape(T * G, 1, M, M)
        )

        # Eager attention — matches original GroupSelfAttention exactly
        scores = torch.matmul(q, k.transpose(-1, -2)) + batched_mask
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_out = torch.matmul(attn_weights, v)  # (T*G, H, M, D)

        # Unshape → project → scatter back
        attn_out = attn_out.transpose(1, 2).reshape(T * G, M, self.inner_dim)
        attn_out = self.o(attn_out).reshape(T, G * M, d)

        return attn_out[:, self._real_flat_idx, :]


# ═══════════════════════════════════════════════════════════════════════════════
# RoPEFusedMHA
# ═══════════════════════════════════════════════════════════════════════════════


class RoPEFusedMHA(nn.Module):
    """Drop-in replacement for ``MHA`` inside ``TimeSelfAttention``.

    Fuses the RoPE rotation into the Triton attention kernel so that rotated Q
    and K tensors are never materialised in global memory.

    Weight keys (q/k/v/o) match ``MHA`` exactly so
    ``load_state_dict(mha.state_dict(), strict=False)`` transfers all
    learnable parameters.  ``inv_freq`` is a non-persistent buffer and is not
    in the state dict.
    """

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()

        self.d_model: int = config.d_model
        self.kv_proj_dim: int = config.d_kv
        self.n_heads: int = config.num_heads
        self.dropout: float = config.dropout_rate
        self.inner_dim: int = self.n_heads * self.kv_proj_dim
        self.config = config

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, self.kv_proj_dim, 2, dtype=torch.int64).float()
                / self.kv_proj_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------
    # Forward — identical signature to MHA.forward()
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """Fused RoPE + time-attention forward via Triton kernel.

        Args:
            hidden_states: ``(B, S, d_model)``
            mask:          ``(B, 1, S, S)`` additive attention mask
            encoder_states: unused (self-attention only)
            position_ids:  ``(B, S)`` integer position indices
            output_attentions: unused (Triton path does not return weights)
        """
        assert position_ids is not None, "position_ids must be provided for RoPEFusedMHA"
        B, S, _ = hidden_states.shape
        H, D = self.n_heads, self.kv_proj_dim

        q = self.q(hidden_states).view(B, S, H, D).transpose(1, 2).contiguous()  # (B, H, S, D)
        k = self.k(hidden_states).view(B, S, H, D).transpose(1, 2).contiguous()
        v = self.v(hidden_states).view(B, S, H, D).transpose(1, 2).contiguous()

        if position_ids.shape[0] == 1 and B > 1:
            position_ids = position_ids.expand(B, -1)

        # The Triton kernel indexes the mask as (B, 1, S, S) using raw strides.
        # The model passes a (B, 1, 1, S) broadcast mask, so expand it to the
        # full square shape before handing off to the kernel.
        if mask.shape[2] != S:
            mask = mask.expand(B, 1, S, S)

        attn_out = triton_fused_rope_attention(
            q, k, v, mask.contiguous(), self.inv_freq.to(hidden_states.device)
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.inner_dim)
        return AttentionOutput(hidden_states=self.o(attn_out), attn_weights=None)
