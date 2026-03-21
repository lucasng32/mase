"""
Optimized drop-in replacements for Chronos-2 attention layers.

``FastGroupSelfAttention`` replaces ``GroupSelfAttention`` with a computation
that exploits the block-diagonal sparsity of the group attention mask.

Three dispatch paths, all chosen at construction time from the precomputed
group partition — the forward pass contains no Python control flow that
depends on runtime tensor values:

  1. **Univariate** (all groups size 1):
       Skip Q, K and the attention matmul entirely.
       softmax([single logit]) = 1, so attn_out = V.

  2. **Triton fused** (CUDA, groups of any size):
       Sort the batch by group, run Q/K/V projections, call a fused Triton
       kernel that iterates only over intra-group pairs — no padding.

  3. **Packed sparse** (CPU or Triton unavailable):
       Pad each group to max_group_size, stack into one tensor, run a
       single batched MHA call, scatter real slots back.

All indexing structures are precomputed once at MASE pass application time
and stored as registered buffers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import torch
from torch import nn

from .configuration_chronos2 import Chronos2CoreConfig
from .layers import AttentionOutput, Chronos2LayerNorm, MHA
from .triton_grouped_attn import is_triton_available, triton_grouped_attention

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
        ids = group_ids.cpu()
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
    UNIVARIATE = auto()
    TRITON = auto()
    PACKED_SPARSE = auto()


class KernelDispatcher:
    """Choose the best kernel variant for a given partition and device.

    Default auto-selection order (all paths remain available via explicit
    ``variant=`` argument):

    1. **Univariate** — all groups size 1: skip Q/K entirely.
    2. **Triton** — JIT-compiled tiled kernel.  Used on CUDA when Triton is
       available and groups are large enough to benefit from tile parallelism
       (``max_group_size > TRITON_CROSSOVER``).
    3. **Packed sparse** — pure PyTorch fallback for CPU or small groups.
    """

    TRITON_CROSSOVER = 8  # min group size at which Triton beats packed-sparse

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
        if partition.all_univariate:
            return KernelVariant.UNIVARIATE
        if (
            device.type == "cuda"
            and is_triton_available()
            and partition.max_group_size > KernelDispatcher.TRITON_CROSSOVER
        ):
            return KernelVariant.TRITON
        return KernelVariant.PACKED_SPARSE


# ═══════════════════════════════════════════════════════════════════════════════
# FastGroupSelfAttention
# ═══════════════════════════════════════════════════════════════════════════════
class FastGroupSelfAttention(nn.Module):
    """Drop-in replacement for ``GroupSelfAttention``.

    State-dict keys match ``GroupSelfAttention`` so weights load with
    ``load_state_dict(..., strict=False)``.
    """

    def __init__(
        self,
        config: Chronos2CoreConfig,
        partition: GroupPartition,
        variant: KernelVariant | None = None,
    ):
        super().__init__()
        self.self_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        self._partition = partition
        self._variant = variant or KernelVariant.PACKED_SPARSE

        if partition.all_univariate:
            self._variant = KernelVariant.UNIVARIATE
        elif self._variant == KernelVariant.TRITON:
            self._precompute_triton_buffers(partition)
        else:
            self._precompute_packed_buffers(partition)

    # ------------------------------------------------------------------
    # Buffer pre-computation
    # ------------------------------------------------------------------
    def _precompute_triton_buffers(self, partition: GroupPartition) -> None:
        """Build sort permutation and cumulative-seqlens for the Triton path."""
        perm_parts: list[torch.Tensor] = []
        cu: list[int] = [0]
        for g in partition.groups:
            perm_parts.append(g)
            cu.append(cu[-1] + g.numel())

        self.register_buffer("_sort_perm", torch.cat(perm_parts))
        self.register_buffer(
            "_cu_seqlens", torch.tensor(cu, dtype=torch.int32)
        )

        # Inverse permutation for unsorting the output.
        inv = torch.empty_like(self._sort_perm)
        inv[self._sort_perm] = torch.arange(len(self._sort_perm))
        self.register_buffer("_unsort_perm", inv)

    def _precompute_packed_buffers(self, partition: GroupPartition) -> None:
        """Build padded gather/scatter indices for the packed-sparse path."""
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

        self.register_buffer("_group_indices_padded", group_indices_padded)
        self.register_buffer("_pad_mask", pad_mask)
        self.register_buffer(
            "_real_flat_idx", torch.tensor(real_flat, dtype=torch.long)
        )
        self.register_buffer(
            "_scatter_b_idx", torch.tensor(scatter_b, dtype=torch.long)
        )
        self._num_groups = G
        self._max_size = M

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """
        Args:
            hidden_states:  ``(B, T, d_model)``
            attention_mask: ``(T, 1, B, B)`` additive group-time mask (used
                only as a dtype/device reference in optimised paths).
        """
        hidden_states = hidden_states.transpose(0, 1)  # (T, B, d)
        T, B, d = hidden_states.shape
        normed = self.layer_norm(hidden_states)

        if self._variant == KernelVariant.UNIVARIATE:
            attn_out = self._forward_univariate(normed)
        elif self._variant == KernelVariant.TRITON:
            attn_out = self._forward_triton(normed)
        else:
            attn_out = self._forward_packed_sparse(normed, hidden_states.dtype)

        output = hidden_states + self.dropout(attn_out)
        return AttentionOutput(
            hidden_states=output.transpose(0, 1), attn_weights=None
        )

    # ------------------------------------------------------------------
    # Dispatch paths
    # ------------------------------------------------------------------
    def _forward_univariate(self, normed: torch.Tensor) -> torch.Tensor:
        """All groups have size 1 → softmax is trivially 1, skip Q and K."""
        v_out = self.self_attention.v(normed)  # (T, B, inner_dim)
        return self.self_attention.o(v_out)  # (T, B, d)

    def _forward_triton(self, normed: torch.Tensor) -> torch.Tensor:
        """Fused Triton kernel — no padding, no Python loops."""
        T, B, d = normed.shape
        mha = self.self_attention
        H, D = mha.n_heads, mha.kv_proj_dim

        # Project Q, K, V: (T, B, d) → (T, B, inner) → (T, B, H, D)
        q = mha.q(normed).view(T, B, H, D)
        k = mha.k(normed).view(T, B, H, D)
        v = mha.v(normed).view(T, B, H, D)

        # Sort batch by group: (T, B, H, D) → (T, B_sorted, H, D)
        q = q[:, self._sort_perm, :, :]
        k = k[:, self._sort_perm, :, :]
        v = v[:, self._sort_perm, :, :]

        # Reshape to (T, H, B_sorted, D) for the kernel
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

        # (T, H, B_sorted, D) → (T, B_sorted, H, D) → (T, B_sorted, inner)
        out = out.permute(0, 2, 1, 3).reshape(T, B, -1)

        # Unsort back to original batch order
        out = out[:, self._unsort_perm, :]

        return mha.o(out)  # (T, B, d)

    def _forward_packed_sparse(
        self, normed: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Gather → pad → batched MHA → scatter. Works on any device."""
        T, B, d = normed.shape
        G, M = self._num_groups, self._max_size

        packed = normed[:, self._group_indices_padded, :].reshape(T * G, M, d)

        batched_mask = (
            self._pad_mask.to(dtype=dtype)
            .unsqueeze(0)
            .expand(T, -1, -1, -1, -1)
            .reshape(T * G, 1, M, M)
        )

        result = self.self_attention(
            packed, mask=batched_mask, output_attentions=False
        )

        flat = result.hidden_states.reshape(T, G * M, d)
        return flat[:, self._real_flat_idx, :]  # (T, B, d)
