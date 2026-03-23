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
from .layers import AttentionOutput, RoPE
from .triton_grouped_attn import is_triton_available, triton_grouped_attention
from .triton_rope_attn import (
    is_triton_available as _rope_triton_available,
    triton_fused_rope_attention,
)

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
        if self._variant == KernelVariant.UNIVARIATE:
            attn_out = self._forward_univariate(hidden_states)
        elif self._variant == KernelVariant.TRITON:
            attn_out = self._forward_triton(hidden_states)
        else:
            attn_out = self._forward_packed_sparse(hidden_states)

        return AttentionOutput(hidden_states=attn_out, attn_weights=None)

    # ------------------------------------------------------------------
    # Dispatch paths
    # ------------------------------------------------------------------
    def _forward_univariate(self, x: torch.Tensor) -> torch.Tensor:
        """All groups size 1 → softmax([single logit]) = 1, so out = V."""
        return self.o(self.v(x))

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

    Fuses the RoPE rotation into the attention kernel so that rotated Q and K
    tensors are **never materialised in global memory**.

    Two dispatch paths (chosen once at construction, no runtime branching):

    1. **Triton** (CUDA + Triton available):
         A single Triton kernel applies RoPE in-register while streaming
         over K/V tiles with an online softmax — identical result to the
         reference path at much lower memory bandwidth.

    2. **PyTorch eager** (CPU or Triton unavailable):
         Computes cos/sin once per forward, applies RoPE with in-place-safe
         operations, then calls standard eager attention.  Slightly cheaper
         than the original ``MHA`` path because cos/sin are computed once
         (not inside ``RoPE.forward`` which re-expands ``inv_freq`` every
         call).

    Weight keys (q/k/v/o and rope_embed.inv_freq) match ``MHA`` exactly so
    ``load_state_dict(mha.state_dict(), strict=False)`` transfers all
    learnable parameters from an existing ``MHA`` instance.  ``inv_freq`` is
    a non-persistent buffer and is not in the state dict.
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

        # inv_freq matches RoPE.inv_freq exactly (same formula)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, self.kv_proj_dim, 2, dtype=torch.int64).float()
                / self.kv_proj_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Dispatch: use Triton on CUDA when available
        self._use_triton: bool = _rope_triton_available()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_rope(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin tables for the given position_ids.

        Returns:
            cos, sin — both shape ``(B, S, D)`` (head dim D = kv_proj_dim).
        """
        inv_freq = self.inv_freq.to(position_ids.device)
        # (B, D/2, 1) x (B, 1, S) → (B, D/2, S) → (B, S, D/2)
        inv_freq_exp = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_exp = position_ids[:, None, :].float()
        freqs = (inv_freq_exp @ pos_exp).transpose(1, 2)          # (B, S, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)                    # (B, S, D)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _eager_rope_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        """Apply RoPE then eager attention. All tensors in (B, H, S, D) form."""
        # cos/sin: (B, S, D) → unsqueeze head dim → (B, 1, S, D)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            h = x.shape[-1] // 2
            return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        scores = torch.matmul(q, k.transpose(-1, -2)) + mask
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
        return torch.matmul(attn_weights, v)

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
        """Fused RoPE + time-attention forward.

        Args:
            hidden_states: ``(B, S, d_model)``
            mask:          ``(B, 1, S, S)`` additive attention mask
            encoder_states: not used (``TimeSelfAttention`` is self-attention only)
            position_ids:  ``(B, S)`` integer position indices
            output_attentions: if True, fall back to eager path that returns weights
        """
        assert position_ids is not None, "position_ids must be provided for RoPEFusedMHA"
        B, S, _ = hidden_states.shape
        H, D = self.n_heads, self.kv_proj_dim

        def _shape(x: torch.Tensor) -> torch.Tensor:
            """(B, S, inner_dim) → (B, H, S, D)"""
            return x.view(B, S, H, D).transpose(1, 2)

        def _unshape(x: torch.Tensor) -> torch.Tensor:
            """(B, H, S, D) → (B, S, inner_dim)"""
            return x.transpose(1, 2).reshape(B, S, self.inner_dim)

        q = _shape(self.q(hidden_states))    # (B, H, S, D)
        k = _shape(self.k(hidden_states))
        v = _shape(self.v(hidden_states))

        if self._use_triton and not output_attentions and hidden_states.is_cuda:
            # Triton path: RoPE fused into the attention kernel
            inv_freq_device = self.inv_freq.to(hidden_states.device)
            # position_ids may be (1, S) — broadcast to (B, S)
            if position_ids.shape[0] == 1 and B > 1:
                position_ids = position_ids.expand(B, -1)
            # Triton kernel expects inv_freq indexed by absolute position, so
            # we pre-scale: freqs[pos] = pos * inv_freq[i]
            # The kernel uses q_range (absolute token index) as position, but
            # actual position_ids may be non-contiguous (e.g. after padding).
            # For TimeSelfAttention position_ids = arange(0, S) (default in
            # Chronos2Encoder.forward), so the kernel's q_range == position_ids.
            # Fall back to eager when position_ids is non-standard.
            if torch.equal(
                position_ids[0],
                torch.arange(S, device=position_ids.device, dtype=position_ids.dtype),
            ):
                # mask: (B, 1, S, S) — contiguous required for stride access
                mask_c = mask.contiguous()
                q_c = q.contiguous()
                k_c = k.contiguous()
                v_c = v.contiguous()
                attn_out = triton_fused_rope_attention(q_c, k_c, v_c, mask_c, inv_freq_device)
                attn_out = _unshape(attn_out)
                return AttentionOutput(hidden_states=self.o(attn_out), attn_weights=None)
            # Non-contiguous positions — fall through to eager

        # PyTorch eager path (CPU / output_attentions / non-contiguous positions)
        cos, sin = self._compute_rope(position_ids, hidden_states.dtype)
        attn_out = self._eager_rope_attn(q, k, v, mask, cos, sin, self.dropout, self.training)
        attn_out = _unshape(attn_out)

        attn_weights_out = None
        if output_attentions:
            # Re-run to get weights (cheap since Q/K are already computed above;
            # we recompute here to keep the forward path branchless for the
            # common no-weights case)
            cos_h = cos.unsqueeze(1)
            sin_h = sin.unsqueeze(1)

            def rotate_half(x: torch.Tensor) -> torch.Tensor:
                h = x.shape[-1] // 2
                return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

            q_rot = q * cos_h + rotate_half(q) * sin_h
            k_rot = k * cos_h + rotate_half(k) * sin_h
            scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) + mask
            attn_weights_out = F.softmax(scores.float(), dim=-1).type_as(scores)

        return AttentionOutput(hidden_states=self.o(attn_out), attn_weights=attn_weights_out)
