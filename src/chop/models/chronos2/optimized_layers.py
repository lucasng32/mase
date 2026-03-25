"""
Optimized drop-in replacement for the MHA module inside GroupSelfAttention using Block-Sparse Row (BSR).
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from .configuration_chronos2 import Chronos2CoreConfig
from .layers import AttentionOutput
from .triton_bsr import run_miniflash_bsr

logger = logging.getLogger(__name__)

class SparseGroupMHA(nn.Module):
    """
    Block-Sparse Self-attention applied along the batch axis using MiniFlash (Triton).
    Replaces the standard GroupSelfAttention inner MHA.
    """
    _global_group_ids_cache: dict[tuple, tuple[torch.Tensor, tuple[int, ...]]] = {}
    _global_metadata_cache: dict[tuple, dict[str, torch.Tensor | int]] = {}
    _max_global_group_cache_size = 64
    _max_global_metadata_cache_size = 128

    def __init__(self, config: Chronos2CoreConfig, group_ids: torch.Tensor | None = None, block_size: int = 16):
        super().__init__()
        self.d_model = config.d_model
        self.kv_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.kv_proj_dim
        self.config = config
        self.block_size = block_size
        
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        # The dropout is handled in the GroupSelfAttention shell natively, 
        # but if we wanted attention-weights dropout, we would handle it here.
        # Since we use Triton fused kernel, dropout inside attention is usually fused if needed.
        # Original Chronos-2 MHA does apply dropout to attention weights.
        # For simplicity in BSR we are leaving it out or you can add it to Triton kernel if required.
        self.dropout = config.dropout_rate

        # Optional static warm-start metadata (still supports runtime dynamic batches)
        self._static_group_ids = group_ids.detach().clone() if group_ids is not None else None

    @classmethod
    def _make_mask_cache_key(cls, mask: torch.Tensor, batch_size: int, device: torch.device) -> tuple:
        return (str(device), batch_size, int(mask.shape[0]), tuple(mask.shape), int(mask.data_ptr()))

    def _infer_group_ids_from_mask(self, mask: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        # mask is additive and shaped (T, 1, B, B) in GroupSelfAttention path.
        if mask.ndim != 4 or mask.shape[2] != batch_size or mask.shape[3] != batch_size:
            raise ValueError(f"Expected mask shape (T, 1, B, B) with B={batch_size}, got {tuple(mask.shape)}")

        allowed = mask[:, 0] == 0
        connectivity = allowed.any(dim=0)
        connectivity.fill_diagonal_(True)

        conn_cpu = connectivity.detach().to(device="cpu")
        n = conn_cpu.shape[0]
        group_ids = torch.full((n,), -1, dtype=torch.long)
        next_gid = 0
        for i in range(n):
            if group_ids[i] >= 0:
                continue
            stack = [i]
            group_ids[i] = next_gid
            while stack:
                cur = stack.pop()
                neighbors = torch.where(conn_cpu[cur])[0].tolist()
                for nb in neighbors:
                    if group_ids[nb] < 0:
                        group_ids[nb] = next_gid
                        stack.append(nb)
            next_gid += 1

        return group_ids.to(device=device)

    def _build_metadata(self, group_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
        device = group_ids.device

        sorted_group_ids, sort_indices = torch.sort(group_ids)
        inverse_sort_indices = torch.argsort(sort_indices)

        batch_size = int(group_ids.shape[0])
        num_blocks_per_t = (batch_size + self.block_size - 1) // self.block_size
        padded_batch_size = num_blocks_per_t * self.block_size

        _, counts = torch.unique_consecutive(sorted_group_ids, return_counts=True)
        group_offsets = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)[:-1]])
        group_starts = group_offsets
        group_ends = group_offsets + counts

        group_block_starts = group_starts // self.block_size
        group_block_ends = (group_ends - 1) // self.block_size

        adj = [set() for _ in range(num_blocks_per_t)]
        for bs, be in zip(group_block_starts.tolist(), group_block_ends.tolist()):
            for r in range(bs, be + 1):
                for c in range(bs, be + 1):
                    adj[r].add(c)

        indptr_list = [0]
        indices_list = []
        for r in range(num_blocks_per_t):
            sorted_cols = sorted(adj[r])
            indices_list.extend(sorted_cols)
            indptr_list.append(len(indices_list))

        indptr = torch.tensor(indptr_list, dtype=torch.int32, device=device)
        indices = torch.tensor(indices_list, dtype=torch.int32, device=device)

        valid_mask = torch.zeros(padded_batch_size, dtype=torch.bool, device=device)
        valid_mask[:batch_size] = True

        padded_sorted_group_ids = torch.full((padded_batch_size,), -1, dtype=torch.long, device=device)
        padded_sorted_group_ids[:batch_size] = sorted_group_ids

        return {
            "sort_indices": sort_indices,
            "inverse_sort_indices": inverse_sort_indices,
            "indptr": indptr,
            "indices": indices,
            "valid_mask": valid_mask,
            "padded_group_ids": padded_sorted_group_ids,
            "padded_batch_size": padded_batch_size,
        }

    def _get_metadata(self, mask: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, torch.Tensor | int]:
        # Group structure is encoded in runtime mask; cache inferred group_ids once and share across all sparse layers.
        mask_key = self._make_mask_cache_key(mask=mask, batch_size=batch_size, device=device)
        cached_groups = self._global_group_ids_cache.get(mask_key)
        if cached_groups is None:
            runtime_group_ids = self._infer_group_ids_from_mask(mask, batch_size=batch_size, device=device)
            signature = tuple(runtime_group_ids.detach().to(device="cpu", dtype=torch.long).tolist())
            self._global_group_ids_cache[mask_key] = (runtime_group_ids.detach().cpu(), signature)
            if len(self._global_group_ids_cache) > self._max_global_group_cache_size:
                self._global_group_ids_cache.clear()
        else:
            runtime_group_ids = cached_groups[0].to(device=device)
            signature = cached_groups[1]

        key = (
            str(device),
            self.block_size,
            batch_size,
            signature,
        )
        cached = self._global_metadata_cache.get(key)
        if cached is None:
            cached = self._build_metadata(runtime_group_ids)
            self._global_metadata_cache[key] = cached
            if len(self._global_metadata_cache) > self._max_global_metadata_cache_size:
                self._global_metadata_cache.clear()
        return cached

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        T, B, D = hidden_states.shape
        device = hidden_states.device

        metadata = self._get_metadata(mask=mask, batch_size=B, device=device)
        sort_indices = metadata["sort_indices"]
        inverse_sort_indices = metadata["inverse_sort_indices"]
        indptr = metadata["indptr"]
        indices = metadata["indices"]
        valid_mask = metadata["valid_mask"]
        padded_group_ids = metadata["padded_group_ids"]
        padded_batch_size = int(metadata["padded_batch_size"])
        
        # Project -> (Time, Batch, Heads, Head_Dim)
        q = self.q(hidden_states).view(T, B, self.n_heads, -1)
        k = self.k(hidden_states).view(T, B, self.n_heads, -1)
        v = self.v(hidden_states).view(T, B, self.n_heads, -1)

        # Sort the batch dimension
        q_sorted = q[:, sort_indices, :, :]
        k_sorted = k[:, sort_indices, :, :]
        v_sorted = v[:, sort_indices, :, :]

        # Scatter into padded buffer
        Padded_B = padded_batch_size
        
        # Minor optimization: if Padded_B == B, we don't need to pad
        if Padded_B == B:
            q_padded = q_sorted
            k_padded = k_sorted
            v_padded = v_sorted
        else:
            q_padded = torch.zeros((T, Padded_B, self.n_heads, self.kv_proj_dim), device=device, dtype=q.dtype)
            k_padded = torch.zeros_like(q_padded)
            v_padded = torch.zeros_like(q_padded)
            
            q_padded[:, :B, :, :] = q_sorted
            k_padded[:, :B, :, :] = k_sorted
            v_padded[:, :B, :, :] = v_sorted

        # Ensure contiguous since we sliced/sorted
        q_padded = q_padded.contiguous()
        k_padded = k_padded.contiguous()
        v_padded = v_padded.contiguous()

        # Execute Triton Kernel (T-aware)
        attn_out_padded = run_miniflash_bsr(
            q_padded, k_padded, v_padded, 
            indptr, indices,
            valid_mask, padded_group_ids
        )

        # Extract valid tokens
        attn_out_sorted = attn_out_padded[:, :B, :]
        
        # Restore original batch order
        attn_out = attn_out_sorted[:, inverse_sort_indices, :].view(T, B, self.inner_dim)

        # Final Projection
        attn_out = self.o(attn_out)
        
        return AttentionOutput(hidden_states=attn_out, attn_weights=None)
