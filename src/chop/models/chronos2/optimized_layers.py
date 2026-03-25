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
    def __init__(self, config: Chronos2CoreConfig, group_ids: torch.Tensor, block_size: int = 16):
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

        self._precompute_static_metadata(group_ids)

    def _precompute_static_metadata(self, group_ids: torch.Tensor):
        device = group_ids.device
        
        # 1. Sort group IDs
        sorted_group_ids, sort_indices = torch.sort(group_ids)
        inverse_sort_indices = torch.argsort(sort_indices)
        
        # 2. Dimensions for padded batch
        batch_size = group_ids.shape[0]
        num_blocks_per_t = (batch_size + self.block_size - 1) // self.block_size
        padded_batch_size = num_blocks_per_t * self.block_size
        
        # 3. Block-Dense adjacency within each group
        unique_groups, counts = torch.unique_consecutive(sorted_group_ids, return_counts=True)
        
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
            sorted_cols = sorted(list(adj[r]))
            indices_list.extend(sorted_cols)
            indptr_list.append(len(indices_list))
            
        indptr = torch.tensor(indptr_list, dtype=torch.int32, device=device)
        indices = torch.tensor(indices_list, dtype=torch.int32, device=device)
        
        # 4. Valid mask and padded group IDs (1D length = padded_batch_size)
        valid_mask = torch.zeros(padded_batch_size, dtype=torch.bool, device=device)
        valid_mask[:batch_size] = True
        
        padded_sorted_group_ids = torch.full((padded_batch_size,), -1, dtype=torch.long, device=device)
        padded_sorted_group_ids[:batch_size] = sorted_group_ids

        # Register buffers (not saved in state dict, generated on the fly)
        self.register_buffer("_sort_indices", sort_indices, persistent=False)
        self.register_buffer("_inverse_sort_indices", inverse_sort_indices, persistent=False)
        self.register_buffer("_indptr", indptr, persistent=False)
        self.register_buffer("_indices", indices, persistent=False)
        self.register_buffer("_valid_mask", valid_mask, persistent=False)
        self.register_buffer("_padded_group_ids", padded_sorted_group_ids, persistent=False)
        
        # Save scalar constants
        self._batch_size = batch_size
        self._padded_batch_size = padded_batch_size

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
        
        # Project -> (Time, Batch, Heads, Head_Dim)
        q = self.q(hidden_states).view(T, B, self.n_heads, -1)
        k = self.k(hidden_states).view(T, B, self.n_heads, -1)
        v = self.v(hidden_states).view(T, B, self.n_heads, -1)

        # Sort the batch dimension
        q_sorted = q[:, self._sort_indices, :, :]
        k_sorted = k[:, self._sort_indices, :, :]
        v_sorted = v[:, self._sort_indices, :, :]

        # Scatter into padded buffer
        Padded_B = self._padded_batch_size
        
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
            self._indptr, self._indices, 
            self._valid_mask, self._padded_group_ids
        )

        # Extract valid tokens
        attn_out_sorted = attn_out_padded[:, :B, :]
        
        # Restore original batch order
        attn_out = attn_out_sorted[:, self._inverse_sort_indices, :].view(T, B, self.inner_dim)

        # Final Projection
        attn_out = self.o(attn_out)
        
        return AttentionOutput(hidden_states=attn_out, attn_weights=None)
