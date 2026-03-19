# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

from dataclasses import dataclass

import torch
# from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import ModelOutput

from .configuration_chronos2 import Chronos2CoreConfig


class RoPE(nn.Module):
    """Applies rotary position embeddings (RoPE) to input tensors.

    Implementation adapted from:
    https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/models/llama/modeling_llama.py#L95
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.inv_freq: torch.Tensor  # type hint for type checker
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (RoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPE.rotate_half(k) * sin)
        return q_embed, k_embed


class Chronos2LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# This is how transformers keeps track of LayerNorm classes ¯\_(ツ)_/¯
ALL_LAYERNORM_LAYERS.append(Chronos2LayerNorm)  # type: ignore


class MLP(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()

        assert not config.is_gated_act, "gated activations are unsupported"
        self.mlp: nn.Module = MLP(config)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


@dataclass
class AttentionOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


class MHA(nn.Module):
    """Multi-head Attention Layer"""

    def __init__(self, config: Chronos2CoreConfig, use_rope: bool = True):
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

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE(dim=self.kv_proj_dim, base=config.rope_theta)

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager attention implementation using manual matmul.

        Args:
            query_states: [batch, n_heads, seq_len, kv_proj_dim]
            key_states: [batch, n_heads, seq_len, kv_proj_dim]
            value_states: [batch, n_heads, seq_len, kv_proj_dim]
            mask: [batch, n_heads, q_len, kv_len]

        Returns:
            attn_output: [batch, n_heads, seq_len, kv_proj_dim]
            attn_weights: [batch, n_heads, q_len, kv_len]
        """
        # Compute attention weights (no scaling - this is the original Chronos-2 implementation)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))  # "bnqd,bnkd->bnqk"
        scores += mask
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """SDPA attention implementation using torch.nn.functional.scaled_dot_product_attention.

        Args:
            query_states: [batch, n_heads, seq_len, kv_proj_dim]
            key_states: [batch, n_heads, seq_len, kv_proj_dim]
            value_states: [batch, n_heads, seq_len, kv_proj_dim]
            mask: [batch, n_heads, q_len, kv_len] - additive mask (0 for valid, -inf for invalid)

        Returns:
            attn_output: [batch, n_heads, seq_len, kv_proj_dim]
            attn_weights: None (SDPA doesn't return weights)
        """
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,  # Match eager implementation (no scaling)
        )

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """Multi-head attention forward pass.

        Args:
            hidden_states : Input tensor of shape [batch_size, seq_len, d_model]
            mask : Attention mask tensor of shape [batch_size, num_heads, q_len, kv_len]
            encoder_states : Encoder states for cross-attention. Defaults to None.
            position_ids : Position IDs for RoPE. Defaults to None.
            output_attentions : Whether to return attention weights. Defaults to False.

        Returns:
            AttentionOutput: Contains:
                - hidden_states : Output tensor of shape [batch_size, seq_len, d_model]
                - attn_weights : Attention weights if output_attentions=True
        """
        if self.use_rope:
            assert position_ids is not None, "position_ids must be provided when self.use_rope=True"

        # Force eager attention if output_attentions is True (only eager returns weights)
        attn_implementation = self.config._attn_implementation
        if output_attentions:
            attn_implementation = "eager"

        seq_length = hidden_states.shape[1]

        def shape(states: torch.Tensor) -> torch.Tensor:
            """(batch, seq_len, inner_dim) -> (batch, n_heads, seq_len, kv_proj_dim)"""
            # return rearrange(states, "b s (h d) -> b h s d", h=self.n_heads, s=seq_length, d=self.kv_proj_dim)
            batch_size = states.shape[0]
            seq_length = states.shape[1]
            states = states.view(batch_size, seq_length, self.n_heads, self.kv_proj_dim)
            return states.transpose(1, 2)

        def unshape(states: torch.Tensor) -> torch.Tensor:
            """(batch, n_heads, seq_len, kv_proj_dim) -> (batch, seq_len, inner_dim)"""
            # return rearrange(states, "b h s d -> b s (h d)", h=self.n_heads, s=seq_length, d=self.kv_proj_dim)
            batch_size = states.shape[0]
            seq_length = states.shape[2] 
            states = states.transpose(1, 2)
            return states.reshape(batch_size, seq_length, -1)
            

        # Construct query states
        query_states = shape(self.q(hidden_states))
        is_cross_attention = encoder_states is not None

        # Construct key/value states
        if is_cross_attention:
            key_states = shape(self.k(encoder_states))
            value_states = shape(self.v(encoder_states))
        else:
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))
            if self.use_rope:
                cos, sin = self.rope_embed(value_states, position_ids)
                query_states, key_states = RoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if attn_implementation == "sdpa":
            attn_output, attn_weights = self._sdpa_attention(query_states, key_states, value_states, mask)
        else:  # eager
            attn_output, attn_weights = self._eager_attention(query_states, key_states, value_states, mask)

        # Project attention output
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        return AttentionOutput(hidden_states=attn_output, attn_weights=attn_weights if output_attentions else None)

import torch
from torch import nn

class FusedQKV_MHA(nn.Module):
    """A drop-in replacement that steals weights from an old MHA module and fuses them."""
    def __init__(self, old_mha):
        super().__init__()
        self.config = old_mha.config
        self.n_heads = old_mha.n_heads
        self.kv_proj_dim = old_mha.kv_proj_dim
        self.inner_dim = old_mha.inner_dim
        self.use_rope = old_mha.use_rope
        self.dropout = old_mha.dropout
        
        # 1. Steal the Rope and Output projections
        if self.use_rope:
            self.rope_embed = old_mha.rope_embed
        self.o = old_mha.o
        
        # 2. Create the massive Fused QKV layer
        self.qkv = nn.Linear(self.config.d_model, 3 * self.inner_dim, bias=False)
        
        # 3. Copy and concatenate the weights from the old module natively in VRAM
        with torch.no_grad():
            self.qkv.weight.data = torch.cat([
                old_mha.q.weight.data, 
                old_mha.k.weight.data, 
                old_mha.v.weight.data
            ], dim=0)

        # 4. Steal the attention routing methods from the old module
        self._sdpa_attention = old_mha._sdpa_attention
        self._eager_attention = old_mha._eager_attention

    def forward(
        self, hidden_states, mask, encoder_states=None, position_ids=None, output_attentions=False
    ):
        if self.use_rope:
            assert position_ids is not None
            
        is_cross_attention = encoder_states is not None
        
        def shape(states: torch.Tensor) -> torch.Tensor:
            B, S = states.shape[0], states.shape[1]
            return states.view(B, S, self.n_heads, self.kv_proj_dim).transpose(1, 2)

        def unshape(states: torch.Tensor) -> torch.Tensor:
            B, S = states.shape[0], states.shape[2] 
            return states.transpose(1, 2).reshape(B, S, -1)

        # ==========================================
        # THE FUSED QKV LOGIC
        # ==========================================
        if not is_cross_attention:
            # ONE read from HBM!
            qkv_states = self.qkv(hidden_states)
            q_out, k_out, v_out = qkv_states.chunk(3, dim=-1)
            
            query_states = shape(q_out)
            key_states = shape(k_out)
            value_states = shape(v_out)
        else:
            # If it's cross attention, we'd need a separate fusion strategy for encoder_states. 
            # For this test, we just let it fail gracefully or you can pass through the old logic.
            raise NotImplementedError("This hack is only for Self-Attention layers.")

        if self.use_rope:
            # RoPE expects the class from layers.py, ensure it's imported
            from chop.models.chronos2.layers import RoPE 
            cos, sin = self.rope_embed(value_states, position_ids)
            query_states, key_states = RoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_implementation = self.config._attn_implementation if not output_attentions else "eager"
        
        if attn_implementation == "sdpa":
            attn_output, attn_weights = self._sdpa_attention(query_states, key_states, value_states, mask)
        else:
            attn_output, attn_weights = self._eager_attention(query_states, key_states, value_states, mask)

        from transformers.utils import ModelOutput
        from dataclasses import dataclass
        @dataclass
        class AttentionOutput(ModelOutput):
            hidden_states: torch.Tensor | None = None
            attn_weights: torch.Tensor | None = None

        return AttentionOutput(
            hidden_states=self.o(unshape(attn_output)), 
            attn_weights=attn_weights if output_attentions else None
        )


class TimeSelfAttention(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.self_attention = MHA(config, use_rope=True)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, position_ids=position_ids, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class GroupSelfAttention(nn.Module):
    """Self-attention applied along the batch axis masked by the group attention mask"""

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        # we don't use RoPE here because there's no natural ordering along the batch axis
        self.self_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False
    ) -> AttentionOutput:
        # flip time and batch axes because attention operates along dim=-2
        # hidden_states = rearrange(hidden_states, "batch time d -> time batch d")
        hidden_states = hidden_states.transpose(0, 1)
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # flip time and batch axes back to their original position
        #hidden_states = rearrange(hidden_states, "time batch d -> batch time d")
        hidden_states = hidden_states.transpose(0, 1)

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class ResidualBlock(nn.Module):
    """A generic residual block which can be used for input and output embedding layers"""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = Chronos2LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out

class FusedTimeGroupAttention(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.time_attention = MHA(config, use_rope=True)
        self.time_layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.time_dropout = nn.Dropout(config.dropout_rate)

        self.group_attention = MHA(config, use_rope=False)
        self.group_layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.group_dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        normed_hidden_states = self.time_layer_norm(hidden_states)
        attention_output_t = self.time_attention(
            normed_hidden_states, position_ids=position_ids, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.time_dropout(attention_output_t[0])

        hidden_states = hidden_states.transpose(0, 1)
        normed_hidden_states = self.group_layer_norm(hidden_states)
        attention_output_g = self.group_attention(
            normed_hidden_states, mask=group_time_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.group_dropout(attention_output_g[0])
        hidden_states = hidden_states.transpose(0, 1)
        
        return AttentionOutput(hidden_states=hidden_states, attn_weights=None)
