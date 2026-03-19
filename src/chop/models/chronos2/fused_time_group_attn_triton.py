# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Triton fused kernel for Chronos-2 Time + Group Self-Attention.
#
# What is fused vs. PyTorch eager:
#   • FlashAttention-2 tiled SRAM loop  →  O(N) peak HBM instead of O(N²)
#   • RoPE applied in PyTorch before the kernel (tl.flip unsupported in Triton)
#   • RMSNorm uses PyTorch built-in (faster than a hand-written Triton kernel
#     for the d_model sizes used in Chronos-2; vectorised CUDA path wins)
#
# Why NOT faster than SDPA out of the box:
#   PyTorch SDPA already dispatches to cuDNN FlashAttention on modern GPUs.
#   The Triton kernel is a meaningful speedup only vs. *eager* attention, and
#   becomes competitive with SDPA at longer sequences (T >= 1024) where the
#   autotuner finds better tile configs than PyTorch's fixed kernel.
#
# Usage:  drop-in for FusedTimeGroupAttention.
# Requires:  triton >= 2.1  (ships with PyTorch >= 2.1)

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .configuration_chronos2 import Chronos2CoreConfig
from .layers import AttentionOutput


# -----------------------------------------------------------------------------
# Autotuned FlashAttention-2 kernel
#
# @triton.autotune tries every (BLOCK_S, num_warps, num_stages) combination
# listed in configs[] and caches the winner per (B, H, S, D_HEAD, HAS_MASK).
# First call is slow (compilation + timing); subsequent calls use the cache.
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_S": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_S": 64},  num_warps=2, num_stages=3),
    ],
    key=["B", "H", "S", "D_HEAD", "HAS_MASK"],
)
@triton.jit
def _flash_attn_fwd(
    # Inputs (Q/K already have RoPE applied if needed)
    Q_ptr, K_ptr, V_ptr,
    Mask_ptr,       # additive bias [B, H, S, S] -- ignored if not HAS_MASK
    # Output
    Out_ptr,
    # Strides
    stride_b,  stride_h,  stride_s,  stride_d,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_mb, stride_mh, stride_mq, stride_mk,
    # Dimensions (used as autotune keys)
    B, H, S,
    D_HEAD:   tl.constexpr,
    BLOCK_S:  tl.constexpr,   # set by autotuner
    HAS_MASK: tl.constexpr,
):
    bh = tl.program_id(0)   # batch x head
    qt = tl.program_id(1)   # query tile

    b = bh // H
    h = bh  % H

    q_start = qt * BLOCK_S
    q_offs  = q_start + tl.arange(0, BLOCK_S)
    d_offs  = tl.arange(0, D_HEAD)
    q_mask  = q_offs < S

    # Load Q tile
    base_q = Q_ptr + b * stride_b + h * stride_h
    Q = tl.load(
        base_q + q_offs[:, None] * stride_s + d_offs[None, :] * stride_d,
        mask=q_mask[:, None], other=0.0,
    ).to(tl.float32)

    # Accumulators (online softmax)
    acc = tl.zeros([BLOCK_S, D_HEAD], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_S],        dtype=tl.float32)
    m_i = tl.full ([BLOCK_S], float("-inf"), dtype=tl.float32)

    base_k = K_ptr + b * stride_b + h * stride_h
    base_v = V_ptr + b * stride_b + h * stride_h

    # Tile loop over K / V
    for kv_tile in range(tl.cdiv(S, BLOCK_S)):
        k_start = kv_tile * BLOCK_S
        k_offs  = k_start + tl.arange(0, BLOCK_S)
        k_mask  = k_offs < S

        K = tl.load(
            base_k + k_offs[:, None] * stride_s + d_offs[None, :] * stride_d,
            mask=k_mask[:, None], other=0.0,
        ).to(tl.float32)

        V = tl.load(
            base_v + k_offs[:, None] * stride_s + d_offs[None, :] * stride_d,
            mask=k_mask[:, None], other=0.0,
        ).to(tl.float32)

        # QK^T -- no scaling, matches Chronos-2 original
        scores = tl.dot(Q, tl.trans(K))   # [BLOCK_S, BLOCK_S]

        # Optional additive mask (0 for valid, -inf for padding)
        if HAS_MASK:
            m_block = tl.load(
                Mask_ptr
                + b * stride_mb + h * stride_mh
                + q_offs[:, None] * stride_mq + k_offs[None, :] * stride_mk,
                mask=q_mask[:, None] & k_mask[None, :],
                other=float("-inf"),
            ).to(tl.float32)
            scores = scores + m_block

        # Mask padding keys
        scores = tl.where(k_mask[None, :], scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.bfloat16), V.to(tl.bfloat16)
        ).to(tl.float32)
        m_i = m_new

    # Normalise and store
    out_base = Out_ptr + b * stride_ob + h * stride_oh
    tl.store(
        out_base + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od,
        (acc / l_i[:, None]).to(tl.bfloat16),
        mask=q_mask[:, None],
    )


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------

def flash_attn_triton(
    q: torch.Tensor,                        # [B, H, S, D] -- RoPE pre-applied
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,    # [B, H, S, S] additive
) -> torch.Tensor:
    B, H, S, D = q.shape
    assert D & (D - 1) == 0, f"head_dim must be power-of-2, got {D}"

    out      = torch.empty_like(q)
    has_mask = mask is not None
    _mask    = mask if has_mask else q      # dummy ptr, never dereferenced

    # grid is a lambda so autotune can choose BLOCK_S before grid is evaluated
    grid = lambda meta: (B * H, triton.cdiv(S, meta["BLOCK_S"]))

    _flash_attn_fwd[grid](
        q, k, v,
        _mask,
        out,
        q.stride(0),   q.stride(1),   q.stride(2),   q.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        _mask.stride(0) if has_mask else 0,
        _mask.stride(1) if has_mask else 0,
        _mask.stride(2) if has_mask else 0,
        _mask.stride(3) if has_mask else 0,
        B, H, S,
        D_HEAD=D,
        HAS_MASK=has_mask,
    )
    return out


# -----------------------------------------------------------------------------
# Drop-in module
# -----------------------------------------------------------------------------

class FusedTimeGroupAttentionTriton(nn.Module):
    """
    Drop-in replacement for FusedTimeGroupAttention.

    Per attention pass:
        PyTorch RMSNorm  ->  Q/K/V GEMM  ->  PyTorch RoPE (time only)
        ->  Triton FlashAttention-2 (tiled, O(N) HBM)  ->  O GEMM  ->  residual

    RMSNorm uses PyTorch (not a custom Triton kernel) because PyTorch's
    vectorised CUDA path beats hand-written Triton for the d_model sizes in
    Chronos-2.  The Triton kernel handles only the attention core where it
    wins: eliminating the O(N^2) attention matrix from HBM.
    """

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.d_model   = config.d_model
        self.d_kv      = config.d_kv
        self.n_heads   = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.eps       = config.layer_norm_epsilon

        # Time attention
        self.t_q      = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.t_k      = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.t_v      = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.t_o      = nn.Linear(self.inner_dim, self.d_model, bias=False)
        self.t_norm_w = nn.Parameter(torch.ones(self.d_model))

        # Group attention
        self.g_q      = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.g_k      = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.g_v      = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.g_o      = nn.Linear(self.inner_dim, self.d_model, bias=False)
        self.g_norm_w = nn.Parameter(torch.ones(self.d_model))

        # RoPE frequency buffer
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.d_kv, 2, dtype=torch.float32) / self.d_kv)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _rms_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """PyTorch RMSNorm -- faster than Triton for Chronos-2 d_model sizes."""
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if w.dtype in (torch.float16, torch.bfloat16):
            x = x.to(w.dtype)
        return w * x

    def _build_rope(self, x, position_ids):
        inv     = self.inv_freq.to(x.device)
        inv_exp = inv[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_exp = position_ids[:, None, :].float()
        freqs   = (inv_exp @ pos_exp).transpose(1, 2)
        emb     = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q, k, cos, sin):
        cos = cos.unsqueeze(1)   # [B, 1, S, D]
        sin = sin.unsqueeze(1)
        return (q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin)

    def _shape(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_kv).transpose(1, 2)

    def _unshape(self, x):
        B, H, S, D = x.shape
        return x.transpose(1, 2).reshape(B, S, H * D)

    # -------------------------------------------------------------------------
    # Single attention pass
    # -------------------------------------------------------------------------

    def _attend(self, h, norm_w, q_proj, k_proj, v_proj, o_proj,
                mask, position_ids=None):
        normed = self._rms_norm(h, norm_w)

        q = self._shape(q_proj(normed))
        k = self._shape(k_proj(normed))
        v = self._shape(v_proj(normed))

        if position_ids is not None:
            cos, sin = self._build_rope(h, position_ids)
            q, k = self._apply_rope(q, k, cos, sin)

        out = flash_attn_triton(q, k, v, mask=mask)
        return h + self.dropout(o_proj(self._unshape(out)))

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        hidden_states:   torch.Tensor,   # [B, T, D]
        position_ids:    torch.Tensor,   # [B, T]
        attention_mask:  torch.Tensor,   # [B, H, T, T]
        group_time_mask: torch.Tensor,   # [T, H, B, B]
        output_attentions: bool = False,
    ) -> AttentionOutput:

        hidden_states = self._attend(
            hidden_states, self.t_norm_w,
            self.t_q, self.t_k, self.t_v, self.t_o,
            attention_mask, position_ids=position_ids,
        )

        hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states = self._attend(
            hidden_states, self.g_norm_w,
            self.g_q, self.g_k, self.g_v, self.g_o,
            group_time_mask,
        )
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        return AttentionOutput(hidden_states=hidden_states, attn_weights=None)

    # -------------------------------------------------------------------------
    # Weight migration from pretrained FusedTimeGroupAttention
    # -------------------------------------------------------------------------

    @classmethod
    def from_fused(cls, src, config: Chronos2CoreConfig):
        m = cls(config)
        m.t_q.weight.data.copy_(src.time_attention.q.weight)
        m.t_k.weight.data.copy_(src.time_attention.k.weight)
        m.t_v.weight.data.copy_(src.time_attention.v.weight)
        m.t_o.weight.data.copy_(src.time_attention.o.weight)
        m.t_norm_w.data.copy_(src.time_layer_norm.weight)
        m.g_q.weight.data.copy_(src.group_attention.q.weight)
        m.g_k.weight.data.copy_(src.group_attention.k.weight)
        m.g_v.weight.data.copy_(src.group_attention.v.weight)
        m.g_o.weight.data.copy_(src.group_attention.o.weight)
        m.g_norm_w.data.copy_(src.group_layer_norm.weight)
        return m


# -----------------------------------------------------------------------------
# Benchmark
# Run:  python -m chop.models.chronos2.fused_time_group_attn_triton
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from chop.models.chronos2.layers import FusedTimeGroupAttention
    from dataclasses import dataclass

    @dataclass
    class _Cfg:
        d_model: int = 512
        d_kv:    int = 64
        num_heads: int = 8
        dropout_rate: float = 0.0
        layer_norm_epsilon: float = 1e-6
        rope_theta: float = 10_000.0
        dense_act_fn: str = "gelu_new"
        d_ff: int = 2048
        is_gated_act: bool = False
        _attn_implementation: str = "eager"

    DEVICE = "cuda"
    DTYPE  = torch.bfloat16
    WARMUP, REPS = 5, 30

    print(f"GPU  : {torch.cuda.get_device_name()}")
    print(f"dtype: {DTYPE}\n")
    print(f"{'Config':<18} {'eager':>8} {'sdpa':>8} {'triton':>8}"
          f"  {'vs eager':>9}  {'vs sdpa':>8}")
    print("-" * 70)

    for B, T in [(4, 128), (4, 512), (4, 1024), (8, 512)]:
        cfg_e = _Cfg(_attn_implementation="eager")
        cfg_s = _Cfg(_attn_implementation="sdpa")

        ref_e = FusedTimeGroupAttention(cfg_e).to(DEVICE, DTYPE).eval()
        ref_s = FusedTimeGroupAttention(cfg_s).to(DEVICE, DTYPE).eval()
        tri   = FusedTimeGroupAttentionTriton.from_fused(ref_e, cfg_e).to(DEVICE, DTYPE).eval()

        h      = torch.randn(B, T, cfg_e.d_model, device=DEVICE, dtype=DTYPE)
        pos    = torch.arange(T, device=DEVICE).unsqueeze(0).expand(B, -1)
        t_mask = torch.zeros(B, cfg_e.num_heads, T, T, device=DEVICE, dtype=DTYPE)
        g_mask = torch.zeros(T, cfg_e.num_heads, B, B, device=DEVICE, dtype=DTYPE)

        # Correctness vs eager
        with torch.no_grad():
            out_e = ref_e(h, pos, t_mask, g_mask).hidden_states
            out_t = tri  (h, pos, t_mask, g_mask).hidden_states
        max_err = (out_e - out_t).abs().max().item()
        tol = {torch.bfloat16: 0.05, torch.float16: 0.005}.get(DTYPE, 1e-4)
        ok  = "OK" if max_err < tol else f"ERR({max_err:.3f})"

        def _bench(fn):
            torch.cuda.synchronize()
            for _ in range(WARMUP): fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPS):  fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / REPS * 1000

        with torch.no_grad():
            ms_e = _bench(lambda: ref_e(h, pos, t_mask, g_mask))
            ms_s = _bench(lambda: ref_s(h, pos, t_mask, g_mask))
            ms_t = _bench(lambda: tri  (h, pos, t_mask, g_mask))

        print(f"B={B} T={T:<5} {ok:<6}"
              f"  {ms_e:>7.2f}ms  {ms_s:>7.2f}ms  {ms_t:>7.2f}ms"
              f"  {ms_e/ms_t:>7.2f}x   {ms_s/ms_t:>6.2f}x")
    
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(3):
                ref_e(h, pos, t_mask, g_mask)
            for _ in range(3):
                ref_s(h, pos, t_mask, g_mask)
            for _ in range(3):
                tri(h, pos, t_mask, g_mask)

    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total", row_limit=40
    ))

    prof.export_chrome_trace("/tmp/chronos2_trace.json")