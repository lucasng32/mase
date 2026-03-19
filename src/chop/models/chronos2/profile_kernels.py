"""
Profile kernel launches per op, labeled by implementation (eager / sdpa / triton).

Output 1 — terminal table: kernel name, count, total us, % of total, which impl
Output 2 — /tmp/chronos2_trace.json: open in chrome://tracing
           each bar shows the NVTX label (op name) it belongs to

Run:
    python -m chop.models.chronos2.profile_kernels
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import contextmanager
from dataclasses import dataclass

from chop.models.chronos2.layers import FusedTimeGroupAttention
from chop.models.chronos2.fused_time_group_attn_triton import FusedTimeGroupAttentionTriton


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented wrappers — each sub-op gets its own NVTX range
# so the profiler trace shows exactly which kernel belongs to which op.
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedEager(nn.Module):
    """Wraps FusedTimeGroupAttention (eager) with per-op NVTX ranges."""

    def __init__(self, inner: FusedTimeGroupAttention):
        super().__init__()
        self.inner = inner
        # Patch MHA._eager_attention so we can label QK / softmax / AV
        _patch_eager_mha(inner.time_attention,  "time")
        _patch_eager_mha(inner.group_attention, "group")

    def forward(self, h, pos, t_mask, g_mask):
        with nvtx("eager/time_norm"):
            normed = self.inner.time_layer_norm(h)
        with nvtx("eager/time_mha"):
            attn_t = self.inner.time_attention(
                normed, position_ids=pos, mask=t_mask)
        with nvtx("eager/time_residual"):
            h = h + self.inner.time_dropout(attn_t[0])

        h = h.transpose(0, 1)
        with nvtx("eager/group_norm"):
            normed = self.inner.group_layer_norm(h)
        with nvtx("eager/group_mha"):
            attn_g = self.inner.group_attention(
                normed, mask=g_mask)
        with nvtx("eager/group_residual"):
            h = h + self.inner.group_dropout(attn_g[0])
        h = h.transpose(0, 1)
        return h


class InstrumentedSDPA(nn.Module):
    """Wraps FusedTimeGroupAttention (sdpa) with per-op NVTX ranges."""

    def __init__(self, inner: FusedTimeGroupAttention):
        super().__init__()
        self.inner = inner

    def forward(self, h, pos, t_mask, g_mask):
        with nvtx("sdpa/time_norm"):
            normed = self.inner.time_layer_norm(h)
        with nvtx("sdpa/time_mha"):          # sdpa dispatches cuDNN FA here
            attn_t = self.inner.time_attention(
                normed, position_ids=pos, mask=t_mask)
        with nvtx("sdpa/time_residual"):
            h = h + self.inner.time_dropout(attn_t[0])

        h = h.transpose(0, 1)
        with nvtx("sdpa/group_norm"):
            normed = self.inner.group_layer_norm(h)
        with nvtx("sdpa/group_mha"):
            attn_g = self.inner.group_attention(
                normed, mask=g_mask)
        with nvtx("sdpa/group_residual"):
            h = h + self.inner.group_dropout(attn_g[0])
        h = h.transpose(0, 1)
        return h


class InstrumentedTriton(nn.Module):
    """Wraps FusedTimeGroupAttentionTriton with per-op NVTX ranges."""

    def __init__(self, inner: FusedTimeGroupAttentionTriton):
        super().__init__()
        self.m = inner

    def forward(self, h, pos, t_mask, g_mask):
        # ── Time pass ────────────────────────────────────────────────────────
        with nvtx("triton/time_norm"):
            normed = self.m._rms_norm(h, self.m.t_norm_w)
        with nvtx("triton/time_qkv"):
            q = self.m._shape(self.m.t_q(normed))
            k = self.m._shape(self.m.t_k(normed))
            v = self.m._shape(self.m.t_v(normed))
        with nvtx("triton/time_rope"):
            cos, sin = self.m._build_rope(h, pos)
            q, k = self.m._apply_rope(q, k, cos, sin)
        with nvtx("triton/time_flash"):          # ← the one Triton kernel
            from chop.models.chronos2.fused_time_group_attn_triton import flash_attn_triton
            out = flash_attn_triton(q, k, v, mask=t_mask)
        with nvtx("triton/time_o_proj"):
            out = self.m.t_o(self.m._unshape(out))
        with nvtx("triton/time_residual"):
            h = h + out

        # ── Group pass ───────────────────────────────────────────────────────
        h = h.transpose(0, 1).contiguous()
        with nvtx("triton/group_norm"):
            normed = self.m._rms_norm(h, self.m.g_norm_w)
        with nvtx("triton/group_qkv"):
            q = self.m._shape(self.m.g_q(normed))
            k = self.m._shape(self.m.g_k(normed))
            v = self.m._shape(self.m.g_v(normed))
        with nvtx("triton/group_flash"):         # ← the one Triton kernel
            out = flash_attn_triton(q, k, v, mask=g_mask)
        with nvtx("triton/group_o_proj"):
            out = self.m.g_o(self.m._unshape(out))
        with nvtx("triton/group_residual"):
            h = h + out
        h = h.transpose(0, 1).contiguous()
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def nvtx(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _patch_eager_mha(mha, prefix: str):
    """Monkey-patch MHA._eager_attention to add NVTX ranges around each op."""
    orig = mha._eager_attention

    def patched(query_states, key_states, value_states, mask):
        with nvtx(f"eager/{prefix}_QK"):
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            scores += mask
        with nvtx(f"eager/{prefix}_softmax"):
            attn_w = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
            attn_w = torch.nn.functional.dropout(attn_w, p=mha.dropout, training=mha.training)
        with nvtx(f"eager/{prefix}_AV"):
            out = torch.matmul(attn_w, value_states)
        return out, attn_w

    mha._eager_attention = patched


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


# ─────────────────────────────────────────────────────────────────────────────
# Build modules
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda"
DTYPE  = torch.bfloat16
B, T   = 4, 512

cfg_e = _Cfg(_attn_implementation="eager")
cfg_s = _Cfg(_attn_implementation="sdpa")

ref_e = FusedTimeGroupAttention(cfg_e).to(DEVICE, DTYPE).eval()
ref_s = FusedTimeGroupAttention(cfg_s).to(DEVICE, DTYPE).eval()
tri_m = FusedTimeGroupAttentionTriton.from_fused(ref_e, cfg_e).to(DEVICE, DTYPE).eval()

eager_mod  = InstrumentedEager(ref_e)
sdpa_mod   = InstrumentedSDPA(ref_s)
triton_mod = InstrumentedTriton(tri_m)

h      = torch.randn(B, T, cfg_e.d_model, device=DEVICE, dtype=DTYPE)
pos    = torch.arange(T, device=DEVICE).unsqueeze(0).expand(B, -1)
t_mask = torch.zeros(B, cfg_e.num_heads, T, T, device=DEVICE, dtype=DTYPE)
g_mask = torch.zeros(T, cfg_e.num_heads, B, B, device=DEVICE, dtype=DTYPE)

# ─────────────────────────────────────────────────────────────────────────────
# Warm up (also triggers Triton autotuner before profiling)
# ─────────────────────────────────────────────────────────────────────────────

print("Warming up (Triton autotune runs here)...")
with torch.no_grad():
    for _ in range(5):
        eager_mod (h, pos, t_mask, g_mask)
        sdpa_mod  (h, pos, t_mask, g_mask)
        triton_mod(h, pos, t_mask, g_mask)
torch.cuda.synchronize()
print("Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Profile — 3 passes of each implementation, back to back
# ─────────────────────────────────────────────────────────────────────────────

TRACE = "/tmp/chronos2_trace.json"

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    with_modules=False,
) as prof:
    with torch.no_grad():
        for _ in range(3):
            with record_function("=== EAGER ==="):
                eager_mod(h, pos, t_mask, g_mask)
        for _ in range(3):
            with record_function("=== SDPA ==="):
                sdpa_mod(h, pos, t_mask, g_mask)
        for _ in range(3):
            with record_function("=== TRITON ==="):
                triton_mod(h, pos, t_mask, g_mask)

torch.cuda.synchronize()

# ─────────────────────────────────────────────────────────────────────────────
# Terminal table — sorted by CUDA time, grouped by impl/op
# ─────────────────────────────────────────────────────────────────────────────

events     = prof.key_averages(group_by_input_shape=False)
raw_events = prof.events()   # individual (non-averaged) events

print(f"Total event keys: {len(events)}  |  Raw events: {len(raw_events)}")

# ── Only look at non-annotation events for CUDA time ─────────────────────────
kernel_events = [e for e in raw_events if not e.is_user_annotation]
print(f"Non-annotation events: {len(kernel_events)}")

if not kernel_events:
    print("ERROR: no kernel events captured.")
    raise SystemExit(1)

def _t(e):
    # Raw FunctionEvent uses .device_time (us), not the *_total variants
    # which only exist on FunctionEventAvg from key_averages()
    v = getattr(e, "device_time", 0)
    if v and v > 0:
        return v
    # fallback for any averaged events mixed in
    for a in ("self_device_time_total", "device_time_total"):
        v = getattr(e, a, None)
        if v and v > 0:
            return v
    return 0

# ── Build NVTX label → kernels map ────────────────────────────────────────────
# Each raw event has a list of CPU-side parent annotations in e.cpu_parent
# (a linked list).  Walk up to find the deepest NVTX range that names the op.
def _nvtx_label(e):
    node = getattr(e, "cpu_parent", None)
    label = None
    while node is not None:
        if getattr(node, "is_user_annotation", False):
            label = node.key   # deepest annotation wins
        node = getattr(node, "cpu_parent", None)
    return label or e.name

# Group by (nvtx_label, impl)
from collections import defaultdict
groups = defaultdict(list)
for e in kernel_events:
    if _t(e) == 0:
        continue
    label = _nvtx_label(e)
    impl  = "other"
    for prefix in ("eager", "sdpa", "triton"):
        if label.startswith(prefix + "/") or prefix.upper() in label:
            impl = prefix
            break
    groups[(impl, label)].append(e)

# ── Print table ───────────────────────────────────────────────────────────────
total = sum(_t(e) for e in kernel_events) or 1
print(f"\n{'Op (NVTX label) / Kernel':<55} {'CUDA us':>10} {'N':>4} {'us/call':>10}")
print("=" * 85)

for impl in ("eager", "sdpa", "triton", "other"):
    impl_keys = [(lbl, evts) for (imp, lbl), evts in groups.items() if imp == impl]
    if not impl_keys:
        continue
    impl_keys.sort(key=lambda x: sum(_t(e) for e in x[1]), reverse=True)
    impl_total = sum(_t(e) for evts in [v for _, v in impl_keys] for e in evts)
    print(f"\n── {impl.upper()}  ({impl_total/1000:.3f} ms total) ──")
    for lbl, evts in impl_keys:
        t_us   = sum(_t(e) for e in evts)
        n      = len(evts)
        pct    = t_us / total * 100
        # show individual kernel names under each NVTX label
        by_name = defaultdict(list)
        for e in evts:
            by_name[e.name].append(e)
        print(f"  [{lbl}]  {t_us:>10.1f} us  ({pct:.1f}%)")
        for kname, kevts in sorted(by_name.items(), key=lambda x: sum(_t(e) for e in x[1]), reverse=True):
            kt = sum(_t(e) for e in kevts)
            print(f"    {kname:<51} {kt:>8.1f} us  n={len(kevts)}"
                  f"  {kt/len(kevts):.1f} us/call")

print(f"\nTotal CUDA: {total/1000:.3f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# Chrome trace — open at chrome://tracing or https://ui.perfetto.dev
# NVTX ranges show as coloured bands; kernel launches appear inside them.
# ─────────────────────────────────────────────────────────────────────────────

prof.export_chrome_trace(TRACE)
print(f"\nChrome trace written to: {TRACE}")
print("Open with:  chrome://tracing  or  https://ui.perfetto.dev")
print("Each coloured band = one NVTX range (op label).")
print("Thin bars inside each band = individual CUDA kernel launches.")