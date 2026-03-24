#!/usr/bin/env python3
"""
Benchmark: GroupSelfAttention baseline vs fast_group_attention_transform_pass.

The pass auto-selects the kernel via KernelDispatcher:
  - TRITON_BUCKETED on CUDA (one Triton launch per power-of-2 bucket)
  - PACKED_SPARSE    on CPU
  - UNIVARIATE       when all groups are size 1 (no Q/K projected)

Run::

    uv run python test/passes/graph/transforms/timeseries/bench_bucketed_attn.py
"""

from __future__ import annotations

import copy
import statistics
import time
from dataclasses import dataclass

import torch

from chop.ir import MaseGraph
from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention, TimeSelfAttention
from chop.models.chronos2.modeling_chronos2 import Chronos2Model
from chop.models.chronos2.triton_grouped_attn import is_triton_available
from chop.passes.graph.transforms.timeseries import fast_group_attention_transform_pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
WARMUP = 20
ITERS = 100

CFG = Chronos2CoreConfig(
    d_model=512,
    d_kv=64,
    d_ff=2048,
    num_layers=1,
    num_heads=8,
    dropout_rate=0.0,
    attn_implementation="sdpa",
    chronos_config={
        "context_length": 512,
        "input_patch_size": 16,
        "input_patch_stride": 16,
        "output_patch_size": 16,
        "quantiles": [0.5],
        "use_reg_token": False,
        "use_arcsinh": False,
    },
)

_CUSTOM_OPS = {
    "modules": {GroupSelfAttention: {}, TimeSelfAttention: {}},
    "functions": {},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bench(fn, args: tuple, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return median latency in milliseconds."""
    use_cuda = DEVICE.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        if use_cuda:
            torch.cuda.synchronize()

    times: list[float] = []
    with torch.no_grad():
        for _ in range(iters):
            if use_cuda:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn(*args)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                fn(*args)
                times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _make_mask(group_ids: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
    B = group_ids.shape[0]
    group_mask = (group_ids[:, None] == group_ids[None, :]).float()
    time_mask = torch.ones(B, T, device=group_ids.device)
    gtm = torch.einsum("qb,bt->qbt", group_mask, time_mask)
    gtm = gtm.permute(2, 0, 1).unsqueeze(1)
    return ((1.0 - gtm) * torch.finfo(dtype).min).to(dtype)


def _apply_pass(group_ids: torch.Tensor) -> GroupSelfAttention:
    """Apply fast_group_attention_transform_pass via MaseGraph.

    Returns the replaced GroupSelfAttention extracted from the traced model,
    ready to benchmark with (hs, mask) inputs.
    """
    model = Chronos2Model(CFG).to(DEVICE).eval()
    mg = MaseGraph(copy.deepcopy(model), hf_input_names=["context", "group_ids"], custom_ops=_CUSTOM_OPS)
    mg, _ = fast_group_attention_transform_pass(mg, pass_args={"group_ids": group_ids.cpu()})
    for m in mg.model.modules():
        if isinstance(m, GroupSelfAttention):
            return m.to(DEVICE).eval()
    raise RuntimeError("No GroupSelfAttention found after pass")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@dataclass
class BenchCase:
    name: str
    description: str
    T: int
    group_ids: torch.Tensor


def build_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []

    # ── Uniform batches ──
    for B, g in [(128, 2), (256, 2), (128, 4), (256, 4), (128, 8), (256, 8)]:
        gids = torch.arange(B, dtype=torch.long) // g
        cases.append(BenchCase(f"uniform g={g} B={B}", f"{B//g} groups of {g}", 64, gids))

    # ── Mixed batches (multiple buckets) ──
    for B_pairs, B_quads in [(64, 64), (128, 128)]:
        gids = torch.cat([
            torch.arange(B_pairs, dtype=torch.long) // 2,
            torch.arange(B_pairs // 2, B_pairs // 2 + B_quads // 4, dtype=torch.long).repeat_interleave(4),
        ])
        cases.append(BenchCase(f"mix2+4 B={len(gids)}", f"{B_pairs//2}×2 + {B_quads//4}×4", 64, gids))

    for scale in [1, 2]:
        gids = torch.tensor(
            [i for i in range(20 * scale) for _ in range(2)]
            + [i for i in range(20 * scale, 30 * scale) for _ in range(4)]
            + [i for i in range(30 * scale, 34 * scale) for _ in range(8)],
            dtype=torch.long,
        )
        cases.append(BenchCase(
            f"mix2+4+8 B={len(gids)}", f"{20*scale}×2 + {10*scale}×4 + {4*scale}×8", 64, gids,
        ))

    # ── Realistic Chronos2 ──
    for scale in [1, 2]:
        gids = torch.tensor(
            list(range(50 * scale))
            + [i for i in range(50 * scale, 100 * scale) for _ in range(2)]
            + [i for i in range(100 * scale, 115 * scale) for _ in range(4)]
            + [i for i in range(115 * scale, 117 * scale) for _ in range(8)],
            dtype=torch.long,
        )
        cases.append(BenchCase(
            f"realistic B={len(gids)}",
            f"{50*scale}×1 + {50*scale}×2 + {15*scale}×4 + {2*scale}×8",
            64, gids,
        ))

    # ── T sweep ──
    for T in [16, 32, 64, 128]:
        gids = torch.tensor(
            [i for i in range(20) for _ in range(2)]
            + [i for i in range(20, 30) for _ in range(4)]
            + [i for i in range(30, 34) for _ in range(8)],
            dtype=torch.long,
        )
        cases.append(BenchCase(f"mix2+4+8 T={T}", f"3-bucket mix, T={T}", T, gids))

    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if DEVICE.type != "cuda":
        print("No CUDA device — Triton unavailable.")
        return
    if not is_triton_available():
        print("Triton not available.")
        return

    print(f"\nDevice : {DEVICE}")
    print(f"Config : d_model={CFG.d_model}  n_heads={CFG.num_heads}  d_kv={CFG.d_kv}")
    print(f"Warmup : {WARMUP}   Iters: {ITERS}")

    baseline_mod = GroupSelfAttention(CFG).to(DEVICE).eval()
    cases = build_cases()

    sep = "─"
    W_case, W_desc, W_ms, W_var, W_sp = 22, 30, 11, 16, 10

    headers = [
        f"{'Case':<{W_case}}",
        f"{'Description':<{W_desc}}",
        f"{'Baseline':>{W_ms}}",
        f"{'Optimized':>{W_ms}}",
        f"{'Variant':>{W_var}}",
        f"{'Speedup':>{W_sp}}",
    ]
    hdr = "  ".join(headers)
    width = len(hdr)

    print()
    print(sep * width)
    print(hdr)
    print(sep * width)

    for case in cases:
        group_ids = case.group_ids.to(DEVICE)
        B = group_ids.shape[0]
        hs   = torch.randn(B, case.T, CFG.d_model, device=DEVICE, dtype=DTYPE)
        mask = _make_mask(group_ids, case.T, DTYPE)

        optimized = _apply_pass(group_ids)
        variant_name = type(optimized.self_attention).__name__

        base_ms = _bench(baseline_mod, (hs, mask))
        opt_ms  = _bench(optimized,    (hs, mask))

        print("  ".join([
            f"{case.name:<{W_case}}",
            f"{case.description:<{W_desc}}",
            f"{base_ms:>{W_ms}.3f}",
            f"{opt_ms:>{W_ms}.3f}",
            f"{variant_name:>{W_var}}",
            f"{base_ms / opt_ms:>{W_sp-1}.2f}x",
        ]))

    print(sep * width)
    print()
    print("All times in milliseconds (median over", ITERS, "iterations).")
    print("Speedup = baseline_ms / optimized_ms  (>1 means optimized is faster)")
    print()

    # ── Univariate section ──────────────────────────────────────────────────
    print("── Univariate (all groups size 1) ──")
    print()

    W_ucase, W_ums, W_usp = 18, 11, 13

    u_headers = [
        f"{'Case':<{W_ucase}}",
        f"{'Baseline':>{W_ums}}",
        f"{'Univariate':>{W_ums}}",
        f"{'Speedup':>{W_usp}}",
    ]
    u_hdr = "  ".join(u_headers)
    u_width = len(u_hdr)

    print(sep * u_width)
    print(u_hdr)
    print(sep * u_width)

    for B in [64, 128, 256, 512]:
        group_ids = torch.arange(B, dtype=torch.long).to(DEVICE)
        hs   = torch.randn(B, 64, CFG.d_model, device=DEVICE, dtype=DTYPE)
        mask = _make_mask(group_ids, 64, DTYPE)

        univ_mod = _apply_pass(group_ids)

        base_ms = _bench(baseline_mod, (hs, mask))
        univ_ms = _bench(univ_mod,     (hs, mask))

        print("  ".join([
            f"{'univ B='+str(B):<{W_ucase}}",
            f"{base_ms:>{W_ums}.3f}",
            f"{univ_ms:>{W_ums}.3f}",
            f"{base_ms / univ_ms:>{W_usp-1}.2f}x",
        ]))

    print(sep * u_width)
    print()
    print("Univariate = UnivariateGroupAwareMHA (no Q/K projected, decided at pass time)")
    print()


if __name__ == "__main__":
    main()
