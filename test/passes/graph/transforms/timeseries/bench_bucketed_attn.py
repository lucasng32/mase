#!/usr/bin/env python3
"""
Benchmark: TRITON_BUCKETED vs TRITON (single-launch) vs PACKED_SPARSE.

Focuses on the cases where bucketed dispatch is expected to win:
  - Mixed group sizes (multiple occupied buckets)
  - Small groups in a large batch (tile waste is worst)

Run::

    uv run python test/passes/graph/transforms/timeseries/bench_bucketed_attn.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.optimized_layers import (
    GroupAwareMHA,
    GroupPartition,
    KernelVariant,
    UnivariateGroupAwareMHA,
)
from chop.models.chronos2.triton_grouped_attn import is_triton_available

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
    num_layers=6,
    num_heads=8,
    dropout_rate=0.0,
    attn_implementation="eager",
)


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


def _make_univariate_gsa(baseline: GroupSelfAttention) -> GroupSelfAttention:
    """Return GroupSelfAttention with inner MHA swapped for UnivariateGroupAwareMHA."""
    opt = GroupSelfAttention(CFG).to(DEVICE).eval()
    opt.load_state_dict(baseline.state_dict())
    mha_sd = baseline.self_attention.state_dict()
    univ = UnivariateGroupAwareMHA(CFG).to(DEVICE)
    univ.load_state_dict({k: v for k, v in mha_sd.items() if k in ("v.weight", "o.weight")})
    opt.self_attention = univ
    return opt


def _make_optimized_gsa(
    baseline: GroupSelfAttention,
    group_ids: torch.Tensor,
    variant: KernelVariant,
) -> GroupSelfAttention:
    """Return GroupSelfAttention with inner MHA swapped for GroupAwareMHA."""
    partition = GroupPartition.from_group_ids(group_ids)
    opt = GroupSelfAttention(CFG).to(DEVICE).eval()
    opt.load_state_dict(baseline.state_dict())
    mha = GroupAwareMHA(CFG, partition, variant=variant).to(DEVICE)
    mha.load_state_dict(baseline.self_attention.state_dict())
    opt.self_attention = mha
    return opt


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

    # ── Uniform batches (1 bucket occupied — baseline for overhead check) ──
    for B, g in [(128, 2), (256, 2), (128, 4), (256, 4), (128, 8), (256, 8)]:
        gids = torch.arange(B, dtype=torch.long) // g
        cases.append(BenchCase(
            f"uniform g={g} B={B}", f"{B//g} groups of {g}", 64, gids,
        ))

    # ── Mixed batches (multiple buckets — where bucketed should win) ──
    # 2-bucket mix: pairs + quads
    for B_pairs, B_quads in [(64, 64), (128, 128)]:
        gids = torch.cat([
            torch.arange(B_pairs, dtype=torch.long) // 2,
            torch.arange(B_pairs // 2, B_pairs // 2 + B_quads // 4, dtype=torch.long).repeat_interleave(4),
        ])
        cases.append(BenchCase(
            f"mix2+4 B={len(gids)}",
            f"{B_pairs//2}×2 + {B_quads//4}×4",
            64, gids,
        ))

    # 3-bucket mix: pairs + quads + octets
    for scale in [1, 2]:
        gids = torch.tensor(
            [i for i in range(20 * scale) for _ in range(2)]
            + [i for i in range(20 * scale, 30 * scale) for _ in range(4)]
            + [i for i in range(30 * scale, 34 * scale) for _ in range(8)],
            dtype=torch.long,
        )
        B = len(gids)
        cases.append(BenchCase(
            f"mix2+4+8 B={B}",
            f"{20*scale}×2 + {10*scale}×4 + {4*scale}×8",
            64, gids,
        ))

    # Realistic Chronos2: lots of small groups, few large
    for scale in [1, 2]:
        gids = torch.tensor(
            list(range(50 * scale))                                    # 50s univariate
            + [i for i in range(50 * scale, 100 * scale) for _ in range(2)]   # 50s pairs
            + [i for i in range(100 * scale, 115 * scale) for _ in range(4)]  # 15s quads
            + [i for i in range(115 * scale, 117 * scale) for _ in range(8)], # 2s octets
            dtype=torch.long,
        )
        B = len(gids)
        cases.append(BenchCase(
            f"realistic B={B}",
            f"{50*scale}×1 + {50*scale}×2 + {15*scale}×4 + {2*scale}×8",
            64, gids,
        ))

    # T sweep on a 3-bucket mix
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

    # ── Table layout ────────────────────────────────────────────────────────
    sep = "─"
    W_case = 22
    W_desc = 30
    W_ms   = 11
    W_sp   = 10

    headers = [
        f"{'Case':<{W_case}}",
        f"{'Description':<{W_desc}}",
        f"{'Baseline':>{W_ms}}",
        f"{'Triton':>{W_ms}}",
        f"{'Bucketed':>{W_ms}}",
        f"{'vs Triton':>{W_sp}}",
        f"{'vs Baseline':>{W_sp}}",
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

        triton_mod  = _make_optimized_gsa(baseline_mod, group_ids, KernelVariant.TRITON)
        bucketed    = _make_optimized_gsa(baseline_mod, group_ids, KernelVariant.TRITON_BUCKETED)

        base_ms     = _bench(baseline_mod, (hs, mask))
        triton_ms   = _bench(triton_mod,   (hs, mask))
        bucketed_ms = _bench(bucketed,     (hs, mask))

        row = [
            f"{case.name:<{W_case}}",
            f"{case.description:<{W_desc}}",
            f"{base_ms:>{W_ms}.3f}",
            f"{triton_ms:>{W_ms}.3f}",
            f"{bucketed_ms:>{W_ms}.3f}",
            f"{triton_ms / bucketed_ms:>{W_sp-1}.2f}x",
            f"{base_ms   / bucketed_ms:>{W_sp-1}.2f}x",
        ]
        print("  ".join(row))

    print(sep * width)
    print()
    print("All times in milliseconds (median over", ITERS, "iterations).")
    print("vs Triton    = triton_ms   / bucketed_ms  (>1 means bucketed is faster)")
    print("vs Baseline  = baseline_ms / bucketed_ms  (>1 means bucketed is faster)")
    print()

    # ── Univariate section ──────────────────────────────────────────────────
    print("── Univariate (all groups size 1): Baseline vs Triton vs Bucketed vs Univariate ──")
    print()

    W_ucase = 18
    W_ums   = 11
    W_usp   = 13

    u_headers = [
        f"{'Case':<{W_ucase}}",
        f"{'Baseline':>{W_ums}}",
        f"{'Triton':>{W_ums}}",
        f"{'Bucketed':>{W_ums}}",
        f"{'Univariate':>{W_ums}}",
        f"{'vs Triton':>{W_usp}}",
        f"{'vs Bucketed':>{W_usp}}",
        f"{'vs Baseline':>{W_usp}}",
    ]
    u_hdr   = "  ".join(u_headers)
    u_width = len(u_hdr)

    print(sep * u_width)
    print(u_hdr)
    print(sep * u_width)

    for B in [64, 128, 256, 512]:
        group_ids = torch.arange(B, dtype=torch.long).to(DEVICE)
        hs   = torch.randn(B, 64, CFG.d_model, device=DEVICE, dtype=DTYPE)
        mask = _make_mask(group_ids, 64, DTYPE)

        triton_mod   = _make_optimized_gsa(baseline_mod, group_ids, KernelVariant.TRITON)
        bucketed_mod = _make_optimized_gsa(baseline_mod, group_ids, KernelVariant.TRITON_BUCKETED)
        univ_mod     = _make_univariate_gsa(baseline_mod)

        base_ms  = _bench(baseline_mod, (hs, mask))
        tri_ms   = _bench(triton_mod,   (hs, mask))
        buck_ms  = _bench(bucketed_mod, (hs, mask))
        univ_ms  = _bench(univ_mod,     (hs, mask))

        row = [
            f"{'univ B='+str(B):<{W_ucase}}",
            f"{base_ms:>{W_ums}.3f}",
            f"{tri_ms:>{W_ums}.3f}",
            f"{buck_ms:>{W_ums}.3f}",
            f"{univ_ms:>{W_ums}.3f}",
            f"{tri_ms  / univ_ms:>{W_usp-1}.2f}x",
            f"{buck_ms / univ_ms:>{W_usp-1}.2f}x",
            f"{base_ms / univ_ms:>{W_usp-1}.2f}x",
        ]
        print("  ".join(row))

    print(sep * u_width)
    print()
    print("Univariate = UnivariateGroupAwareMHA (pass-time decision, no Q/K allocated)")
    print("vs Triton    = triton_ms   / univ_ms  (>1 means univariate is faster)")
    print("vs Bucketed  = bucketed_ms / univ_ms  (>1 means univariate is faster)")
    print("vs Baseline  = baseline_ms / univ_ms  (>1 means univariate is faster)")
    print()


if __name__ == "__main__":
    main()
