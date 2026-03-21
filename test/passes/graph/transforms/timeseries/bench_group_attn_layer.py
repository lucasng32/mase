#!/usr/bin/env python3
"""
Layer-level benchmark: GroupSelfAttention vs FastGroupSelfAttention.

Profiles only the attention layer in isolation — no encoder, no embedding,
no loss. Gives the clearest picture of the kernel-level speedup.

Run::

    uv run python test/passes/graph/transforms/timeseries/bench_group_attn_layer.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.optimized_layers import (
    FastGroupSelfAttention,
    GroupPartition,
    KernelDispatcher,
)

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
    attn_implementation="sdpa",
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


def _peak_memory_mb(fn, args: tuple) -> float:
    if DEVICE.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    with torch.no_grad():
        fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / (1024**2)


def _make_group_time_mask(
    group_ids: torch.Tensor, T: int, dtype: torch.dtype
) -> torch.Tensor:
    B = group_ids.shape[0]
    group_mask = (group_ids[:, None] == group_ids[None, :]).float()
    time_mask = torch.ones(B, T, device=group_ids.device)
    gtm = torch.einsum("qb,bt->qbt", group_mask, time_mask)
    gtm = gtm.permute(2, 0, 1).unsqueeze(1)
    gtm = (1.0 - gtm) * torch.finfo(dtype).min
    return gtm.to(dtype)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@dataclass
class BenchCase:
    name: str
    B: int
    T: int
    group_ids: torch.Tensor
    description: str


def build_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []

    for B in [16, 32, 64, 128]:
        cases.append(BenchCase(f"univ B={B}", B, 64, torch.arange(B, dtype=torch.long), f"{B} independent series"))

    for B in [32, 64, 128]:
        cases.append(BenchCase(f"pairs B={B}", B, 64, torch.arange(B, dtype=torch.long) // 2, f"{B // 2} groups of 2"))

    for B in [32, 64, 128]:
        cases.append(BenchCase(f"quads B={B}", B, 64, torch.arange(B, dtype=torch.long) // 4, f"{B // 4} groups of 4"))

    for B in [32, 64, 128]:
        cases.append(BenchCase(f"oct B={B}", B, 64, torch.arange(B, dtype=torch.long) // 8, f"{B // 8} groups of 8"))

    for B in [64, 128]:
        cases.append(BenchCase(f"g16 B={B}", B, 64, torch.arange(B, dtype=torch.long) // 16, f"{B // 16} groups of 16"))

    for B in [32, 64]:
        cases.append(BenchCase(f"dense B={B}", B, 64, torch.zeros(B, dtype=torch.long), f"1 group of {B}"))

    mixed_26 = torch.tensor(
        [0, 1, 2, 3] + [4, 4, 5, 5, 6, 6] + [7, 7, 7, 7, 8, 8, 8, 8] + [9] * 8,
        dtype=torch.long,
    )
    cases.append(BenchCase("mixed B=26", 26, 64, mixed_26, "4×1 + 3×2 + 2×4 + 1×8"))

    mixed_56 = torch.tensor(
        list(range(8))
        + [i for i in range(8, 16) for _ in range(2)]
        + [i for i in range(16, 20) for _ in range(4)]
        + [i for i in range(20, 22) for _ in range(8)],
        dtype=torch.long,
    )
    cases.append(BenchCase("mixed B=56", 56, 64, mixed_56, "8×1 + 8×2 + 4×4 + 2×8"))

    for T in [16, 32, 64, 128]:
        cases.append(BenchCase(f"pairs T={T}", 64, T, torch.arange(64, dtype=torch.long) // 2, f"32 pairs, T={T}"))

    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device : {DEVICE}")
    print(f"Config : d_model={CFG.d_model}  n_heads={CFG.num_heads}  d_kv={CFG.d_kv}")
    print(f"Warmup : {WARMUP}   Iters: {ITERS}")
    print(f"Layer  : GroupSelfAttention only (no encoder, no embedding)")
    print()

    baseline_mod = GroupSelfAttention(CFG).to(DEVICE).eval()
    baseline_sd = baseline_mod.state_dict()

    cases = build_cases()
    has_cuda = DEVICE.type == "cuda"

    sep = "─"
    col_case = 22
    col_desc = 24
    col_ms = 12
    col_sp = 9
    col_mem = 10
    col_var = 14

    hdr_parts = [
        f"{'Case':<{col_case}}",
        f"{'Description':<{col_desc}}",
        f"{'Baseline (ms)':>{col_ms}}",
        f"{'Ours (ms)':>{col_ms}}",
        f"{'Variant':>{col_var}}",
        f"{'Speedup':>{col_sp}}",
    ]
    if has_cuda:
        hdr_parts += [f"{'Mem base':>{col_mem}}", f"{'Mem ours':>{col_mem}}"]

    hdr = "  ".join(hdr_parts)
    width = len(hdr)

    print(sep * width)
    print(hdr)
    print(sep * width)

    for case in cases:
        group_ids = case.group_ids.to(DEVICE)
        partition = GroupPartition.from_group_ids(group_ids)
        variant = KernelDispatcher.select(partition, DEVICE)

        hs = torch.randn(case.B, case.T, CFG.d_model, device=DEVICE, dtype=DTYPE)
        mask = _make_group_time_mask(group_ids, case.T, DTYPE).to(DEVICE)

        fast = FastGroupSelfAttention(CFG, partition, variant=variant).to(DEVICE).eval()
        fast.load_state_dict(baseline_sd, strict=False)

        base_ms = _bench(baseline_mod, (hs, mask))
        fast_ms = _bench(fast, (hs, mask))
        speedup = base_ms / fast_ms

        row_parts = [
            f"{case.name:<{col_case}}",
            f"{case.description:<{col_desc}}",
            f"{base_ms:>{col_ms}.3f}",
            f"{fast_ms:>{col_ms}.3f}",
            f"{variant.name.lower():>{col_var}}",
            f"{speedup:>{col_sp - 1}.2f}x",
        ]

        if has_cuda:
            mem_base = _peak_memory_mb(baseline_mod, (hs, mask))
            mem_fast = _peak_memory_mb(fast, (hs, mask))
            row_parts += [
                f"{mem_base:>{col_mem - 2}.1f}MB",
                f"{mem_fast:>{col_mem - 2}.1f}MB",
            ]

        print("  ".join(row_parts))

    print(sep * width)
    print()


if __name__ == "__main__":
    main()
