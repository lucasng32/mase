#!/usr/bin/env python3
"""
End-to-end benchmark: Chronos2Encoder with GroupAwareMHA (TRITON_BUCKETED)
vs the unmodified baseline and the old single-launch TRITON variant.

The full encoder stack is timed — all num_layers encoder blocks, each
containing TimeSelfAttention, GroupSelfAttention, and FFN.  Inputs are
synthetic but match the shapes produced by the patching + embedding layers
upstream.

Run::

    uv run python test/passes/graph/transforms/timeseries/bench_e2e_chronos2.py
"""

from __future__ import annotations

import copy
import statistics
import time
from dataclasses import dataclass

import torch

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.modeling_chronos2 import Chronos2Encoder
from chop.models.chronos2.optimized_layers import (
    GroupAwareMHA,
    GroupPartition,
    KernelVariant,
)
from chop.models.chronos2.triton_grouped_attn import is_triton_available

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
WARMUP = 10
ITERS  = 50

# Realistic Chronos2 encoder config (matches the small pretrained variant)
CFG = Chronos2CoreConfig(
    d_model=512,
    d_kv=64,
    d_ff=2048,
    num_layers=6,
    num_heads=8,
    dropout_rate=0.0,
    attn_implementation="eager",
)

# Sequence length fed into the encoder.
# context_length=512, patch_size=16 → 32 context patches + 1 future patch = 33
SEQ_LEN = 33


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


def _apply_transform(encoder: Chronos2Encoder, group_ids: torch.Tensor, variant: KernelVariant) -> Chronos2Encoder:
    """Deep-copy the encoder and swap every GroupSelfAttention inner MHA."""
    enc = copy.deepcopy(encoder)
    partition = GroupPartition.from_group_ids(group_ids.cpu())
    for module in enc.modules():
        if not isinstance(module, GroupSelfAttention):
            continue
        if isinstance(module.self_attention, GroupAwareMHA):
            continue
        device = next(module.parameters()).device
        mha = GroupAwareMHA(module.self_attention.config, partition, variant=variant)
        mha.load_state_dict(module.self_attention.state_dict())
        mha.to(device)
        module.self_attention = mha
    return enc


def _make_inputs(B: int, seq_len: int, group_ids: torch.Tensor):
    """Synthetic encoder inputs matching shapes from the embedding layer."""
    embeds = torch.randn(B, seq_len, CFG.d_model, device=DEVICE, dtype=DTYPE)
    attn_mask = torch.ones(B, seq_len, device=DEVICE, dtype=DTYPE)
    gids = group_ids.to(DEVICE)
    return embeds, attn_mask, gids


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------
@dataclass
class BenchCase:
    name: str
    description: str
    group_ids: torch.Tensor


def build_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []

    # Univariate — all independent
    for B in [64, 128, 256]:
        cases.append(BenchCase(
            f"univariate B={B}",
            f"{B} independent series",
            torch.arange(B, dtype=torch.long),
        ))

    # Uniform pairs
    for B in [64, 128, 256]:
        cases.append(BenchCase(
            f"pairs B={B}",
            f"{B//2} groups of 2",
            torch.arange(B, dtype=torch.long) // 2,
        ))

    # Uniform quads
    for B in [64, 128, 256]:
        cases.append(BenchCase(
            f"quads B={B}",
            f"{B//4} groups of 4",
            torch.arange(B, dtype=torch.long) // 4,
        ))

    # Realistic mixed: univariate + pairs + quads + octets
    for scale in [1, 2]:
        gids = torch.tensor(
            list(range(20 * scale))
            + [i for i in range(20 * scale, 60 * scale) for _ in range(2)]
            + [i for i in range(60 * scale, 75 * scale) for _ in range(4)]
            + [i for i in range(75 * scale, 77 * scale) for _ in range(8)],
            dtype=torch.long,
        )
        B = len(gids)
        cases.append(BenchCase(
            f"realistic B={B}",
            f"{20*scale}×1 + {40*scale}×2 + {15*scale}×4 + {2*scale}×8",
            gids,
        ))

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

    print(f"\nDevice  : {DEVICE}")
    print(f"Config  : d_model={CFG.d_model}  n_heads={CFG.num_heads}  "
          f"d_kv={CFG.d_kv}  num_layers={CFG.num_layers}")
    print(f"Seq len : {SEQ_LEN} patches  (context_length=512, patch_size=16, +1 future patch)")
    print(f"Warmup  : {WARMUP}   Iters: {ITERS}")

    # One shared baseline encoder (random init — no pretrained weights needed)
    torch.manual_seed(0)
    baseline = Chronos2Encoder(CFG).to(DEVICE).eval()

    cases  = build_cases()
    sep    = "─"
    W_case = 22
    W_desc = 34
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
    hdr   = "  ".join(headers)
    width = len(hdr)

    print()
    print(sep * width)
    print(hdr)
    print(sep * width)

    for case in cases:
        B        = case.group_ids.shape[0]
        embeds, attn_mask, gids = _make_inputs(B, SEQ_LEN, case.group_ids)

        triton_enc   = _apply_transform(baseline, case.group_ids, KernelVariant.TRITON)
        bucketed_enc = _apply_transform(baseline, case.group_ids, KernelVariant.TRITON_BUCKETED)

        def _run_baseline():   baseline(embeds, group_ids=gids, attention_mask=attn_mask)
        def _run_triton():     triton_enc(embeds, group_ids=gids, attention_mask=attn_mask)
        def _run_bucketed():   bucketed_enc(embeds, group_ids=gids, attention_mask=attn_mask)

        base_ms     = _bench(_run_baseline,  ())
        triton_ms   = _bench(_run_triton,    ())
        bucketed_ms = _bench(_run_bucketed,  ())

        row = [
            f"{case.name:<{W_case}}",
            f"{case.description:<{W_desc}}",
            f"{base_ms:>{W_ms}.3f}",
            f"{triton_ms:>{W_ms}.3f}",
            f"{bucketed_ms:>{W_ms}.3f}",
            f"{triton_ms   / bucketed_ms:>{W_sp-1}.2f}x",
            f"{base_ms     / bucketed_ms:>{W_sp-1}.2f}x",
        ]
        print("  ".join(row))

    print(sep * width)
    print()
    print("All times in milliseconds (median over", ITERS, "iterations).")
    print("Triton    = old single-launch kernel (BLOCK_M=32, global max_group_size)")
    print("Bucketed  = new per-bucket kernel (BLOCK_M matched to bucket size)")
    print("vs Triton    = triton_ms   / bucketed_ms  (>1 means bucketed is faster)")
    print("vs Baseline  = baseline_ms / bucketed_ms  (>1 means bucketed is faster)")
    print()


if __name__ == "__main__":
    main()
