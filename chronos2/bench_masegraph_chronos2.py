#!/usr/bin/env python3
"""
End-to-end Chronos2Encoder benchmark via MaseGraph transform pass.

Applies ``fast_group_attention_transform_pass`` through the MaseGraph API
(the same path used in production), validates that outputs are numerically
identical to the unmodified baseline, and reports inference speedup.

MaseGraph tracing note
----------------------
``Chronos2Encoder.forward`` calls ``torch.finfo(inputs_embeds.dtype)`` which
fails with symbolic FX proxies.  We work around this by wrapping the encoder
in ``TracableEncoder`` — a thin shell that accepts pre-computed masks as
explicit tensor inputs so that no dtype inspection occurs inside the traced
graph.  The ``precompute_masks`` helper runs outside the graph and produces
the same masks that the full encoder would normally construct internally.

Run::

    uv run python test/passes/graph/transforms/timeseries/bench_masegraph_chronos2.py
"""

from __future__ import annotations

import copy
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from chop.ir import MaseGraph
from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import (
    Chronos2LayerNorm,
    FeedForward,
    GroupSelfAttention,
    TimeSelfAttention,
)
from chop.models.chronos2.modeling_chronos2 import Chronos2Encoder
from chop.passes.graph.transforms.timeseries import fast_group_attention_transform_pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
WARMUP = 10
ITERS  = 50

CFG = Chronos2CoreConfig(
    d_model=512,
    d_kv=64,
    d_ff=2048,
    num_layers=6,
    num_heads=8,
    dropout_rate=0.0,
    attn_implementation="eager",
)

# context_length=512, patch_size=16 → 32 context patches + 1 future patch
SEQ_LEN = 33

# Leaf modules for FX — their forward() is opaque during tracing.
CUSTOM_OPS = {
    "modules": {
        GroupSelfAttention: {},
        TimeSelfAttention: {},
        FeedForward: {},
        Chronos2LayerNorm: {},
    },
    "functions": {},
}


# ---------------------------------------------------------------------------
# TracableEncoder wrapper
# ---------------------------------------------------------------------------
class TracableEncoder(nn.Module):
    """Encoder shell with pre-computed masks as explicit inputs.

    ``Chronos2Encoder.forward`` calls ``torch.finfo(inputs_embeds.dtype)``
    which FX cannot handle with symbolic proxies.  This wrapper moves the
    two mask computations outside the traced region.  The inner block loop
    is identical to the original encoder.
    """

    def __init__(self, encoder: Chronos2Encoder) -> None:
        super().__init__()
        self.dropout          = encoder.dropout
        self.block            = encoder.block
        self.final_layer_norm = encoder.final_layer_norm

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        extended_attn_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        h = self.dropout(inputs_embeds)
        for layer in self.block:
            out = layer(
                h,
                position_ids=position_ids,
                attention_mask=extended_attn_mask,
                group_time_mask=group_time_mask,
                output_attentions=False,
            )
            h = out[0]
        h = self.final_layer_norm(h)
        return self.dropout(h)


def precompute_masks(
    group_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (extended_attn_mask, group_time_mask, position_ids).

    Reproduces the two mask computations from ``Chronos2Encoder.forward``
    so they can be supplied as concrete tensors to ``TracableEncoder``.
    """
    fmin = torch.finfo(dtype).min

    # extended time mask  (B, 1, 1, T)  — all-ones attn_mask → all-zeros mask
    ext = attn_mask[:, None, None, :].to(dtype)
    ext = (1.0 - ext) * fmin

    # group × time mask  (T, 1, B, B)
    gm  = (group_ids[:, None] == group_ids[None, :]).float().to(device)
    tm  = torch.einsum("qb,bt->qbt", gm, attn_mask.float())
    gtm = tm.permute(2, 0, 1).unsqueeze(1)
    gtm = ((1.0 - gtm) * fmin).to(dtype)

    # position ids  (1, T)
    T   = attn_mask.shape[1]
    pos = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)

    return ext.to(device), gtm, pos


# ---------------------------------------------------------------------------
# MaseGraph construction
# ---------------------------------------------------------------------------
def _build_mg(encoder: Chronos2Encoder) -> MaseGraph:
    """Trace a fresh copy of the encoder into a MaseGraph."""
    return MaseGraph(
        TracableEncoder(copy.deepcopy(encoder)).cpu(),
        hf_input_names=[
            "inputs_embeds",
            "extended_attn_mask",
            "group_time_mask",
            "position_ids",
        ],
        custom_ops=CUSTOM_OPS,
    )


def _apply_pass(mg: MaseGraph, group_ids: torch.Tensor) -> tuple[MaseGraph, dict]:
    return fast_group_attention_transform_pass(
        mg, pass_args={"group_ids": group_ids.cpu()}
    )


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def _bench(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return median latency in milliseconds."""
    use_cuda = DEVICE.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if use_cuda:
            torch.cuda.synchronize()

    times: list[float] = []
    with torch.no_grad():
        for _ in range(iters):
            if use_cuda:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


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

    for B in [64, 128, 256]:
        cases.append(BenchCase(
            f"univariate B={B}", f"{B} independent series",
            torch.arange(B, dtype=torch.long),
        ))

    for B in [64, 128, 256]:
        cases.append(BenchCase(
            f"pairs B={B}", f"{B // 2} groups of 2",
            torch.arange(B, dtype=torch.long) // 2,
        ))

    for B in [64, 128, 256]:
        cases.append(BenchCase(
            f"quads B={B}", f"{B // 4} groups of 4",
            torch.arange(B, dtype=torch.long) // 4,
        ))

    for scale in [1, 2]:
        gids = torch.tensor(
            list(range(20 * scale))
            + [i for i in range(20 * scale, 60 * scale) for _ in range(2)]
            + [i for i in range(60 * scale, 75 * scale) for _ in range(4)]
            + [i for i in range(75 * scale, 77 * scale) for _ in range(8)],
            dtype=torch.long,
        )
        cases.append(BenchCase(
            f"realistic B={len(gids)}",
            f"{20*scale}×1 + {40*scale}×2 + {15*scale}×4 + {2*scale}×8",
            gids,
        ))

    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\nDevice  : {DEVICE}")
    print(f"Config  : d_model={CFG.d_model}  n_heads={CFG.num_heads}  "
          f"d_kv={CFG.d_kv}  num_layers={CFG.num_layers}")
    print(f"Seq len : {SEQ_LEN} patches")
    print(f"Warmup  : {WARMUP}   Iters: {ITERS}")
    print()
    print("Optimized = fast_group_attention_transform_pass applied via MaseGraph.")
    print("Correct   = output matches baseline to atol=1e-4 on a fresh random batch.")

    torch.manual_seed(0)
    baseline = Chronos2Encoder(CFG).to(DEVICE).eval()
    te_baseline = TracableEncoder(baseline).to(DEVICE).eval()

    cases = build_cases()

    sep    = "─"
    W_case = 22
    W_desc = 34
    W_ok   = 9
    W_ms   = 11
    W_sp   = 10

    headers = [
        f"{'Case':<{W_case}}",
        f"{'Description':<{W_desc}}",
        f"{'Correct':>{W_ok}}",
        f"{'Baseline':>{W_ms}}",
        f"{'Optimized':>{W_ms}}",
        f"{'Speedup':>{W_sp}}",
    ]
    hdr   = "  ".join(headers)
    width = len(hdr)

    print()
    print(sep * width)
    print(hdr)
    print(sep * width)

    for case in cases:
        gids      = case.group_ids.to(DEVICE)
        B         = gids.shape[0]
        embeds    = torch.randn(B, SEQ_LEN, CFG.d_model, device=DEVICE, dtype=DTYPE)
        attn_mask = torch.ones(B, SEQ_LEN, device=DEVICE, dtype=DTYPE)

        ext, gtm, pos = precompute_masks(gids, attn_mask, DTYPE, DEVICE)

        # ── Build MaseGraph and apply pass ───────────────────────────────
        mg = _build_mg(baseline)
        mg, info = _apply_pass(mg, gids)
        opt_model = mg.model.to(DEVICE).eval()

        # ── Correctness (fresh random batch) ────────────────────────────
        torch.manual_seed(1337)
        embeds_check = torch.randn(B, SEQ_LEN, CFG.d_model, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            ref = te_baseline(embeds_check, ext, gtm, pos)
            opt = opt_model(embeds_check, ext, gtm, pos)
        max_diff = (ref - opt).abs().max().item()
        correct  = torch.allclose(ref, opt, atol=1e-4)

        # ── Speed ────────────────────────────────────────────────────────
        base_ms = _bench(lambda: te_baseline(embeds, ext, gtm, pos))
        opt_ms  = _bench(lambda: opt_model(embeds, ext, gtm, pos))

        row = [
            f"{case.name:<{W_case}}",
            f"{case.description:<{W_desc}}",
            f"{'✓' if correct else f'✗ {max_diff:.1e}':>{W_ok}}",
            f"{base_ms:>{W_ms}.3f}",
            f"{opt_ms:>{W_ms}.3f}",
            f"{base_ms / opt_ms:>{W_sp-1}.2f}x",
        ]
        print("  ".join(row))

    print(sep * width)
    print()
    print(f"All times in milliseconds (median over {ITERS} iterations).")
    print("Speedup > 1.0x means optimized is faster than baseline.")
    print()


if __name__ == "__main__":
    main()
