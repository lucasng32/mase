#!/usr/bin/env python3
"""
Group attention benchmark — three sections in one run:

  1. Layer-level  : isolated GroupSelfAttention
                    Baseline vs Triton (old) vs Bucketed (new)
                    across uniform, mixed, and realistic group distributions.

  2. Univariate   : all-univariate GroupSelfAttention
                    Baseline vs Triton vs Bucketed vs Univariate (new dedicated module)

  3. End-to-end   : full Chronos2Encoder (all 6 layers: TimeSelfAttention +
                    GroupSelfAttention + FFN per block)
                    Baseline vs Triton vs Bucketed

Run::

    uv run python test/passes/graph/transforms/timeseries/bench_group_attention.py

Requires CUDA + Triton.  All inputs are synthetic — no pretrained weights needed.

Columns
-------
Baseline   : unmodified GroupSelfAttention (full B×B mask)
Triton     : GroupAwareMHA with KernelVariant.TRITON  (single launch, BLOCK_M=32 globally)
Bucketed   : GroupAwareMHA with KernelVariant.TRITON_BUCKETED  (per-bucket launch)
Univariate : UnivariateGroupAwareMHA  (pass-time decision, no Q/K allocated)

vs X = X_ms / target_ms  —  values > 1 mean the target column is faster than X.
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
    UnivariateGroupAwareMHA,
)
from chop.models.chronos2.triton_grouped_attn import is_triton_available

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

# Layer-level benchmark settings
LAYER_WARMUP = 20
LAYER_ITERS  = 100

# End-to-end benchmark settings
E2E_WARMUP = 10
E2E_ITERS  = 50

# Encoder sequence length: context_length=512, patch_size=16 → 32 patches + 1 future
E2E_SEQ_LEN = 33

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
# Timing helper
# ---------------------------------------------------------------------------
def _bench(fn, args: tuple, warmup: int, iters: int) -> float:
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


# ---------------------------------------------------------------------------
# Layer-level module builders
# ---------------------------------------------------------------------------
def _make_mask(group_ids: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
    B = group_ids.shape[0]
    group_mask = (group_ids[:, None] == group_ids[None, :]).float()
    time_mask  = torch.ones(B, T, device=group_ids.device)
    gtm = torch.einsum("qb,bt->qbt", group_mask, time_mask)
    gtm = gtm.permute(2, 0, 1).unsqueeze(1)
    return ((1.0 - gtm) * torch.finfo(dtype).min).to(dtype)


def _make_optimized_gsa(
    baseline: GroupSelfAttention,
    group_ids: torch.Tensor,
    variant: KernelVariant,
) -> GroupSelfAttention:
    partition = GroupPartition.from_group_ids(group_ids)
    opt = GroupSelfAttention(CFG).to(DEVICE).eval()
    opt.load_state_dict(baseline.state_dict())
    mha = GroupAwareMHA(CFG, partition, variant=variant).to(DEVICE)
    mha.load_state_dict(baseline.self_attention.state_dict())
    opt.self_attention = mha
    return opt


def _make_univariate_gsa(baseline: GroupSelfAttention) -> GroupSelfAttention:
    opt = GroupSelfAttention(CFG).to(DEVICE).eval()
    opt.load_state_dict(baseline.state_dict())
    mha_sd = baseline.self_attention.state_dict()
    univ = UnivariateGroupAwareMHA(CFG).to(DEVICE)
    univ.load_state_dict({k: v for k, v in mha_sd.items() if k in ("v.weight", "o.weight")})
    opt.self_attention = univ
    return opt


# ---------------------------------------------------------------------------
# End-to-end encoder builder
# ---------------------------------------------------------------------------
def _apply_encoder_transform(
    encoder: Chronos2Encoder,
    group_ids: torch.Tensor,
    variant: KernelVariant,
) -> Chronos2Encoder:
    enc = copy.deepcopy(encoder)
    partition = GroupPartition.from_group_ids(group_ids.cpu())
    for module in enc.modules():
        if not isinstance(module, GroupSelfAttention):
            continue
        if isinstance(module.self_attention, (GroupAwareMHA, UnivariateGroupAwareMHA)):
            continue
        device = next(module.parameters()).device
        mha = GroupAwareMHA(module.self_attention.config, partition, variant=variant)
        mha.load_state_dict(module.self_attention.state_dict())
        mha.to(device)
        module.self_attention = mha
    return enc


def _apply_encoder_univariate(encoder: Chronos2Encoder) -> Chronos2Encoder:
    enc = copy.deepcopy(encoder)
    for module in enc.modules():
        if not isinstance(module, GroupSelfAttention):
            continue
        if isinstance(module.self_attention, (GroupAwareMHA, UnivariateGroupAwareMHA)):
            continue
        device = next(module.parameters()).device
        mha_sd = module.self_attention.state_dict()
        univ = UnivariateGroupAwareMHA(module.self_attention.config)
        univ.load_state_dict({k: v for k, v in mha_sd.items() if k in ("v.weight", "o.weight")})
        univ.to(device)
        module.self_attention = univ
    return enc


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------
def _section(title: str):
    print()
    print(f"{'─' * 4}  {title}  {'─' * 4}")
    print()


def _table_header(cols: list[str]) -> tuple[str, int]:
    hdr = "  ".join(cols)
    return hdr, len(hdr)


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
@dataclass
class LayerCase:
    name: str
    description: str
    T: int
    group_ids: torch.Tensor


@dataclass
class E2ECase:
    name: str
    description: str
    group_ids: torch.Tensor


def _layer_cases() -> list[LayerCase]:
    cases: list[LayerCase] = []

    for B, g in [(128, 2), (256, 2), (128, 4), (256, 4), (128, 8), (256, 8)]:
        cases.append(LayerCase(
            f"uniform g={g} B={B}", f"{B//g} groups of {g}", 64,
            torch.arange(B, dtype=torch.long) // g,
        ))

    for B_pairs, B_quads in [(64, 64), (128, 128)]:
        gids = torch.cat([
            torch.arange(B_pairs, dtype=torch.long) // 2,
            torch.arange(B_pairs // 2, B_pairs // 2 + B_quads // 4, dtype=torch.long).repeat_interleave(4),
        ])
        cases.append(LayerCase(f"mix2+4 B={len(gids)}", f"{B_pairs//2}×2 + {B_quads//4}×4", 64, gids))

    for scale in [1, 2]:
        gids = torch.tensor(
            [i for i in range(20 * scale) for _ in range(2)]
            + [i for i in range(20 * scale, 30 * scale) for _ in range(4)]
            + [i for i in range(30 * scale, 34 * scale) for _ in range(8)],
            dtype=torch.long,
        )
        cases.append(LayerCase(f"mix2+4+8 B={len(gids)}", f"{20*scale}×2 + {10*scale}×4 + {4*scale}×8", 64, gids))

    for scale in [1, 2]:
        gids = torch.tensor(
            list(range(50 * scale))
            + [i for i in range(50 * scale, 100 * scale) for _ in range(2)]
            + [i for i in range(100 * scale, 115 * scale) for _ in range(4)]
            + [i for i in range(115 * scale, 117 * scale) for _ in range(8)],
            dtype=torch.long,
        )
        cases.append(LayerCase(
            f"realistic B={len(gids)}",
            f"{50*scale}×1 + {50*scale}×2 + {15*scale}×4 + {2*scale}×8",
            64, gids,
        ))

    for T in [16, 32, 64, 128]:
        gids = torch.tensor(
            [i for i in range(20) for _ in range(2)]
            + [i for i in range(20, 30) for _ in range(4)]
            + [i for i in range(30, 34) for _ in range(8)],
            dtype=torch.long,
        )
        cases.append(LayerCase(f"mix2+4+8 T={T}", f"3-bucket mix, T={T}", T, gids))

    return cases


def _e2e_cases() -> list[E2ECase]:
    cases: list[E2ECase] = []

    for B in [64, 128, 256]:
        cases.append(E2ECase(f"univariate B={B}", f"{B} independent series",
                             torch.arange(B, dtype=torch.long)))
    for B in [64, 128, 256]:
        cases.append(E2ECase(f"pairs B={B}", f"{B//2} groups of 2",
                             torch.arange(B, dtype=torch.long) // 2))
    for B in [64, 128, 256]:
        cases.append(E2ECase(f"quads B={B}", f"{B//4} groups of 4",
                             torch.arange(B, dtype=torch.long) // 4))

    for scale in [1, 2]:
        gids = torch.tensor(
            list(range(20 * scale))
            + [i for i in range(20 * scale, 60 * scale) for _ in range(2)]
            + [i for i in range(60 * scale, 75 * scale) for _ in range(4)]
            + [i for i in range(75 * scale, 77 * scale) for _ in range(8)],
            dtype=torch.long,
        )
        cases.append(E2ECase(
            f"realistic B={len(gids)}",
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

    sep = "─"
    print(f"\nDevice : {DEVICE}")
    print(f"Config : d_model={CFG.d_model}  n_heads={CFG.num_heads}  "
          f"d_kv={CFG.d_kv}  num_layers={CFG.num_layers}")

    # ── 1. Layer-level ───────────────────────────────────────────────────────
    _section("1 / 3  —  Layer-level  (isolated GroupSelfAttention, T=64 unless noted)")
    print(f"Warmup: {LAYER_WARMUP}   Iters: {LAYER_ITERS}")

    torch.manual_seed(0)
    layer_baseline = GroupSelfAttention(CFG).to(DEVICE).eval()

    W0, W1, W2 = 22, 30, 11
    W3 = 10
    hdr, width = _table_header([
        f"{'Case':<{W0}}", f"{'Description':<{W1}}",
        f"{'Baseline':>{W2}}", f"{'Triton':>{W2}}", f"{'Bucketed':>{W2}}",
        f"{'vs Triton':>{W3}}", f"{'vs Baseline':>{W3}}",
    ])
    print(sep * width)
    print(hdr)
    print(sep * width)

    for c in _layer_cases():
        gids = c.group_ids.to(DEVICE)
        hs   = torch.randn(gids.shape[0], c.T, CFG.d_model, device=DEVICE, dtype=DTYPE)
        mask = _make_mask(gids, c.T, DTYPE)

        triton_mod   = _make_optimized_gsa(layer_baseline, gids, KernelVariant.TRITON)
        bucketed_mod = _make_optimized_gsa(layer_baseline, gids, KernelVariant.TRITON_BUCKETED)

        base_ms = _bench(layer_baseline, (hs, mask), LAYER_WARMUP, LAYER_ITERS)
        tri_ms  = _bench(triton_mod,     (hs, mask), LAYER_WARMUP, LAYER_ITERS)
        bck_ms  = _bench(bucketed_mod,   (hs, mask), LAYER_WARMUP, LAYER_ITERS)

        print("  ".join([
            f"{c.name:<{W0}}", f"{c.description:<{W1}}",
            f"{base_ms:>{W2}.3f}", f"{tri_ms:>{W2}.3f}", f"{bck_ms:>{W2}.3f}",
            f"{tri_ms / bck_ms:>{W3-1}.2f}x", f"{base_ms / bck_ms:>{W3-1}.2f}x",
        ]))

    print(sep * width)
    print("vs Triton   = triton_ms   / bucketed_ms")
    print("vs Baseline = baseline_ms / bucketed_ms")

    # ── 2. Univariate ────────────────────────────────────────────────────────
    _section("2 / 3  —  Univariate  (all groups size 1, T=64)")
    print(f"Warmup: {LAYER_WARMUP}   Iters: {LAYER_ITERS}")

    W4 = 13
    hdr2, width2 = _table_header([
        f"{'Case':<18}",
        f"{'Baseline':>{W2}}", f"{'Triton':>{W2}}", f"{'Bucketed':>{W2}}", f"{'Univariate':>{W2}}",
        f"{'vs Triton':>{W4}}", f"{'vs Bucketed':>{W4}}", f"{'vs Baseline':>{W4}}",
    ])
    print(sep * width2)
    print(hdr2)
    print(sep * width2)

    univ_mod = _make_univariate_gsa(layer_baseline)

    for B in [64, 128, 256, 512]:
        gids = torch.arange(B, dtype=torch.long).to(DEVICE)
        hs   = torch.randn(B, 64, CFG.d_model, device=DEVICE, dtype=DTYPE)
        mask = _make_mask(gids, 64, DTYPE)

        tri_mod  = _make_optimized_gsa(layer_baseline, gids, KernelVariant.TRITON)
        bck_mod  = _make_optimized_gsa(layer_baseline, gids, KernelVariant.TRITON_BUCKETED)

        base_ms = _bench(layer_baseline, (hs, mask), LAYER_WARMUP, LAYER_ITERS)
        tri_ms  = _bench(tri_mod,        (hs, mask), LAYER_WARMUP, LAYER_ITERS)
        bck_ms  = _bench(bck_mod,        (hs, mask), LAYER_WARMUP, LAYER_ITERS)
        univ_ms = _bench(univ_mod,       (hs, mask), LAYER_WARMUP, LAYER_ITERS)

        print("  ".join([
            f"{'univ B='+str(B):<18}",
            f"{base_ms:>{W2}.3f}", f"{tri_ms:>{W2}.3f}", f"{bck_ms:>{W2}.3f}", f"{univ_ms:>{W2}.3f}",
            f"{tri_ms  / univ_ms:>{W4-1}.2f}x",
            f"{bck_ms  / univ_ms:>{W4-1}.2f}x",
            f"{base_ms / univ_ms:>{W4-1}.2f}x",
        ]))

    print(sep * width2)
    print("Univariate = UnivariateGroupAwareMHA — no Q/K, decided at pass time")
    print("vs X = X_ms / univ_ms")

    # ── 3. End-to-end ────────────────────────────────────────────────────────
    _section(f"3 / 3  —  End-to-end  (full Chronos2Encoder, {E2E_SEQ_LEN} patches)")
    print(f"Seq len: {E2E_SEQ_LEN} patches  "
          f"(context_length=512, patch_size=16, +1 future patch)")
    print(f"Warmup: {E2E_WARMUP}   Iters: {E2E_ITERS}")

    torch.manual_seed(0)
    e2e_baseline = Chronos2Encoder(CFG).to(DEVICE).eval()

    W5, W6 = 34, 10
    hdr3, width3 = _table_header([
        f"{'Case':<{W0}}", f"{'Description':<{W5}}",
        f"{'Baseline':>{W2}}", f"{'Triton':>{W2}}", f"{'Bucketed':>{W2}}",
        f"{'vs Triton':>{W6}}", f"{'vs Baseline':>{W6}}",
    ])
    print(sep * width3)
    print(hdr3)
    print(sep * width3)

    for c in _e2e_cases():
        B    = c.group_ids.shape[0]
        gids = c.group_ids.to(DEVICE)
        emb  = torch.randn(B, E2E_SEQ_LEN, CFG.d_model, device=DEVICE, dtype=DTYPE)
        amask = torch.ones(B, E2E_SEQ_LEN, device=DEVICE, dtype=DTYPE)

        tri_enc = _apply_encoder_transform(e2e_baseline, c.group_ids, KernelVariant.TRITON)
        bck_enc = _apply_encoder_transform(e2e_baseline, c.group_ids, KernelVariant.TRITON_BUCKETED)

        # Use UnivariateGroupAwareMHA for the encoder univariate cases
        is_univ = c.group_ids.unique().shape[0] == c.group_ids.shape[0]
        if is_univ:
            bck_enc = _apply_encoder_univariate(e2e_baseline)

        def _base():    e2e_baseline(emb, group_ids=gids, attention_mask=amask)
        def _triton():  tri_enc(emb, group_ids=gids, attention_mask=amask)
        def _bucketed(): bck_enc(emb, group_ids=gids, attention_mask=amask)

        base_ms = _bench(_base,    (), E2E_WARMUP, E2E_ITERS)
        tri_ms  = _bench(_triton,  (), E2E_WARMUP, E2E_ITERS)
        bck_ms  = _bench(_bucketed, (), E2E_WARMUP, E2E_ITERS)

        label = c.name + (" [univ]" if is_univ else "")
        print("  ".join([
            f"{label:<{W0}}", f"{c.description:<{W5}}",
            f"{base_ms:>{W2}.3f}", f"{tri_ms:>{W2}.3f}", f"{bck_ms:>{W2}.3f}",
            f"{tri_ms  / bck_ms:>{W6-1}.2f}x",
            f"{base_ms / bck_ms:>{W6-1}.2f}x",
        ]))

    print(sep * width3)
    print("Bucketed column uses UnivariateGroupAwareMHA for all-univariate cases [univ].")
    print("vs Triton   = triton_ms   / bucketed_ms")
    print("vs Baseline = baseline_ms / bucketed_ms")
    print()


if __name__ == "__main__":
    main()
