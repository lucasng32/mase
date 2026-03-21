#!/usr/bin/env python3
"""
End-to-end benchmark: full Chronos2 encoder, baseline vs optimised.

Constructs Chronos2Encoder + Chronos2Model from config (random weights, no
download needed), swaps every GroupSelfAttention in the encoder for
FastGroupSelfAttention, then measures:

  1. The full encoder forward (all 6 blocks: TimeAttn + GroupAttn + FFN).
  2. The complete Chronos2Model forward (patch → embed → encoder → head).

Both use the same random weights so the only difference is the GroupAttn kernel.

Run::

    uv run python test/passes/graph/transforms/timeseries/bench_chronos2_e2e.py
"""

from __future__ import annotations

import copy
import statistics
import time

import torch

from chop.models.chronos2.configuration_chronos2 import (
    Chronos2CoreConfig,
    Chronos2ForecastingConfig,
)
from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.modeling_chronos2 import Chronos2Encoder, Chronos2Model
from chop.models.chronos2.optimized_layers import (
    FastGroupSelfAttention,
    GroupPartition,
    KernelDispatcher,
)

# ---------------------------------------------------------------------------
# Config — Chronos-2-Small dimensions
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
WARMUP = 10
ITERS = 50

# Transformer core
CORE_CFG = Chronos2CoreConfig(
    d_model=512,
    d_kv=64,
    d_ff=2048,
    num_layers=6,
    num_heads=8,
    dropout_rate=0.0,
    attn_implementation="sdpa",
)

# Forecasting wrapper — needed to instantiate Chronos2Model
CHRONOS_CFG_DICT = dict(
    context_length=512,
    input_patch_size=32,
    input_patch_stride=32,
    output_patch_size=32,
    quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    use_reg_token=True,
    use_arcsinh=True,
    max_output_patches=8,
)
CONTEXT_LENGTH = CHRONOS_CFG_DICT["context_length"]
NUM_OUTPUT_PATCHES = 4


def _make_full_config() -> Chronos2CoreConfig:
    cfg = Chronos2CoreConfig(
        d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_heads=8,
        dropout_rate=0.0, attn_implementation="eager",
    )
    cfg.chronos_config = CHRONOS_CFG_DICT
    return cfg


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _bench(fn, args: tuple, warmup: int = WARMUP, iters: int = ITERS) -> float:
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


def _peak_mb(fn, args: tuple) -> float:
    if DEVICE.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    with torch.no_grad():
        fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / (1024**2)


# ---------------------------------------------------------------------------
# Module swap (mirrors MASE pass without requiring MaseGraph)
# ---------------------------------------------------------------------------
def _optimise_encoder(encoder: Chronos2Encoder, group_ids: torch.Tensor) -> str:
    """Replace every GroupSelfAttention with FastGroupSelfAttention in-place."""
    partition = GroupPartition.from_group_ids(group_ids)
    for block in encoder.block:
        module = block.layer[1]
        if not isinstance(module, GroupSelfAttention):
            continue
        device = next(module.parameters()).device
        variant = KernelDispatcher.select(partition, device)
        fast = FastGroupSelfAttention(
            config=module.self_attention.config,
            partition=partition,
            variant=variant,
        )
        fast.load_state_dict(module.state_dict(), strict=False)
        fast.to(device)
        block.layer[1] = fast
    # All layers share the same partition so variant is uniform
    first_fast = encoder.block[0].layer[1]
    return first_fast._variant.name.lower()


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------
def _print_table(rows: list[dict], has_cuda: bool) -> None:
    sep = "─"
    cols = dict(name=18, desc=26, base_ms=14, fast_ms=14, variant=14, speedup=9)
    if has_cuda:
        cols["mem_base"] = 10
        cols["mem_fast"] = 10

    headers = {
        "name": "Case", "desc": "Description",
        "base_ms": "Baseline (ms)", "fast_ms": "Ours (ms)",
        "variant": "Variant", "speedup": "Speedup",
        "mem_base": "Mem base", "mem_fast": "Mem ours",
    }

    def fmt_hdr(k):
        w = cols[k]
        h = headers[k]
        return f"{h:<{w}}" if k in ("name", "desc") else f"{h:>{w}}"

    hdr = "  ".join(fmt_hdr(k) for k in cols)
    width = len(hdr)
    print(sep * width)
    print(hdr)
    print(sep * width)

    for r in rows:
        parts = [
            f"{r['name']:<{cols['name']}}",
            f"{r['desc']:<{cols['desc']}}",
            f"{r['base_ms']:>{cols['base_ms']}.3f}",
            f"{r['fast_ms']:>{cols['fast_ms']}.3f}",
            f"{r['variant']:>{cols['variant']}}",
            f"{r['speedup']:>{cols['speedup'] - 1}.2f}x",
        ]
        if has_cuda:
            parts += [
                f"{r['mem_base']:>{cols['mem_base'] - 2}.1f}MB",
                f"{r['mem_fast']:>{cols['mem_fast'] - 2}.1f}MB",
            ]
        print("  ".join(parts))

    print(sep * width)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def _mixed(n_univ, n_pairs, n_quads, n_octs) -> torch.Tensor:
    ids = []
    g = 0
    for _ in range(n_univ):  ids.append(g); g += 1
    for _ in range(n_pairs):  ids += [g, g]; g += 1
    for _ in range(n_quads):  ids += [g]*4; g += 1
    for _ in range(n_octs):   ids += [g]*8; g += 1
    return torch.tensor(ids, dtype=torch.long)


def _case(name, group_ids, desc):
    return (name, group_ids, desc)


CASES = [
    # Univariate sweep
    _case("univ B=100",  torch.arange(100, dtype=torch.long),           "100 independent"),
    _case("univ B=250",  torch.arange(250, dtype=torch.long),           "250 independent"),
    _case("univ B=500",  torch.arange(500, dtype=torch.long),           "500 independent"),
    # Uniform pairs
    _case("pairs B=100", torch.arange(100, dtype=torch.long) // 2,      "50 groups of 2"),
    _case("pairs B=250", torch.arange(250, dtype=torch.long) // 2,      "125 groups of 2"),
    _case("pairs B=500", torch.arange(500, dtype=torch.long) // 2,      "250 groups of 2"),
    # Uniform quads
    _case("quads B=100", torch.arange(100, dtype=torch.long) // 4,      "25 groups of 4"),
    _case("quads B=500", torch.arange(500, dtype=torch.long) // 4,      "125 groups of 4"),
    # Groups of 16
    _case("g16 B=160",   torch.arange(160, dtype=torch.long) // 16,     "10 groups of 16"),
    _case("g16 B=496",   torch.arange(496, dtype=torch.long) // 16,     "31 groups of 16"),
    # Mixed — actual B derived from tensor
    _case("mixed ~132",  _mixed(20, 20, 10, 4),                         "20×1+20×2+10×4+4×8"),
    _case("mixed ~500",  _mixed(100, 75, 50, 12),                       "100×1+75×2+50×4+12×8"),
]


# ---------------------------------------------------------------------------
# Section 1: encoder-only
# ---------------------------------------------------------------------------
def bench_encoder(has_cuda: bool) -> None:
    print("━" * 60)
    print("  SECTION 1 — Encoder only (6 × TimeAttn + GroupAttn + FFN)")
    print("━" * 60)

    T = 33  # num_context_patches (512/32) + 1 reg token + num_output_patches
    rows = []
    for name, group_ids, desc in CASES:
        group_ids = group_ids.to(DEVICE)
        B = group_ids.shape[0]
        attn_mask = torch.ones(B, T, device=DEVICE, dtype=DTYPE)

        # Baseline encoder
        base_enc = Chronos2Encoder(CORE_CFG).to(DEVICE).eval()
        hidden = torch.randn(B, T, CORE_CFG.d_model, device=DEVICE, dtype=DTYPE)

        # Optimised encoder — deep copy then swap
        fast_enc = copy.deepcopy(base_enc).eval()
        variant = _optimise_encoder(fast_enc, group_ids)

        def run_base(h): return base_enc(attention_mask=attn_mask, inputs_embeds=h, group_ids=group_ids)
        def run_fast(h): return fast_enc(attention_mask=attn_mask, inputs_embeds=h, group_ids=group_ids)

        base_ms = _bench(run_base, (hidden,))
        fast_ms = _bench(run_fast, (hidden,))
        row = dict(name=name, desc=desc, base_ms=base_ms, fast_ms=fast_ms,
                   variant=variant, speedup=base_ms / fast_ms)
        if has_cuda:
            row["mem_base"] = _peak_mb(run_base, (hidden,))
            row["mem_fast"] = _peak_mb(run_fast, (hidden,))
        rows.append(row)

    _print_table(rows, has_cuda)
    print()


# ---------------------------------------------------------------------------
# Section 2: full model (patch → embed → encoder → head)
# ---------------------------------------------------------------------------
def bench_full_model(has_cuda: bool) -> None:
    print("━" * 60)
    print("  SECTION 2 — Full Chronos2Model (patch → encoder → head)")
    print("━" * 60)

    rows = []
    for name, group_ids, desc in CASES:
        group_ids_dev = group_ids.to(DEVICE)
        B = group_ids_dev.shape[0]
        context = torch.randn(B, CONTEXT_LENGTH, device=DEVICE, dtype=DTYPE)

        # Baseline full model
        cfg = _make_full_config()
        base_model = Chronos2Model(cfg).to(DEVICE).eval()

        # Optimised full model
        fast_model = copy.deepcopy(base_model).eval()
        variant = _optimise_encoder(fast_model.encoder, group_ids_dev)

        def run_base(c): return base_model(c, group_ids=group_ids_dev, num_output_patches=NUM_OUTPUT_PATCHES)
        def run_fast(c): return fast_model(c, group_ids=group_ids_dev, num_output_patches=NUM_OUTPUT_PATCHES)

        base_ms = _bench(run_base, (context,))
        fast_ms = _bench(run_fast, (context,))
        row = dict(name=name, desc=desc, base_ms=base_ms, fast_ms=fast_ms,
                   variant=variant, speedup=base_ms / fast_ms)
        if has_cuda:
            row["mem_base"] = _peak_mb(run_base, (context,))
            row["mem_fast"] = _peak_mb(run_fast, (context,))
        rows.append(row)

    _print_table(rows, has_cuda)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    has_cuda = DEVICE.type == "cuda"
    print(f"Device         : {DEVICE}")
    print(f"Config         : d_model={CORE_CFG.d_model}  n_heads={CORE_CFG.num_heads}"
          f"  n_layers={CORE_CFG.num_layers}  d_kv={CORE_CFG.d_kv}")
    print(f"Context length : {CONTEXT_LENGTH}  patch=32  output_patches={NUM_OUTPUT_PATCHES}")
    print(f"Warmup: {WARMUP}   Iters: {ITERS}")
    print()

    bench_encoder(has_cuda)
    bench_full_model(has_cuda)


if __name__ == "__main__":
    main()
