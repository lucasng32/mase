"""
Benchmark: Fused RoPE+Attention (Triton) vs Baseline MHA with RoPE.

Compares three implementations:
  1. baseline_eager  — MHA(attn_implementation="eager") + RoPE applied before matmul
  2. baseline_sdpa   — MHA(attn_implementation="sdpa") + RoPE applied before matmul (production default)
  3. fused_triton    — inner MHA replaced directly with RoPEFusedMHA

Usage:
    uv run chronos2/benchmark_rope_attn.py
"""
import os
import sys
import time

# ── nvidia library path (mirrors test.py) ──────────────────────────────────
venv_base = os.path.expanduser("~/mase/.venv/lib/python3.11/site-packages/nvidia")
nvidia_libs: list[str] = []
if os.path.exists(venv_base):
    for root, dirs, _ in os.walk(venv_base):
        if "lib" in dirs:
            nvidia_libs.append(os.path.join(root, "lib"))
os.environ["LD_LIBRARY_PATH"] = (
    ":".join(nvidia_libs) + ":/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import MHA, TimeSelfAttention
from chop.models.chronos2.optimized_layers import RoPEFusedMHA
from chop.models.chronos2.triton_rope_attn import is_triton_available

# ── benchmark config ────────────────────────────────────────────────────────

WARMUP = 20
REPEATS = 100
DTYPE = torch.float32

# (batch_size, seq_length) pairs to sweep
CONFIGS = [
    (1,  64),
    (4,  64),
    (4,  128),
    (8,  128),
    (8,  256),
    (16, 256),
    (16, 512),
    (16, 1024),
    (16, 2048)
]


# ── helpers ─────────────────────────────────────────────────────────────────

def make_inputs(
    B: int, S: int, d_model: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (hidden_states, mask, position_ids) on the given device."""
    hidden = torch.randn(B, S, d_model, device=device, dtype=dtype)
    mask = torch.zeros(B, 1, S, S, device=device, dtype=dtype)
    pos = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    return hidden, mask, pos


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_fn(
    fn,
    *args,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    device: torch.device,
) -> float:
    """Return mean latency in milliseconds."""
    # warm-up
    for _ in range(warmup):
        fn(*args)
    _sync(device)

    if device.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(repeats):
            fn(*args)
        end_ev.record()
        torch.cuda.synchronize()
        return start_ev.elapsed_time(end_ev) / repeats  # ms per call
    else:
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(*args)
        return (time.perf_counter() - t0) / repeats * 1_000  # ms per call


def check_outputs_close(
    a: torch.Tensor,
    b: torch.Tensor,
    label: str,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> bool:
    close = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    if not close:
        max_diff = (a.float() - b.float()).abs().max().item()
        print(f"    [WARN] {label}: max abs diff = {max_diff:.2e}")
    return close


# ── build modules ────────────────────────────────────────────────────────────

def build_modules(
    config: Chronos2CoreConfig, device: torch.device, dtype: torch.dtype
) -> tuple[TimeSelfAttention, TimeSelfAttention, TimeSelfAttention]:
    """Return (baseline_eager, baseline_sdpa, fused) with shared random weights.

    The fused module is produced by directly replacing the inner MHA of
    a TimeSelfAttention instance with RoPEFusedMHA.
    """
    def _make_tsa(attn_impl: str) -> TimeSelfAttention:
        cfg = Chronos2CoreConfig(
            d_model=config.d_model,
            d_kv=config.d_kv,
            d_ff=config.d_ff,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            rope_theta=config.rope_theta,
            attn_implementation=attn_impl,
        )
        return TimeSelfAttention(cfg).to(device, dtype=dtype)

    eager_tsa = _make_tsa("eager")
    sdpa_tsa  = _make_tsa("sdpa")
    fused_tsa = _make_tsa("sdpa")

    # Sync all weights to the same random initialisation
    sd = eager_tsa.state_dict()
    sdpa_tsa.load_state_dict(sd, strict=True)
    fused_tsa.load_state_dict(sd, strict=True)

    # Directly swap the inner MHA with RoPEFusedMHA
    original_mha = fused_tsa.self_attention
    fused_mha = RoPEFusedMHA(config=original_mha.config)
    fused_mha.load_state_dict(original_mha.state_dict(), strict=False)
    fused_mha.to(device)
    fused_tsa.self_attention = fused_mha

    for m in (eager_tsa, sdpa_tsa, fused_tsa):
        m.eval()

    return eager_tsa, sdpa_tsa, fused_tsa


# ── main ─────────────────────────────────────────────────────────────────────

def _print_header() -> None:
    print(
        f"\n{'B':>4} {'S':>5} | "
        f"{'eager (ms)':>11} "
        f"{'sdpa (ms)':>10} "
        f"{'fused (ms)':>11} | "
        f"{'vs eager':>9} "
        f"{'vs sdpa':>8} | "
        f"{'match':>6}"
    )
    print("-" * 78)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"Triton        : {'available ✓' if is_triton_available() else 'NOT available (eager fallback)'}")
    print(f"dtype         : {DTYPE}")
    print(f"Warmup/repeats: {WARMUP} / {REPEATS}")

    # Check CUDA actually works (GPU may be present but unsupported by this PyTorch build)
    if device.type == "cuda":
        try:
            torch.zeros(1, device=device)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            print(f"[WARN] CUDA device unusable ({exc}); falling back to CPU.")
            device = torch.device("cpu")

    config = Chronos2CoreConfig()
    print(
        f"Model config  : d_model={config.d_model}, "
        f"num_heads={config.num_heads}, d_kv={config.d_kv}"
    )

    # Validate the transform pass once and report what it did
    _probe_eager, _probe_sdpa, _probe_fused = build_modules(config, device, DTYPE)
    fused_inner = _probe_fused.self_attention
    print(f"Fused module  : {type(fused_inner).__name__} (direct construction)")

    _print_header()

    for B, S in CONFIGS:
        hidden, mask, pos = make_inputs(B, S, config.d_model, device, DTYPE)

        eager_tsa, sdpa_tsa, fused_tsa = build_modules(config, device, DTYPE)

        # — correctness check —————————————————————————————————————————————
        with torch.no_grad():
            out_eager = eager_tsa(hidden, mask, position_ids=pos).hidden_states
            out_sdpa  = sdpa_tsa(hidden,  mask, position_ids=pos).hidden_states
            out_fused = fused_tsa(hidden, mask, position_ids=pos).hidden_states

        sdpa_ok  = check_outputs_close(out_eager, out_sdpa,  "sdpa vs eager")
        fused_ok = check_outputs_close(out_eager, out_fused, f"fused vs eager (B={B},S={S})")
        match_str = "✓" if fused_ok else "✗"

        # — latency ───────────────────────────────────────────────────────
        def _run_eager():
            eager_tsa(hidden, mask, position_ids=pos)

        def _run_sdpa():
            sdpa_tsa(hidden, mask, position_ids=pos)

        def _run_fused():
            fused_tsa(hidden, mask, position_ids=pos)

        with torch.no_grad():
            t_eager = benchmark_fn(_run_eager, device=device)
            t_sdpa  = benchmark_fn(_run_sdpa,  device=device)
            t_fused = benchmark_fn(_run_fused, device=device)

        su_eager = t_eager / t_fused
        su_sdpa  = t_sdpa  / t_fused

        print(
            f"{B:>4} {S:>5} | "
            f"{t_eager:>11.3f} "
            f"{t_sdpa:>10.3f} "
            f"{t_fused:>11.3f} | "
            f"{su_eager:>8.2f}x "
            f"{su_sdpa:>7.2f}x | "
            f"{match_str:>6}"
        )

    print()


if __name__ == "__main__":
    main()
