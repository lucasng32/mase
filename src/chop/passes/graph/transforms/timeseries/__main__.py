"""
Two-stage benchmark:

1. Isolated module benchmark — GroupSelfAttention vs FastGroupSelfAttention
   with synthetic inputs. Isolates exactly the op our optimisation targets,
   free from TimeSelfAttention / FeedForward noise.

2. Full-model benchmark — baseline Chronos2Model vs post-transform mg.model,
   using the same high-gain input cases.
"""

import os
import statistics
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("HOME", os.environ.get("USERPROFILE", str(Path.home())))

import torch

from chop.models import get_model
from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.optimized_layers import FastGroupSelfAttention, compute_groups

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32
print(f"Device: {DEVICE}")

# ── Load model (for config and full-model benchmark) ─────────────────────────
model = get_model("chronos-2", pretrained=True)
model.eval().to(DEVICE)

cfg     = model.encoder.block[0].layer[1].self_attention.config
C_LEN   = model.config.chronos_config["context_length"]
OUT_PATCH = model.config.chronos_config["output_patch_size"]
D_MODEL = cfg.d_model
print(f"d_model={D_MODEL}, n_heads={cfg.num_heads}, d_kv={cfg.d_kv}")

# ── Benchmark helper ──────────────────────────────────────────────────────────
def benchmark(fn, args: tuple, warmup=10, iters=50) -> float:
    use_cuda = DEVICE == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        if use_cuda:
            torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            if use_cuda:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(*args); e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                import time
                t0 = time.perf_counter()
                fn(*args)
                times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)

# ── Mask construction (mirrors Chronos2Encoder._construct_and_invert_group_time_mask) ──
def make_group_time_mask(group_ids: torch.Tensor, T: int) -> torch.Tensor:
    B          = group_ids.shape[0]
    group_mask = (group_ids[:, None] == group_ids[None, :]).float()
    time_mask  = torch.ones(B, T, device=group_ids.device)
    gtm        = torch.einsum("qb,bt->qbt", group_mask, time_mask)
    gtm        = gtm.permute(2, 0, 1).unsqueeze(1)
    gtm        = (1.0 - gtm) * torch.finfo(DTYPE).min
    return gtm.to(DTYPE)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ISOLATED MODULE BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Isolated GroupSelfAttention benchmark (synthetic inputs)")
print("=" * 60)

# T=64 patches = 1024-token context / patch_size=16.
# Sweep B to show quadratic savings as groups grow sparse.
T = 64

iso_cases = {
    "univ  B=32":  torch.arange(32,  dtype=torch.long),
    "univ  B=64":  torch.arange(64,  dtype=torch.long),
    "univ  B=128": torch.arange(128, dtype=torch.long),
    "pairs B=64":  torch.arange(64,  dtype=torch.long) // 2,
    "pairs B=128": torch.arange(128, dtype=torch.long) // 2,
    "quads B=128": torch.arange(128, dtype=torch.long) // 4,
}

baseline_mod = GroupSelfAttention(cfg).to(DEVICE).eval()

print(f"\n{'case':<16} {'baseline (ms)':>14} {'fast (ms)':>10} {'speedup':>9}")
print("-" * 54)

for name, group_ids in iso_cases.items():
    B         = group_ids.shape[0]
    group_ids = group_ids.to(DEVICE)
    hs        = torch.randn(B, T, D_MODEL, device=DEVICE, dtype=DTYPE)
    mask      = make_group_time_mask(group_ids, T)

    groups    = compute_groups(group_ids)
    fast_mod  = FastGroupSelfAttention(cfg, groups=groups).to(DEVICE).eval()
    fast_mod.load_state_dict(baseline_mod.state_dict(), strict=False)

    base_ms = benchmark(baseline_mod, (hs, mask))
    fast_ms = benchmark(fast_mod,     (hs, mask))
    print(f"{name:<16} {base_ms:>14.3f} {fast_ms:>10.3f} {base_ms/fast_ms:>8.3f}x")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FULL-MODEL BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Full-model benchmark (baseline vs post-transform)")
print("=" * 60)

from chop import MaseGraph
from chop.models.chronos2.modeling_chronos2 import TimeSelfAttention
from chop.passes.graph.transforms.timeseries import fast_group_attention_transform_pass

B = 64

full_cases = {
    "univ  B=64":  torch.arange(B, dtype=torch.long),
    "pairs B=64":  torch.arange(B, dtype=torch.long) // 2,
    "quads B=64":  torch.arange(B, dtype=torch.long) // 4,
}

def make_full_inputs(group_ids):
    return {
        "context":            torch.randn(B, C_LEN, device=DEVICE),
        "group_ids":          group_ids.to(DEVICE),
        "num_output_patches": OUT_PATCH,
    }

mg = MaseGraph(
    model=model,
    hf_input_names=["context", "group_ids", "num_output_patches"],
    custom_ops={
        "modules": {GroupSelfAttention: {}, TimeSelfAttention: {}},
        "functions": {},
    },
)
mg.model.chronos_config = model.chronos_config

print(f"\n{'case':<16} {'baseline (ms)':>14} {'fast (ms)':>10} {'speedup':>9}")
print("-" * 54)

for name, group_ids in full_cases.items():
    inputs = make_full_inputs(group_ids)

    # Fresh transform per case since group_ids differ
    from copy import deepcopy
    mg_case = deepcopy(mg)
    mg_case, info = fast_group_attention_transform_pass(
        mg_case, pass_args={"group_ids": group_ids}
    )
    mg_case.fx_graph.lint()

    fast_model = mg_case.model  # capture before next loop iteration overwrites mg_case
    base_ms = benchmark(lambda x: model(**x),       (inputs,), warmup=5, iters=20)
    fast_ms = benchmark(lambda x: fast_model(**x),  (inputs,), warmup=5, iters=20)
    print(f"{name:<16} {base_ms:>14.2f} {fast_ms:>10.2f} {base_ms/fast_ms:>8.3f}x")
