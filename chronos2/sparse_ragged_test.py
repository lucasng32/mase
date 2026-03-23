import os
import torch
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault("HOME", os.environ.get("USERPROFILE", str(Path.home())))

from chop.models import get_model

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "amazon/chronos-2"
CONTEXT_LEN = 1440
BATCH_SIZES = [64, 128, 256]

# ==========================================
# 1. Model Setup
# ==========================================
def setup_models():
    print("Loading Dense (Baseline) Model...")
    dense_model = get_model("chronos-2", pretrained=True, model_id=MODEL_ID, group_attn_backend="dense", _attn_implementation="sdpa")
    dense_model = dense_model.to(DEVICE).to(torch.bfloat16)
    dense_model.eval()
    
    print("Loading Sparse (BSR) Model...")
    sparse_model = get_model("chronos-2", pretrained=True, model_id=MODEL_ID, group_attn_backend="bsr", _attn_implementation="sdpa")
    sparse_model = sparse_model.to(DEVICE).to(torch.bfloat16)
    sparse_model.eval()

    print("Loading Ragged Model...")
    ragged_model = get_model("chronos-2", pretrained=True, model_id=MODEL_ID, group_attn_backend="ragged", _attn_implementation="sdpa")
    ragged_model = ragged_model.to(DEVICE).to(torch.bfloat16)
    ragged_model.eval()

    return dense_model, sparse_model, ragged_model

# ==========================================
# 2. Synthetic Data Generators
# ==========================================
def _generate_base_tensors(batch_size: int, context_length: int):
    context = torch.randn(batch_size, context_length, dtype=torch.bfloat16, device=DEVICE)
    context_mask = torch.ones(batch_size, context_length, dtype=torch.bfloat16, device=DEVICE)
    return context, context_mask

def generate_uniform_pairs(batch_size: int, context_length: int = 1440, group_size: int = 2):
    """Standard multivariate pairs. (Best case for Tensor Core alignment)."""
    context, context_mask = _generate_base_tensors(batch_size, context_length)
    num_groups = max(1, batch_size // group_size)
    group_ids = torch.arange(num_groups, device=DEVICE).repeat_interleave(group_size)
    
    if len(group_ids) < batch_size:
        pad = torch.full((batch_size - len(group_ids),), num_groups - 1, device=DEVICE)
        group_ids = torch.cat([group_ids, pad])
    return context, context_mask, group_ids

def generate_independent_data(batch_size: int, context_length: int = 1440):
    """Every sequence is independent. (The BSR padding nightmare)."""
    context, context_mask = _generate_base_tensors(batch_size, context_length)
    group_ids = torch.arange(batch_size, device=DEVICE)
    return context, context_mask, group_ids

def generate_skewed_data(batch_size: int, context_length: int = 1440):
    """Power-law distribution. (The Ragged Optimizer)."""
    context, context_mask = _generate_base_tensors(batch_size, context_length)
    sizes = []
    remaining = batch_size
    
    if remaining > 10:
        sizes.append(remaining // 2)
        remaining -= sizes[-1]
    while remaining > 20:
        chunk = max(2, remaining // 4)
        sizes.append(chunk)
        remaining -= chunk
    while remaining > 0:
        sizes.append(1)
        remaining -= 1
        
    sizes_tensor = torch.tensor(sizes, device=DEVICE)
    group_ids = torch.repeat_interleave(torch.arange(len(sizes), device=DEVICE), sizes_tensor)
    group_ids = group_ids[torch.randperm(batch_size, device=DEVICE)]
    return context, context_mask, group_ids

def generate_single_massive_group(batch_size: int, context_length: int = 1440):
    """All sequences interact. (The Dense OOM Test)."""
    context, context_mask = _generate_base_tensors(batch_size, context_length)
    group_ids = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
    return context, context_mask, group_ids

# ==========================================
# 3. GPU Benchmark Harness
# ==========================================
def run_pure_gpu_benchmark(model, context, context_mask, group_ids, num_runs=5):
    torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(2):
        with torch.no_grad():
            model(context=context, context_mask=context_mask, group_ids=group_ids)
            
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        with torch.no_grad():
            model(context=context, context_mask=context_mask, group_ids=group_ids)
    end_event.record()
    
    torch.cuda.synchronize()
    
    avg_latency_ms = start_event.elapsed_time(end_event) / num_runs
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return avg_latency_ms, peak_vram_mb

# ==========================================
# 4. Test Runner
# ==========================================
def run_scenario(name, data_func, models, batch_size):
    print(f"\n--- Scenario: {name} | Batch Size: {batch_size} ---")
    context, context_mask, group_ids = data_func(batch_size, CONTEXT_LEN)
    dense_model, sparse_model, ragged_model = models
    
    # Test Sparse
    try:
        sparse_lat, sparse_vram = run_pure_gpu_benchmark(sparse_model, context, context_mask, group_ids)
        print(f"  [SPARSE] Latency: {sparse_lat:>6.1f} ms | VRAM: {sparse_vram:>6.1f} MB")
    except Exception as e:
        print(f"  [SPARSE] FAILED: {type(e).__name__}")
        torch.cuda.empty_cache()

    # Test Ragged
    try:
        ragged_lat, ragged_vram = run_pure_gpu_benchmark(ragged_model, context, context_mask, group_ids)
        print(f"  [RAGGED] Latency: {ragged_lat:>6.1f} ms | VRAM: {ragged_vram:>6.1f} MB")
    except Exception as e:
        print(f"  [RAGGED] FAILED: {type(e).__name__}")
        torch.cuda.empty_cache()

    # Test Dense (Highest risk of OOM)
    try:
        dense_lat, dense_vram = run_pure_gpu_benchmark(dense_model, context, context_mask, group_ids)
        print(f"  [DENSE]  Latency: {dense_lat:>6.1f} ms | VRAM: {dense_vram:>6.1f} MB")
    except torch.cuda.OutOfMemoryError:
        print(f"  [DENSE]  💥 OUT OF MEMORY (OOM)")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [DENSE]  FAILED: {type(e).__name__}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    models = setup_models()
    
    scenarios = [
        ("Independent Series (Worst-case BSR)", generate_independent_data),
        ("Uniform Pairs (Standard Multivariate)", generate_uniform_pairs),
        ("Skewed Distribution (Best-case Ragged)", generate_skewed_data),
        ("Single Massive Group (Worst-case Dense)", generate_single_massive_group)
    ]
    
    for b in BATCH_SIZES:
        for name, func in scenarios:
            run_scenario(name, func, models, b)
            
    print("\nBenchmark Suite Complete.")