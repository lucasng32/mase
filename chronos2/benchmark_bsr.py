import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from chop.models import get_model
from chop.passes.graph.transforms.timeseries.FastBSRGroupAtten import fast_bsr_group_attention_transform_pass
from chop.ir.graph.mase_graph import MaseGraph
from chop.models.chronos2.layers import GroupSelfAttention

def benchmark_forward(model, kwargs, warmup=10, runs=50):
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(**kwargs)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            model(**kwargs)
    torch.cuda.synchronize()
    
    return (time.time() - start) / runs * 1000  # ms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available. BSR pass requires Triton and CUDA.")
        return

    print("Loading Chronos-2 model...")
    model = get_model('chronos-2', pretrained=False)
    
    # Trace Eager attention correctly
    if hasattr(model.config, '_attn_implementation'):
        model.config._attn_implementation = 'sdpa'
        
    model = model.eval().to(device).to(torch.bfloat16)

    batch_sizes = [16, 64, 128, 256]
    c_len = 1024  # Realistic context length for timeseries
    out_patch = 16

    cases = ["univariate", "pairs", "multivariate"]

    results_base = {case: [] for case in cases}
    results_bsr = {case: [] for case in cases}
    speedups = {case: [] for case in cases}

    print(f"Benchmarking with Context Length = {c_len} (bfloat16)...")

    for N in batch_sizes:
        print(f"\n=== Benchmarking Batch Size: {N} ===")
        
        group_id_cases = {
            "univariate": torch.arange(N, dtype=torch.long, device=device),
            "pairs": (torch.arange(N, dtype=torch.long, device=device) // 2),
            "multivariate": torch.zeros(N, dtype=torch.long, device=device)
        }

        for case_name, group_ids in group_id_cases.items():
            print(f"  Case: {case_name}")
            dummy_in = {
                'context': torch.randn((N, c_len), dtype=torch.bfloat16, device=device),
                'group_ids': group_ids,
                'num_output_patches': out_patch,
            }

            # Base model time (using the untouched model)
            base_time = benchmark_forward(model, dummy_in)
            results_base[case_name].append(base_time)

            import copy
            model_copy = copy.deepcopy(model)

            # BSR model time
            mg = MaseGraph(
                model_copy, 
                hf_input_names=list(dummy_in.keys()),
                custom_ops={"modules": {GroupSelfAttention: {"name": "GroupSelfAttention"}}}
            )
            mg, info = fast_bsr_group_attention_transform_pass(mg, pass_args={"group_ids": group_ids})
            
            bsr_time = benchmark_forward(mg.model, dummy_in)
            results_bsr[case_name].append(bsr_time)

            speedup = base_time / bsr_time
            speedups[case_name].append(speedup)
            
            print(f"    Base: {base_time:.2f} ms | BSR: {bsr_time:.2f} ms | Speedup: {speedup:.2f}x")
            
            # Free memory
            del mg
            del model_copy
            torch.cuda.empty_cache()

    # Plotting
    os.makedirs("artifacts/bsr_benchmark", exist_ok=True)
    
    # 1. Latency Plot
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, case in enumerate(cases):
        plt.plot(batch_sizes, results_base[case], marker='o', linestyle='--', color=colors[i], alpha=0.5, label=f"{case} (Base)")
        plt.plot(batch_sizes, results_bsr[case], marker='s', color=colors[i], linewidth=2, label=f"{case} (BSR)")
        
    plt.xlabel("Batch Size (N_VARIATES)")
    plt.ylabel("Latency (ms)")
    plt.title(f"Chronos-2 Group Attention: Base vs BSR Latency (C_LEN={c_len})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/bsr_benchmark/bsr_latency.png")
    plt.close()

    # 2. Speedup Plot
    plt.figure(figsize=(10, 6))
    for i, case in enumerate(cases):
        plt.plot(batch_sizes, speedups[case], marker='^', color=colors[i], linewidth=2, label=case)
        
    plt.axhline(1.0, color='black', linestyle='--', label="Baseline (1.0x)")
    plt.xlabel("Batch Size (N_VARIATES)")
    plt.ylabel("Speedup (Base Time / BSR Time)")
    plt.title(f"Chronos-2 Group Attention BSR Speedup (C_LEN={c_len})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/bsr_benchmark/bsr_speedup.png")
    plt.close()

    print("\nBenchmarks complete. Plots saved to 'artifacts/bsr_benchmark/' directory.")

if __name__ == "__main__":
    main()
