import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.sparse_group_mha import SparseGroupMHA

def benchmark_module(module, kwargs, warmup=20, runs=100):
    for _ in range(warmup):
        with torch.no_grad():
            module(**kwargs)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            module(**kwargs)
    torch.cuda.synchronize()
    
    return (time.time() - start) / runs * 1000

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Requires CUDA")
        return

    config = Chronos2CoreConfig(
        d_model=512,
        d_kv=64,
        num_heads=8,
        dropout_rate=0.0,
        attn_implementation="sdpa"
    )

    batch_sizes = [16, 32, 64, 128, 256, 512]
    # We use a relatively small T since Chronos-2 TimeSelfAttention typically runs on T,
    # but GroupSelfAttention runs on B (N_VARIATES).
    T = 16 

    cases = ["univariate", "pairs", "halves", "multivariate"]

    results_base = {case: [] for case in cases}
    results_bsr = {case: [] for case in cases}
    speedups = {case: [] for case in cases}

    print(f"Benchmarking GroupSelfAttention layer. T = {T}")

    for B in batch_sizes:
        print(f"\n=== Benchmarking Batch Size (N_VARIATES): {B} ===")
        
        group_id_cases = {
            "univariate": torch.arange(B, dtype=torch.long, device=device),
            "pairs": (torch.arange(B, dtype=torch.long, device=device) // 2),
            "halves": (torch.arange(B, dtype=torch.long, device=device) // max(1, B // 2)),
            "multivariate": torch.zeros(B, dtype=torch.long, device=device)
        }

        for case_name, group_ids in group_id_cases.items():
            print(f"  Case: {case_name}")
            
            # 1. Base Dense GroupSelfAttention
            base_layer = GroupSelfAttention(config).eval().to(device)
            # Create dense mask for base layer
            group_mask = group_ids[:, None] == group_ids[None, :]
            attention_mask = torch.ones(B, T, device=device, dtype=torch.float32)
            group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)
            group_time_mask = group_time_mask.permute(2, 0, 1).unsqueeze(1)
            group_time_mask = (1.0 - group_time_mask) * torch.finfo(torch.float32).min
            
            hidden_states = torch.randn(B, T, config.d_model, device=device, dtype=torch.float32)
            
            base_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": group_time_mask
            }
            
            # Check correctness
            # We must set seed to ensure same weights or just copy weights
            
            base_time = benchmark_module(base_layer, base_kwargs)
            results_base[case_name].append(base_time)

            # 2. BSR SparseGroupMHA
            bsr_layer = GroupSelfAttention(config).eval().to(device)
            # copy weights
            bsr_layer.load_state_dict(base_layer.state_dict())
            
            sparse_mha = SparseGroupMHA(config, group_ids=group_ids).eval().to(device)
            sparse_mha.load_state_dict(bsr_layer.self_attention.state_dict())
            bsr_layer.self_attention = sparse_mha
            
            # The mask argument is ignored by SparseGroupMHA, but required by GroupSelfAttention signature
            bsr_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": group_time_mask
            }
            
            bsr_time = benchmark_module(bsr_layer, bsr_kwargs)
            results_bsr[case_name].append(bsr_time)

            speedup = base_time / bsr_time
            speedups[case_name].append(speedup)
            
            print(f"    Base: {base_time:.3f} ms | BSR: {bsr_time:.3f} ms | Speedup: {speedup:.2f}x")

    # Plotting
    os.makedirs("artifacts/bsr_layer_benchmark", exist_ok=True)
    
    # 1. Latency Plot
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, case in enumerate(cases):
        plt.plot(batch_sizes, results_base[case], marker='o', linestyle='--', color=colors[i], alpha=0.5, label=f"{case} (Base)")
        plt.plot(batch_sizes, results_bsr[case], marker='s', color=colors[i], linewidth=2, label=f"{case} (BSR)")
        
    plt.xlabel("Batch Size (N_VARIATES)")
    plt.ylabel("Latency (ms)")
    plt.title(f"GroupSelfAttention Layer: Base SDPA vs Triton BSR (T={T})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/bsr_layer_benchmark/bsr_layer_latency.png")
    plt.close()

    # 2. Speedup Plot
    plt.figure(figsize=(10, 6))
    for i, case in enumerate(cases):
        plt.plot(batch_sizes, speedups[case], marker='^', color=colors[i], linewidth=2, label=case)
        
    plt.axhline(1.0, color='black', linestyle='--', label="Baseline (1.0x)")
    plt.xlabel("Batch Size (N_VARIATES)")
    plt.ylabel("Speedup (Base Time / BSR Time)")
    plt.title(f"GroupSelfAttention Layer BSR Speedup (T={T})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/bsr_layer_benchmark/bsr_layer_speedup.png")
    plt.close()

    print("\nLayer benchmarks complete. Plots saved to 'artifacts/bsr_layer_benchmark/' directory.")

if __name__ == "__main__":
    main()
