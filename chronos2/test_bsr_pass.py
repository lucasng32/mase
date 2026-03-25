import torch
import time
from chop.models import get_model
from chop.passes.graph.transforms.timeseries.FastBSRGroupAtten import fast_bsr_group_attention_transform_pass
import torch.fx as fx
from chop.ir.graph.mase_graph import MaseGraph

from chop.models.chronos2.layers import GroupSelfAttention

def test_bsr_pass():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available. BSR pass requires Triton and CUDA.")
        return

    print("Loading original Chronos-2 model...")
    model = get_model('chronos-2', pretrained=False)
    
    # We have to turn off SDPA to trace eager attention correctly sometimes
    if hasattr(model.config, '_attn_implementation'):
        model.config._attn_implementation = 'eager'
        
    model = model.eval().to(device)

    # Prepare inputs
    N_VARIATES = 32
    C_LEN = 128
    OUT_PATCH = 16
    
    group_ids = torch.cat([
        torch.zeros(16, dtype=torch.long),
        torch.ones(16, dtype=torch.long)
    ]).to(device)
    
    dummy_in = {
        'context': torch.randn((N_VARIATES, C_LEN), device=device),
        'group_ids': group_ids,
        'num_output_patches': OUT_PATCH,
    }

    print("Running base model for correctness...")
    with torch.no_grad():
        out_base = model(**dummy_in)
        
    print("Tracing MaseGraph...")
    # Trace the model
    # MaseGraph expects a dictionary of inputs to determine shapes
    mg = MaseGraph(
        model, 
        hf_input_names=list(dummy_in.keys()),
        custom_ops={
            "modules": {
                GroupSelfAttention: {"name": "GroupSelfAttention"}
            }
        }
    )

    print("Applying BSR Group Attention Transform Pass...")
    pass_args = {"group_ids": group_ids}
    mg, info = fast_bsr_group_attention_transform_pass(mg, pass_args=pass_args)
    print(f"Pass replaced {info['replaced']} GroupSelfAttention nodes.")

    print("Running BSR model for correctness...")
    with torch.no_grad():
        out_bsr = mg.model(**dummy_in)

    # Check correctness
    base_preds = out_base.quantile_preds if hasattr(out_base, 'quantile_preds') else out_base['quantile_preds']
    bsr_preds = out_bsr.quantile_preds if hasattr(out_bsr, 'quantile_preds') else (out_bsr['quantile_preds'] if isinstance(out_bsr, dict) else out_bsr[1])
    diff = (base_preds - bsr_preds).abs().max()
    print(f"Max difference between base and BSR: {diff.item()}")
    assert diff.item() < 1e-4, "BSR model output does not match base model output!"

    # Benchmark
    print("Benchmarking Base Model...")
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(**dummy_in)
            
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            model(**dummy_in)
    torch.cuda.synchronize()
    base_time = (time.time() - start) / 50
    print(f"Base Model Time: {base_time * 1000:.2f} ms")

    print("Benchmarking BSR Model...")
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            mg.model(**dummy_in)
            
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            mg.model(**dummy_in)
    torch.cuda.synchronize()
    bsr_time = (time.time() - start) / 50
    print(f"BSR Model Time: {bsr_time * 1000:.2f} ms")
    
    speedup = base_time / bsr_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    test_bsr_pass()
