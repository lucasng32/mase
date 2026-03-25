import torch
import time
from chop.models import get_model
from chop.passes.graph.transforms.timeseries.FastBSRGroupAtten import fast_bsr_group_attention_transform_pass
from chop.ir.graph.mase_graph import MaseGraph
from chop.models.chronos2.layers import GroupSelfAttention

DEVICE = "cuda"

def run_bench(model, context, context_mask, group_ids, num_runs=5):
    # Warmup
    for _ in range(2):
        with torch.no_grad():
            model(context=context, context_mask=context_mask, group_ids=group_ids)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_runs):
        with torch.no_grad():
            model(context=context, context_mask=context_mask, group_ids=group_ids)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

def main():
    print("Loading SDPA Base Model...")
    base_model = get_model("chronos-2", pretrained=False)
    base_model = base_model.to(DEVICE).to(torch.bfloat16)
    base_model.eval()

    B = 256
    T = 1440
    print(f"Testing Batch Size: {B}, Context Length: {T}")
    context = torch.randn(B, T, dtype=torch.bfloat16, device=DEVICE)
    context_mask = torch.ones(B, T, dtype=torch.bfloat16, device=DEVICE)
    group_ids = torch.arange(B, device=DEVICE)  # Independent series

    print("Benchmarking Base (SDPA)...")
    base_time = run_bench(base_model, context, context_mask, group_ids)
    print(f"Base Time: {base_time:.2f} ms")

    import copy
    model_copy = copy.deepcopy(base_model)
    print("Tracing MaseGraph...")
    mg = MaseGraph(
        model_copy, 
        hf_input_names=["context", "context_mask", "group_ids"],
        custom_ops={"modules": {GroupSelfAttention: {"name": "GroupSelfAttention"}}}
    )
    mg, info = fast_bsr_group_attention_transform_pass(mg, pass_args={"group_ids": group_ids})
    
    print("Benchmarking BSR...")
    bsr_time = run_bench(mg.model, context, context_mask, group_ids)
    print(f"BSR Time: {bsr_time:.2f} ms")
    print(f"Speedup: {base_time / bsr_time:.2f}x")

main()
