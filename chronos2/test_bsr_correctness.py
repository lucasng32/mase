import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from chop.models import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "amazon/chronos-2"

def generate_test_data(batch_size: int, context_length: int, num_groups: int):
    context = torch.randn(batch_size, context_length, dtype=torch.bfloat16, device=DEVICE)
    context_mask = torch.ones(batch_size, context_length, dtype=torch.bfloat16, device=DEVICE)
    
    group_size = max(1, batch_size // num_groups)
    group_ids = torch.arange(num_groups, device=DEVICE).repeat_interleave(group_size)
    if len(group_ids) < batch_size:
        pad = torch.full((batch_size - len(group_ids),), num_groups - 1, device=DEVICE)
        group_ids = torch.cat([group_ids, pad])
        
    return context, context_mask, group_ids

def run_correctness_test():
    print("Loading Models...")
    # Dense baseline
    dense_model = get_model("chronos-2", pretrained=True, model_id=MODEL_ID, group_attn_backend="dense", _attn_implementation="eager")
    dense_model = dense_model.to(DEVICE).to(torch.bfloat16)
    dense_model.eval()

    # Sparse BSR
    sparse_model = get_model("chronos-2", pretrained=True, model_id=MODEL_ID, group_attn_backend="bsr", _attn_implementation="sdpa")
    sparse_model = sparse_model.to(DEVICE).to(torch.bfloat16)
    sparse_model.eval()

    scenarios = [
        ("Independent", 64, 64),
        ("Pairs", 64, 32),
        ("Large Group", 64, 1),
    ]
    
    for name, b_size, n_groups in scenarios:
        print(f"\n--- Scenario: {name} (Batch: {b_size}, Groups: {n_groups}) ---")
        context, context_mask, group_ids = generate_test_data(b_size, 1440, n_groups)
        
        with torch.no_grad():
            out_dense = dense_model(context=context, context_mask=context_mask, group_ids=group_ids).quantile_preds
            out_sparse = sparse_model(context=context, context_mask=context_mask, group_ids=group_ids).quantile_preds
            
        diff_sparse = torch.abs(out_dense - out_sparse)
        
        print(f"Sparse vs Dense -> Max Diff: {diff_sparse.max().item():.4f}, Mean Diff: {diff_sparse.mean().item():.4f}")
        
if __name__ == "__main__":
    run_correctness_test()
