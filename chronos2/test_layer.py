import torch

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention, SparseGroupSelfAttention
from chop.models.chronos2.modeling_chronos2 import create_flattened_bsr_metadata

def test_single_layer():
    config = Chronos2CoreConfig(
        d_model=512, d_kv=64, d_ff=2048, num_layers=2, num_heads=8,
        use_sparse_group_attn=False, _attn_implementation="sdpa"
    )
    
    dense_layer = GroupSelfAttention(config).cuda()
    sparse_layer = SparseGroupSelfAttention(config).cuda()
    
    state_dict = dense_layer.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("self_attention."):
            new_state_dict[k.replace("self_attention.", "")] = v
        else:
            new_state_dict[k] = v
    sparse_layer.load_state_dict(new_state_dict)
    
    dense_layer.eval()
    sparse_layer.eval()
    
    # T=10, B=64, D=512
    T, B, D = 10, 64, 512
    hidden_states = torch.randn(B, T, D, device="cuda")
    
    # All in one group
    group_ids = torch.zeros(B, dtype=torch.long, device="cuda")
    attention_mask = torch.ones(B, T, dtype=torch.float32, device="cuda")
    
    # Dense Mask
    group_mask = group_ids[:, None] == group_ids[None, :]
    group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)
    group_time_mask = group_time_mask.permute(2, 0, 1).unsqueeze(1)
    group_time_mask = (1.0 - group_time_mask) * torch.finfo(torch.float32).min
    
    # Sparse Mask
    bsr_metadata = create_flattened_bsr_metadata(group_ids, T, block_size=16)
    
    with torch.no_grad():
        out_dense = dense_layer(hidden_states.clone(), attention_mask=group_time_mask).hidden_states
        out_sparse = sparse_layer(hidden_states.clone(), bsr_metadata=bsr_metadata).hidden_states
        
    diff = torch.abs(out_dense - out_sparse)
    print(f"Max Diff: {diff.max().item():.6f}, Mean Diff: {diff.mean().item():.6f}")

if __name__ == "__main__":
    test_single_layer()
