import torch
from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.optimized_layers import SparseGroupMHA

device = "cuda"
T = 64
B = 128
config = Chronos2CoreConfig(d_model=512, d_kv=64, num_heads=8, dropout_rate=0.0)

# Univariate
group_ids = torch.arange(B, dtype=torch.long, device=device)

sparse_mha = SparseGroupMHA(config, group_ids).to(device)

hidden_states = torch.randn(T, B, config.d_model, device=device)

try:
    out = sparse_mha(hidden_states, mask=None)
    print("Success!")
except Exception as e:
    print("Error:", e)
