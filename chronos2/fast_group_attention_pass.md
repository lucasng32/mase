# FastGroupAttention: MASE Pass Implementation Plan

This document covers the implementation plan for replacing `GroupSelfAttention` in
Chronos-2 with an optimised kernel via a MASE graph transform pass.

For background on why `GroupSelfAttention` exists and how group IDs work, see
`group_attention_analysis.md`.

---

## 1. Problem

The current `GroupSelfAttention` (`layers.py:352`) computes the full `batch × batch`
attention matrix and masks out cross-group entries with `-inf`:

```python
# layers.py:367-375
hidden_states = hidden_states.transpose(0, 1)   # (time, batch, d_model)
normed = self.layer_norm(hidden_states)
attn_out = self.self_attention(normed, mask=attention_mask)  # full batch×batch matmul
hidden_states = hidden_states + self.dropout(attn_out[0])
hidden_states = hidden_states.transpose(0, 1)
```

For a batch of `B` independent series (`group_ids = arange(B)`), the group mask is
diagonal — every off-diagonal cell is `-inf`. The model still computes all `B²` dot
products, contributing zero information but consuming full compute and memory.

**At `B=32`, `seq=128`, `heads=8`, `d_kv=64`:** the group attention matmul is
`32 × 8 × 128 × 128 = 4M` dot products, of which only `32 × 128 × 128 / 32 = 128K`
(3%) are non-masked in the univariate case.

---

## 2. Strategy

Group IDs are known at batch assembly time — before any tensor values are examined.
This means the sparsity pattern of the group mask is a **runtime constant** per forward
pass, not a learned or dynamic quantity.

The optimisation strategy:

1. **Detect group structure** at dispatch time from `group_ids`
2. **Dispatch to a bucket-size kernel** that only computes attention within each group
3. **Pre-compile all bucket variants** at pass application time so production inference
   never triggers a Triton recompile

No recompilation in production regardless of what group IDs arrive, because:
- Group IDs change the *mask data*, not the *kernel structure*
- Bucket sizes (kernel `constexpr` parameters) are fixed at a small set pre-warmed at startup

---

## 3. Bucket Dispatch

Groups in a batch can have different sizes. A single kernel variant handles all groups
of size `<= BUCKET` by padding smaller groups to `BUCKET` with masked-out entries.

```
group_ids = [0, 0, 1, 1, 1, 2, 2, 2, 2]
group sizes: 2, 3, 4

BUCKETS = [1, 2, 4, 8, 16]
→ max group size = 4
→ dispatch to BUCKET=4 kernel

group [0,0]     → padded to [0,0,pad,pad]   (2/4 real)
group [1,1,1]   → padded to [1,1,1,pad]     (3/4 real)
group [2,2,2,2] → [2,2,2,2]                 (4/4 real)
```

Padded slots get `-inf` in the mask before softmax — they contribute nothing to output.

**Worst-case padding waste:** less than 2× for any group size, since each group always
falls within one bucket step. For the dominant univariate case (`BUCKET=1`), there is
zero padding waste and the kernel degenerates to an identity (each series attends only
to itself, which is a no-op).

---

## 4. File Layout

```
src/chop/
├── passes/graph/transforms/timeseries/
│   ├── FastGroupAtten.py          ← MASE pass (graph walker + module swap)
│   └── __init__.py
│
├── models/chronos2/
│   └── optimized_layers.py        ← FastGroupSelfAttention nn.Module
│                                     (contains Triton kernel + bucket dispatch)
```

---

## 5. `optimized_layers.py` — the replacement module

`FastGroupSelfAttention` is a drop-in replacement for `GroupSelfAttention`. It has the
same interface (`forward(hidden_states, attention_mask, output_attentions)`) but
internally dispatches to a bucket-specialized kernel.

```
FastGroupSelfAttention
├── __init__(config, buckets)
│     └── _prewarm_kernels()     ← compile all bucket variants at construction
├── forward(hidden_states, attention_mask)
│     ├── _get_bucket(group_ids) ← pick smallest bucket >= max_group_size
│     └── _run_kernel(bucket)    ← hits Triton cache, no recompile
└── [weight attributes match GroupSelfAttention for state_dict compatibility]
```

### Pre-warming

```python
def _prewarm_kernels(self):
    for bucket in self.buckets:
        dummy_h = torch.zeros(1, bucket, self.config.d_model, device="cuda")
        dummy_mask = torch.zeros(bucket, 1, 1, bucket, device="cuda")
        self._run_kernel(dummy_h, dummy_mask, bucket)
        # Triton JIT compiles and caches the binary for this bucket
```

Pre-warming runs once at pass application time. The compiled binaries are cached to
`~/.triton/cache/` and reused across process restarts.

### Kernel dispatch

```python
def forward(self, hidden_states, attention_mask, output_attentions=False):
    # group_ids derivable from attention_mask structure, or passed explicitly
    max_group = _max_group_size(attention_mask)
    bucket = next(b for b in self.buckets if b >= max_group)
    return self._run_kernel(hidden_states, attention_mask, bucket)
```

---

## 6. `FastGroupAtten.py` — the MASE pass

Follows the same pattern as `lora.py`: walk the `MaseGraph`, find `GroupSelfAttention`
nodes, swap in `FastGroupSelfAttention`, copy weights, recompile.

```python
def fast_group_attention_transform_pass(mg: MaseGraph, pass_args={}):
    buckets = pass_args.get("buckets", [1, 2, 4, 8, 16])

    for node in mg.nodes:
        target = deepgetattr(mg.model, node.target) if node.op == "call_module" else None

        if node.op == "call_module" and isinstance(target, GroupSelfAttention):
            new_module = FastGroupSelfAttention(
                config=target.self_attention.config,
                buckets=buckets,
            )
            # copy weights — state_dict keys match GroupSelfAttention
            new_module.load_state_dict(target.state_dict())

            deepsetattr(mg.model, node.target, new_module)

    mg.model.recompile()
    return mg, {}
```

The pass finds every `GroupSelfAttention` in every encoder layer and replaces all of
them in one shot. `Chronos2Model` has `num_layers` encoder blocks each containing one
`GroupSelfAttention`, so for a 6-layer model this replaces 6 modules.

---

## 7. Registering the pass

Add to `src/chop/passes/graph/__init__.py`:

```python
from .transforms.timeseries.FastGroupAtten import fast_group_attention_transform_pass

PASSES = {
    ...
    "fast_group_attention": fast_group_attention_transform_pass,
}
```

Then usable in any MASE config:

```toml
[passes.fast_group_attention]
buckets = [1, 2, 4, 8, 16]
```

---

## 8. Implementation Order

| Step | What | Why first |
|---|---|---|
| 1 | Implement `FastGroupSelfAttention` with plain PyTorch grouped matmuls (no Triton) | Validate correctness against `GroupSelfAttention` before touching kernels |
| 2 | Write the MASE pass and verify it finds and replaces all nodes correctly | Separate graph surgery concerns from kernel concerns |
| 3 | Benchmark plain PyTorch grouped matmul vs. current masked full-matrix | Quantify how much is recoverable before Triton |
| 4 | Write Triton kernel for the inner grouped attention with `BUCKET: tl.constexpr` | Only add Triton complexity after correctness and baseline perf are established |
| 5 | Add pre-warming and bucket dispatch | Final production-ready form |

---

## 9. Correctness Check

After applying the pass, the outputs should match the original to within floating point
tolerance:

```python
mg_original = MaseGraph(model)
mg_fast = fast_group_attention_transform_pass(deepcopy(mg_original), pass_args={})[0]

out_original = mg_original.model(context=x, group_ids=group_ids, ...)
out_fast     = mg_fast.model(context=x, group_ids=group_ids, ...)

assert torch.allclose(out_original.quantile_preds, out_fast.quantile_preds, atol=1e-5)
```

The univariate case (`group_ids = arange(B)`) is the simplest to test since the group
mask is diagonal and the expected behaviour is that each series attends only to itself.
