# Chronos-2 Fused Time-Group Attention Kernel Findings

## 1. Correctness Compared to Non-Fused Base
The initial fused version's output was **mathematically incorrect**, leading to a spike in the MASE error from `4.79` (baseline) to `25.44` during profiling. 
This was because the custom Triton kernel (`triton_attention.py`) did not accept an `attention_mask` or `group_time_mask`. In the Chronos-2 implementation, these masks are critical for applying causal masking (Time Attention) and isolating separate time series (Group Attention). Because the Triton kernel ignored these masks, the model attended to invalid and future tokens, destroying the validity of the predictions.

## 2. Why the Fused Kernel was so Slow
The massive slowdown (3.36 seconds per sample vs 0.0084 seconds) was caused by a combination of the `MaseGraph` FX tracing mechanism and memory layout inefficiencies:

* **The `num_output_patches` Hardcoding Bug (FX Tracing):** When `MaseGraph` traces the model, it hardcodes the default argument `num_output_patches=1` into the FX graph. When `predict_fev` requested a horizon of 24 (which requires 2 patches of 16), the traced model ignored the requested `num_output_patches=2` and only generated 1 patch. This forced `Chronos2Pipeline` into a highly unoptimized fallback loop (`_autoregressive_unroll_for_long_horizon`) that iteratively re-evaluated the entire context sequence multiple times per sample.
* **Uncoalesced Memory (Triton):** The Triton kernel ran slowly for these shapes because it received transposed, non-contiguous `q, k, v` tensors. For sequence lengths matching the batch size (`N_CTX=batch` in group attention), Triton is noticeably slower than PyTorch's native `F.scaled_dot_product_attention` (SDPA), which leverages highly optimized paths in cuBLAS for non-standard memory layouts.

## 3. Code Organization & Architecture
The FX transform logic in `fused_time_group_attention.py` is structurally sound. It successfully identifies the `TimeSelfAttention -> GroupSelfAttention` pattern, creates the `FusedTimeGroupAttention` layer, copies weights correctly, and patches the `kwargs` so the new node receives both masks.

However, the implementation inside `FusedTimeGroupAttention` (in `src/chop/models/chronos2/layers.py`) should use PyTorch's native SDPA instead of the custom Triton kernel to properly support masking and benefit from native optimizations.

## 4. Actions Taken
* `src/chop/models/chronos2/layers.py` was updated to replace `triton_flash_attention` with PyTorch's `torch.nn.functional.scaled_dot_product_attention`. This guarantees that `attention_mask` and `group_time_mask` are applied appropriately and allows PyTorch to optimize the memory layout under the hood, matching the mathematical behavior of the baseline model.

## 5. Next Steps
* To fully resolve the remaining prediction speed issue when tracing, you will need to add `"num_output_patches"` to the `hf_input_names` kwargs when creating the `MaseGraph` or explicitly avoid FX tracing during the dynamic autoregressive generation loops.