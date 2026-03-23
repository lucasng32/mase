# Chronos-2 Group Attention: Architecture & Optimisation Notes

## 1. Chronos-2 Architecture Overview

Chronos-2 is a **pure encoder** model (no decoder). The entire sequence — past history
plus future slots — is concatenated and fed into a single transformer encoder in one shot.

```
Input per series: [past patches | REG token | future patches]
                        ↓              ↓            ↓
              ┌──────────────────────────────────────────┐
              │  Encoder (N layers)                       │
              │    1. TimeSelfAttention   (temporal axis) │
              │    2. GroupSelfAttention  (batch axis)    │
              │    3. FeedForward                         │
              └──────────────────────────────────────────┘
                              ↓ (future token hidden states)
                       Quantile head → predictions
```

**Patching:** the raw time series is chunked into fixed-size patches (16 timesteps each).
Each patch becomes one token. The encoder sees
`num_context_patches + 1 (REG) + num_output_patches` tokens per series.

Each token has 3 components concatenated: `[time_encoding | values | mask]`.

---

## 2. What `group_ids` Is

`group_ids` is a 1-D integer tensor of shape `(batch_size,)` — **one label per series,
not per patch**. It encodes which series in the flat batch belong to the same forecasting
task.

```python
group_ids = [0, 0, 1, 1, 1, 2]
#             ^^^^  task 0    ^^^^^ task 1   ^^ task 2
```

- Series with the **same** ID form a group and can attend to each other.
- Series with **different** IDs are completely isolated — attention scores between them
  are set to `-inf` before softmax.
- Only equality matters; the actual integer values are arbitrary.

---

## 3. Why Groups Are Needed

### The problem with sequence concatenation

The naive approach to multivariate forecasting is to concatenate all series patches into
one long sequence and run standard self-attention:

```
Series A: [A1, A2, A3, A4]
Series B: [B1, B2, B3, B4]
Concat:   [A1, A2, A3, A4, B1, B2, B3, B4]  ← 8 tokens, O(n²) attention
```

This lets Ai attend to all patches of B across all time positions. With 100 series of
8192 timesteps, the sequence becomes 819,200 tokens — prohibitively expensive.

### What GroupSelfAttention does instead

Each series keeps its own sequence. GroupSelfAttention operates **across the batch
dimension at each patch position**:

```
group_ids = [0, 0]

TimeSelfAttention:    A1↔A2↔A3↔A4    (each series attends along its own time axis)
                      B1↔B2↔B3↔B4

GroupSelfAttention:   A1↔B1,  A2↔B2,  A3↔B3,  A4↔B4
                      (same patch position only, across series)
```

Cross-series information is exchanged at matching patch positions, not across all
time positions. This is cheaper and more structured than full concatenation.

GroupSelfAttention has **no positional embeddings** — series within a group have no
natural ordering, so none is imposed.

### Batching multiple independent tasks

Groups also solve the GPU batching problem. Without them, independent tasks would need
separate forward passes. With groups, many tasks of different sizes can be packed into
one batch:

```
group_ids = [0, 0, 0, 1, 2, 2, 2, 2, 2, 3]

Task 0: series 0,1,2        (3-variate)
Task 1: series 3            (univariate)
Task 2: series 4,5,6,7,8   (5-variate)
Task 3: series 9            (univariate)
```

The group mask is a `(B, B)` block-diagonal boolean matrix:

```python
group_mask = group_ids[:, None] == group_ids[None, :]
```

Combined with the per-series `attention_mask` (valid vs padded patches) via einsum,
then inverted to `-inf` for blocked positions.

---

## 4. `future_covariates`

`future_covariates` fills in the **value + mask** components of the future patch tokens.

- **No covariates** (`NaN` or `None`): future patches have `values=0, mask=0`. The
  encoder sees unknown future slots and predicts them.
- **With covariates** (real values): for some series in the group, future values are
  known. The encoder sees real future values for those series and can use them — via
  GroupSelfAttention — to inform the prediction of target series in the same group.

The model does not treat covariates specially in the architecture. They are just series
in the same group whose future tokens happen to have `mask=1`. The distinction only
appears in the loss: `loss_mask = target_mask * inv_future_covariate_mask` — loss is
only computed where the future is **not** a known covariate.

**Example:**
```
group_ids = [0, 0]
future_covariates = [[NaN, NaN, ...],   # series 0 = target to forecast
                     [22.5, 23.1, ...]] # series 1 = known future (e.g. temperature)
```

The encoder uses series 1's known future to improve series 0's prediction via group
attention.

---

## 5. Common Inference Cases

| Case | `group_ids` | Group mask | Notes |
|---|---|---|---|
| Univariate | `arange(B)` | Identity (diagonal) | No cross-series attention; GroupSelfAttention is a no-op |
| Fully multivariate | `zeros(B)` | All-ones | Every series attends to every other |
| Fixed equal groups | `[0,0,1,1,...]` | Block-diagonal (equal blocks) | Structured datasets with known variate count |
| Heterogeneous | arbitrary | Block-diagonal (variable blocks) | Mixed tasks; mainly training/evaluation |

**For inference, univariate is the dominant case** (Chronos-2's zero-shot benchmark
suite, users migrating from Chronos-1). Fully multivariate is the second priority when
users explicitly want cross-series mixing. Heterogeneous batching is mainly a training
concern.

---

## 6. Are Group IDs Known Ahead of Time?

**Yes — always.** Group IDs are determined by task structure, not data values. By the
time you know what to feed the model, you already know the grouping:

| Scenario | When known |
|---|---|
| Single series | Model setup time — always `[0]` |
| Batch of independent series | Batch assembly — always `arange` |
| Multivariate dataset (fixed variates) | Dataset load time — always `zeros` or fixed pattern |
| Mixed heterogeneous batch | Batch assembly by DataLoader — from task metadata, not tensor values |

Even in the heterogeneous training case, group IDs are assigned from dataset metadata
before any tensor values are examined.

**Implication for optimisation:** the group mask sparsity pattern is a **compile-time
constant** for any fixed deployment scenario. A transformation pass can treat it as
static, enabling:

- Pre-computing and caching the group mask as a constant buffer
- Replacing GroupSelfAttention with a sparse kernel matched to the known sparsity
  pattern (e.g. block-diagonal for fixed group sizes)
- For the univariate case: the group mask is the identity — GroupSelfAttention
  degenerates to each series attending only to itself and could be entirely skipped
- Fusing or eliminating the `einsum` + invert steps when the mask is constant

The only dynamic component on top of the static group structure is the `attention_mask`
(which patches have valid, non-NaN values). If the input data is assumed complete (no
missing values), even `group_time_mask` becomes fully static.
