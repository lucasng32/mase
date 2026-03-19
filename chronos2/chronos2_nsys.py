import os
import math
import time
import warnings
from pathlib import Path
import inspect
from types import SimpleNamespace
import pandas as pd

import torch
import numpy as np
from torch.utils.data import DataLoader

# Assuming these are available in your environment
from chop.models.chronos2.pipeline import Chronos2Pipeline
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from chop.models import get_model
import fev
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms.fused_time_group_attention import fused_time_group_attention_pass

warnings.filterwarnings('ignore')
os.environ.setdefault("HOME", os.environ.get("USERPROFILE", str(Path.home())))

# ── 1. Config ─────────────────────────────────────────────────────────────────
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN          = 1440   # context length per time series
N_FEATURES       = 20     # number of variates per sample
FORECAST_HORIZON = 96     # prediction length
BATCH_SIZES      = [20, 40, 60, 80, 100]  # number of multivariate samples per batch
WARMUP_RUNS      = 3
BENCH_RUNS       = 5
MODEL_ID         = 'amazon/chronos-2'

print(f'Device: {DEVICE}')
print(f'Sequence length: {SEQ_LEN}')
print(f'Number of features: {N_FEATURES}')
print(f'Forecast horizon: {FORECAST_HORIZON}')

# ── 2. Load Model ─────────────────────────────────────────────────────────────
model = get_model("chronos-2", pretrained=True, model_id=MODEL_ID)
model = model.to(DEVICE).to(torch.bfloat16)
pipeline = Chronos2Pipeline(model=model)
model.eval()

OUTPUT_PATCH_SIZE = model.chronos_config.output_patch_size
NUM_OUTPUT_PATCHES = math.ceil(FORECAST_HORIZON / OUTPUT_PATCH_SIZE)
FUTURE_LEN = NUM_OUTPUT_PATCHES * OUTPUT_PATCH_SIZE

print(f'\n✓ Model loaded successfully from {MODEL_ID}')
print(f'  Model type:        {type(model).__name__}')
print(f'  Output patch size: {OUTPUT_PATCH_SIZE}')
print(f'  Num output patches for horizon={FORECAST_HORIZON}: {NUM_OUTPUT_PATCHES}')

# ── 3. Benchmark Function ─────────────────────────────────────────────────────
def fev_bench_inference_time(model, model_name, task_configs, batch_sizes, output_dir="artifacts"):
    # Define benchmark
    benchmark = fev.Benchmark.from_list(task_configs)

    # Run benchmark for each model and batch size
    summaries = []
    inference_times = {}
    print(f'\n=== Benchmarking model: {model_name} ===')

    # Create pipeline for the model
    pipeline = Chronos2Pipeline(model=model)

    for task in benchmark.tasks:
        print(f'\n--- Task: {task.task_name} ---')
        inference_times[task.task_name] = {}

        for batch_size in batch_sizes:
            predictions, inference_time = pipeline.predict_fev(task, batch_size)
            summary = task.evaluation_summary(predictions, model_name)
            summaries.append(summary)

            inference_times[task.task_name][batch_size] = inference_time
            
            inference_time_per_sample = inference_time / batch_size
            print(f'Batch size: {batch_size} | Inference time per sample: {inference_time_per_sample:.4f} sec')

    # Save results to CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    summaries_df = pd.DataFrame(summaries)
    summaries_path = Path(output_dir) / f"{model_name}_summaries.csv"
    summaries_df.to_csv(summaries_path, index=False)
    print(f'\nSaved summaries → {summaries_path}')

    inference_rows = []
    for task_name, batch_dict in inference_times.items():
        for batch_size, t in batch_dict.items():
            inference_rows.append({
                "task": task_name,
                "batch_size": batch_size,
                "inference_time_s": t,
                "inference_time_per_sample_s": t / batch_size,
            })
    inference_df = pd.DataFrame(inference_rows)
    inference_path = Path(output_dir) / f"{model_name}_inference_times.csv"
    inference_df.to_csv(inference_path, index=False)
    print(f'Saved inference times → {inference_path}')

    return summaries_df, inference_df

# ── 4. Run Baseline Benchmark ─────────────────────────────────────────────────
tasks_configs = [
    {
        "dataset_path": "autogluon/chronos_datasets",
        "dataset_config": "m4_hourly",
        "horizon": 24,
    },
]

print("\nStarting Baseline Benchmark...")
# NVTX Marker for Baseline
torch.cuda.nvtx.range_push("Chronos2_Baseline_Inference")
fev_bench_inference_time(
    model=model,
    model_name="Baseline",
    task_configs=tasks_configs,
    batch_sizes=BATCH_SIZES,
)
torch.cuda.nvtx.range_pop()