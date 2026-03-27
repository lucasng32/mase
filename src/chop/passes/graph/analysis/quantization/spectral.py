from collections import defaultdict
from itertools import islice
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from chop.ir.graph import MaseGraph


def _compute_band_energies(x: torch.Tensor, num_bands: int) -> torch.Tensor:
    x = x.detach().to(torch.float32)
    if x.numel() == 0:
        return torch.zeros(num_bands, dtype=torch.float32)

    if x.ndim == 0:
        return torch.cat([x.abs().reshape(1), torch.zeros(max(0, num_bands - 1))])[:num_bands]

    if x.shape[-1] < 2:
        return torch.cat([x.abs().mean().reshape(1), torch.zeros(max(0, num_bands - 1))])[:num_bands]

    flat = x.reshape(-1, x.shape[-1])
    spectrum = torch.fft.rfft(flat, dim=-1).abs().mean(dim=0)
    if spectrum.numel() == 0:
        return torch.zeros(num_bands, dtype=torch.float32)

    chunk_size = max(1, math.ceil(spectrum.numel() / num_bands))
    bands = []
    for band_idx in range(num_bands):
        start = band_idx * chunk_size
        end = min((band_idx + 1) * chunk_size, spectrum.numel())
        if start >= spectrum.numel():
            bands.append(torch.tensor(0.0, dtype=torch.float32))
        else:
            bands.append(spectrum[start:end].mean())
    return torch.stack(bands)


def _fake_quantize_integer(x: torch.Tensor, width: int, frac_width: int) -> torch.Tensor:
    scale = float(2**frac_width)
    qmin = -(2 ** (width - 1))
    qmax = (2 ** (width - 1)) - 1
    quantized = torch.round(x * scale).clamp(qmin, qmax)
    return quantized / scale


def _spectral_distance(reference: torch.Tensor, quantized: torch.Tensor, num_bands: int) -> float:
    ref_bands = _compute_band_energies(reference, num_bands=num_bands)
    q_bands = _compute_band_energies(quantized, num_bands=num_bands)
    denom = ref_bands.abs().sum().item() or 1.0
    return float(torch.abs(ref_bands - q_bands).sum().item() / denom)


def _extract_tensor(maybe_tensor):
    if isinstance(maybe_tensor, (tuple, list)):
        return maybe_tensor[0]
    return maybe_tensor


def _fake_quantized_linear_output(
    module: nn.Linear,
    input_sample: torch.Tensor,
    width: int,
    data_frac_width: int,
    weight_frac_width: int,
    bias_width: int,
    bias_frac_width: int,
) -> torch.Tensor:
    input_q = _fake_quantize_integer(input_sample.detach().cpu().to(torch.float32), width=width, frac_width=data_frac_width)
    weight_q = _fake_quantize_integer(
        module.weight.detach().cpu().to(torch.float32), width=width, frac_width=weight_frac_width
    )

    bias = module.bias
    bias_q = None
    if bias is not None:
        bias_q = _fake_quantize_integer(
            bias.detach().cpu().to(torch.float32), width=bias_width, frac_width=bias_frac_width
        )

    return F.linear(input_q, weight_q, bias_q)


def profile_spectral_statistics_analysis_pass(graph: MaseGraph, pass_args: dict | None = None):
    pass_args = pass_args or {}
    calibration_batches = pass_args["calibration_batches"]
    max_batches = int(pass_args.get("max_batches", 8))
    num_bands = int(pass_args.get("num_bands", 8))
    max_samples_per_layer = int(pass_args.get("max_samples_per_layer", 4))
    target_layers = set(pass_args.get("target_layers", []))

    module_targets = []
    for node in graph.nodes:
        if node.op != "call_module":
            continue
        module_name = str(node.target)
        module = graph.modules.get(module_name)
        if not isinstance(module, nn.Linear):
            continue
        if target_layers and module_name not in target_layers:
            continue
        module_targets.append((module_name, module))

    stats = {
        module_name: {
            "band_energy_sum": torch.zeros(num_bands, dtype=torch.float32),
            "num_observations": 0,
            "max_abs": 0.0,
            "samples": [],
            "input_samples": [],
            "output_samples": [],
        }
        for module_name, _ in module_targets
    }

    hooks = []
    for module_name, module in module_targets:
        def _hook(_module, _inputs, output, module_name=module_name):
            input_tensor = _extract_tensor(_inputs)
            output_tensor = _extract_tensor(output)

            input_tensor = input_tensor.detach().cpu().to(torch.float32)
            output_tensor = output_tensor.detach().cpu().to(torch.float32)

            stats[module_name]["band_energy_sum"] += _compute_band_energies(output_tensor, num_bands=num_bands)
            stats[module_name]["num_observations"] += 1
            stats[module_name]["max_abs"] = max(
                stats[module_name]["max_abs"],
                float(max(input_tensor.abs().max().item(), output_tensor.abs().max().item())),
            )
            if len(stats[module_name]["samples"]) < max_samples_per_layer:
                stats[module_name]["samples"].append(output_tensor)
                stats[module_name]["input_samples"].append(input_tensor)
                stats[module_name]["output_samples"].append(output_tensor)

        hooks.append(module.register_forward_hook(_hook))

    was_training = graph.model.training
    graph.model.eval()
    with torch.no_grad():
        for batch in islice(iter(calibration_batches), max_batches):
            graph.model(**batch)

    for hook in hooks:
        hook.remove()

    if was_training:
        graph.model.train()

    spectral_stats = {}
    for module_name, values in stats.items():
        observations = max(1, values["num_observations"])
        spectral_stats[module_name] = {
            "mean_band_energy": values["band_energy_sum"] / observations,
            "num_observations": values["num_observations"],
            "max_abs": values["max_abs"],
            "samples": values["samples"],
            "input_samples": values["input_samples"],
            "output_samples": values["output_samples"],
        }

    return graph, {"spectral_stats": spectral_stats}


def spectral_calibrate_quantization(
    graph: MaseGraph,
    spectral_stats: dict,
    target_layers: list[str] | None = None,
    width: int = 8,
    bias_width: int = 8,
    default_frac_width: int = 4,
    frac_width_candidates: list[int] | None = None,
    num_bands: int = 8,
):
    frac_width_candidates = frac_width_candidates or list(range(width))
    target_layers = set(target_layers or spectral_stats.keys())

    frac_width_by_module = {}
    for module_name, layer_stats in spectral_stats.items():
        if module_name not in target_layers:
            continue
        module = graph.modules.get(module_name)
        if not isinstance(module, nn.Linear):
            continue

        input_samples = layer_stats.get("input_samples", [])
        output_samples = layer_stats.get("output_samples", [])
        if not input_samples or not output_samples:
            continue

        best_data_frac_width = default_frac_width
        best_weight_frac_width = default_frac_width
        best_output_score = float("inf")
        best_bias_frac_width = min(default_frac_width, bias_width - 1)
        for data_frac_width in frac_width_candidates:
            for weight_frac_width in frac_width_candidates:
                bias_frac_width = min(weight_frac_width, bias_width - 1)
                score = 0.0
                for input_sample, output_sample in zip(input_samples, output_samples, strict=False):
                    quantized_output = _fake_quantized_linear_output(
                        module,
                        input_sample=input_sample,
                        width=width,
                        data_frac_width=data_frac_width,
                        weight_frac_width=weight_frac_width,
                        bias_width=bias_width,
                        bias_frac_width=bias_frac_width,
                    )
                    score += _spectral_distance(output_sample, quantized_output, num_bands=num_bands)
                score /= len(input_samples)
                if score < best_output_score:
                    best_output_score = score
                    best_data_frac_width = data_frac_width
                    best_weight_frac_width = weight_frac_width
                    best_bias_frac_width = bias_frac_width

        frac_width_by_module[module_name] = {
            "data_in_frac_width": best_data_frac_width,
            "weight_frac_width": best_weight_frac_width,
            "bias_frac_width": best_bias_frac_width,
            "data_in_width": width,
            "weight_width": width,
            "bias_width": bias_width,
            "output_spectral_score": best_output_score,
        }

    return frac_width_by_module
