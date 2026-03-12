from types import SimpleNamespace

import torch

from chop.passes.graph.analysis.quantization import (
    profile_spectral_statistics_analysis_pass,
    spectral_calibrate_quantization,
)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_a = torch.nn.Linear(8, 8)
        self.linear_b = torch.nn.Linear(8, 4)

    def forward(self, context, **kwargs):
        hidden = self.linear_a(context)
        output = self.linear_b(hidden)
        return {"quantile_preds": output}


def _fake_graph():
    model = _TinyModel()
    modules = dict(model.named_modules())
    nodes = [
        SimpleNamespace(op="call_module", target="linear_a"),
        SimpleNamespace(op="call_module", target="linear_b"),
    ]
    return SimpleNamespace(model=model, modules=modules, nodes=nodes)


def test_profile_spectral_statistics_collects_samples():
    graph = _fake_graph()
    calibration_batches = [
        {"context": torch.randn(2, 8)},
        {"context": torch.randn(2, 8)},
    ]
    _, result = profile_spectral_statistics_analysis_pass(
        graph,
        {
            "calibration_batches": calibration_batches,
            "max_batches": 2,
            "num_bands": 4,
        },
    )
    assert "linear_a" in result["spectral_stats"]
    assert result["spectral_stats"]["linear_a"]["num_observations"] > 0


def test_spectral_calibrate_quantization_returns_frac_widths():
    graph = _fake_graph()
    spectral_stats = {
        "linear_a": {
            "samples": [torch.randn(2, 8), torch.randn(2, 8)],
        },
        "linear_b": {
            "samples": [torch.randn(2, 4)],
        },
    }
    config = spectral_calibrate_quantization(graph, spectral_stats, width=8)
    assert "linear_a" in config
    assert "data_in_frac_width" in config["linear_a"]
