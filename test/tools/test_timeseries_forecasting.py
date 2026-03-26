from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from chop.tools.get_input import get_dummy_input, get_hf_input_names
from chop.tools.plt_wrapper.timeseries import TimeSeriesForecastingModelWrapper


class _ToyForecastDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        base = float(idx + 1)
        return {
            "past_values": torch.full((8,), base),
            "future_values": torch.full((6,), base + 0.5),
        }


class _ToyDataModule:
    def __init__(self):
        self.batch_size = 2
        self.name = "toy-forecast"
        self.model_name = "chronos-2"
        self.train_dataset = _ToyForecastDataset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)


class _ToyChronosModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.chronos_config = SimpleNamespace(output_patch_size=4)

    def forward(self, context, future_target, **kwargs):
        prediction = future_target.unsqueeze(1).repeat(1, 3, 1) * self.scale
        loss = torch.mean((prediction[:, 1] - future_target) ** 2)
        return {"loss": loss, "quantile_preds": prediction}


def _model_info():
    return SimpleNamespace(
        is_vision_model=False,
        is_physical_model=False,
        is_nerf_model=False,
        is_nlp_model=False,
        is_timeseries_model=True,
        name="chronos-2",
    )


def test_get_hf_input_names_for_timeseries():
    input_names = get_hf_input_names(_model_info(), "forecasting")
    assert input_names == [
        "context",
        "group_ids",
        "future_covariates",
        "num_output_patches",
    ]


def test_get_dummy_input_for_timeseries():
    dummy_input = get_dummy_input(
        model_info=_model_info(),
        data_module=_ToyDataModule(),
        task="forecasting",
        device="cpu",
        model=_ToyChronosModel(),
    )
    assert dummy_input["context"].shape == (1, 8)
    assert dummy_input["future_target"].shape == (1, 6)
    assert dummy_input["num_output_patches"] == 2


def test_timeseries_wrapper_training_step():
    wrapper = TimeSeriesForecastingModelWrapper(
        model=_ToyChronosModel(),
        dataset_info=SimpleNamespace(num_classes=None),
        optimizer="adam",
    )
    batch = _ToyForecastDataset()[0]
    batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    loss = wrapper.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
