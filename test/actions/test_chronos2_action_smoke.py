import importlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.modeling_chronos2 import Chronos2Model
from chop.tools.get_input import ModelSource
from chop.tools.plt_wrapper.timeseries import TimeSeriesForecastingModelWrapper


class _ToyChronosForecastDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        base = float(idx + 1)
        return {
            "past_values": torch.full((16,), base, dtype=torch.float32),
            "future_values": torch.full((8,), base + 0.5, dtype=torch.float32),
        }


class _ToyChronosDataModule:
    def __init__(self):
        self.batch_size = 2
        self.name = "toy-forecast"
        self.model_name = "chronos-2"
        self.train_dataset = _ToyChronosForecastDataset()
        self.val_dataset = _ToyChronosForecastDataset()
        self.prepare_called = False
        self.setup_called = False

    def prepare_data(self):
        self.prepare_called = True

    def setup(self, stage=None):
        self.setup_called = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def _small_chronos2_model() -> Chronos2Model:
    config = Chronos2CoreConfig(
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        attn_implementation="eager",
        chronos_config={
            "context_length": 16,
            "input_patch_size": 4,
            "input_patch_stride": 4,
            "output_patch_size": 4,
            "quantiles": [0.1, 0.5, 0.9],
            "use_reg_token": False,
            "use_arcsinh": False,
        },
    )
    return Chronos2Model(config).eval()


def _model_info():
    return SimpleNamespace(
        model_source=ModelSource.HF_TRANSFORMERS,
        is_vision_model=False,
        is_physical_model=False,
        is_nerf_model=False,
        is_nlp_model=False,
        is_timeseries_model=True,
        name="chronos-2",
    )


def _dataset_info():
    return SimpleNamespace(num_classes=None)


def test_train_action_smoke_for_chronos2(monkeypatch, tmp_path):
    train_mod = importlib.import_module("chop.actions.train")

    captured = {}

    class _FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def fit(self, pl_model, datamodule):
            captured["pl_model"] = pl_model
            captured["datamodule"] = datamodule

    def _fake_load_model(load_name, load_type, model):
        captured["load_name"] = load_name
        captured["load_type"] = load_type
        return model

    def _fake_save_mase_graph(graph, pass_args=None):
        captured["saved_graph"] = graph
        captured["save_dir"] = Path(pass_args)
        return graph, {}

    monkeypatch.setattr(train_mod.pl, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train_mod, "load_model", _fake_load_model)
    monkeypatch.setattr(train_mod, "save_mase_graph_interface_pass", _fake_save_mase_graph)

    data_module = _ToyChronosDataModule()
    data_module.prepare_data()
    data_module.setup()

    train_mod.train(
        model=_small_chronos2_model(),
        model_info=_model_info(),
        data_module=data_module,
        dataset_info=_dataset_info(),
        task="forecasting",
        optimizer="adam",
        learning_rate=1e-4,
        weight_decay=0.0,
        scheduler_args={},
        plt_trainer_args={"max_epochs": 1, "accelerator": "cpu"},
        auto_requeue=False,
        save_path=tmp_path / "training",
        visualizer=None,
        load_name="dummy.mz",
        load_type="mz",
    )

    assert isinstance(captured["pl_model"], TimeSeriesForecastingModelWrapper)
    assert captured["datamodule"] is data_module
    assert hasattr(captured["saved_graph"].model, "chronos_config")
    assert captured["save_dir"].name == "transformed_ckpt"


def test_transform_action_smoke_for_chronos2(monkeypatch, tmp_path):
    transform_mod = importlib.import_module("chop.actions.transform")

    config_path = tmp_path / "chronos2_transform.toml"
    config_path.write_text(
        "[transform]\nstyle = \"graph\"\n\n[passes.report_node_type]\n",
        encoding="utf-8",
    )

    captured = {}

    def _fake_save_mase_graph(graph, pass_args=None):
        captured["saved_graph"] = graph
        captured["save_dir"] = Path(pass_args)
        return graph, {}

    monkeypatch.setattr(transform_mod, "save_mase_graph_interface_pass", _fake_save_mase_graph)

    data_module = _ToyChronosDataModule()
    data_module.prepare_data()
    data_module.setup()

    transform_mod.transform(
        model=_small_chronos2_model(),
        model_info=_model_info(),
        model_name="chronos-2",
        data_module=data_module,
        task="forecasting",
        config=str(config_path),
        save_dir=tmp_path / "transform",
        load_name=None,
        load_type=None,
        accelerator="cpu",
    )

    assert hasattr(captured["saved_graph"].model, "chronos_config")
    assert captured["save_dir"].name == "transformed_ckpt"


def test_search_action_smoke_for_chronos2(monkeypatch, tmp_path):
    search_mod = importlib.import_module("chop.actions.search.search")

    captured = {}

    class _FakeSearchSpace:
        def __init__(self, **kwargs):
            captured["search_space_kwargs"] = kwargs
            self.built = False

        def build_search_space(self):
            self.built = True

    class _FakeStrategy:
        def __init__(self, **kwargs):
            captured["strategy_kwargs"] = kwargs

        def search(self, search_space):
            captured["searched"] = True
            captured["search_space"] = search_space
            assert search_space.built

    monkeypatch.setattr(search_mod, "get_search_space_cls", lambda name: _FakeSearchSpace)
    monkeypatch.setattr(search_mod, "get_search_strategy_cls", lambda name: _FakeStrategy)

    data_module = _ToyChronosDataModule()
    search_mod.search(
        model=_small_chronos2_model(),
        model_info=_model_info(),
        task="forecasting",
        dataset_info=_dataset_info(),
        data_module=data_module,
        search_config={
            "search": {
                "strategy": {"name": "dummy"},
                "search_space": {"name": "dummy"},
            }
        },
        save_path=tmp_path / "search",
        accelerator="cpu",
        load_name=None,
        load_type=None,
        visualizer=None,
    )

    dummy_input = captured["search_space_kwargs"]["dummy_input"]
    assert data_module.prepare_called
    assert data_module.setup_called
    assert dummy_input["context"].shape == (1, 16)
    assert dummy_input["future_target"].shape == (1, 8)
    assert dummy_input["num_output_patches"] == 2
    assert captured["searched"]
