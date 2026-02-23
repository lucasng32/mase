"""
MASE time-series dataset wrappers for Chronos-2 fine-tuning.

Loads datasets from ``autogluon/chronos_datasets`` on HuggingFace and follows
the standard MASE dataset pattern:

* ``prepare_data()``  – triggers HuggingFace download/cache (main process only)
* ``setup()``         – converts raw records into numpy arrays ready for indexing
* ``__len__`` / ``__getitem__`` – sliding-window PyTorch Dataset interface

Each ``__getitem__`` returns::

    {
        "past_values":   FloatTensor of shape (context_length,)
        "future_values": FloatTensor of shape (prediction_length,)
    }

HuggingFace record schema (autogluon/chronos_datasets)::

    {
        "id": "T1",
        "start":   "2016-07-01 00:00:00",
        "target":  [1.0, 2.5, 3.1, ...],
    }
"""

import logging
from typing import Optional

import numpy as np
import torch
import datasets as hf_datasets
from torch.utils.data import Dataset

from ..utils import add_dataset_info, DatasetSource, MaseDatasetInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HF split resolution
# ---------------------------------------------------------------------------

# For each MASE split, the ordered list of HF splits to try
# autogluon/chronos_datasets only publishes a "train" split on HuggingFace.
# All MASE splits therefore resolve to "train".
_SPLIT_FALLBACK: dict[str, list[str]] = {
    "train":      ["train"],
    "validation": ["train"],
    "test":       ["train"],
    "pred":       ["train"],
}


def _resolve_hf_split(hf_name: str, hf_config: Optional[str], mase_split: str) -> str:
    """Return the first HF split that exists for the dataset."""
    try:
        infos = hf_datasets.get_dataset_infos(hf_name, trust_remote_code=True)
        splits = (
            infos[hf_config].splits.keys()
            if hf_config
            else list(infos.values())[0].splits.keys()
        )
        available = set(splits)
    except Exception:
        available = {"train"}

    for candidate in _SPLIT_FALLBACK.get(mase_split, ["train"]):
        if candidate in available:
            return candidate

    return "train"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TimeSeriesBaseDataset(Dataset):
    """
    Base HuggingFace → PyTorch adapter for time-series datasets.

    Subclasses set ``hf_name``, ``hf_config``, and are decorated with
    ``@add_dataset_info``.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"validation"``, ``"test"``, ``"pred"``.
    context_length : int
        Past time steps provided as input.
    prediction_length : int
        Future time steps to predict.
    stride : int
        Step between successive sliding windows (1 = fully overlapping).
    auto_setup : bool
        Call ``prepare_data()`` and ``setup()`` at construction time.
        Set False when used inside ``MaseDataModule``.
    """

    info: MaseDatasetInfo = None  # attached by @add_dataset_info

    # --- subclass config ---
    hf_name: str = None    # HuggingFace dataset repo, e.g. "autogluon/chronos_datasets"
    hf_config: str = None  # HuggingFace config name, e.g. "m4_daily"

    def __init__(
        self,
        split: str,
        context_length: int = 512,
        prediction_length: int = 64,
        stride: int = 1,
        auto_setup: bool = True,
    ) -> None:
        super().__init__()

        assert split in ["train", "validation", "test", "pred"], (
            f"Unknown split '{split}'. Must be one of train/validation/test/pred."
        )
        assert self.hf_name is not None, (
            f"{self.__class__.__name__} must define 'hf_name'."
        )

        self.split = split
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.stride = stride

        # Populated by setup()
        self._series: list[np.ndarray] = []   # one 1-D float32 array per time series
        self._item_ids: list[str] = []
        self._windows: list[tuple[int, int]] = []  # (series_idx, start_idx)

        if auto_setup:
            self.prepare_data()
            self.setup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hf_split(self) -> str:
        return _resolve_hf_split(self.hf_name, self.hf_config, self.split)

    def _load_hf(self) -> hf_datasets.Dataset:
        kwargs: dict = {
            "path": self.hf_name,
            "split": self._hf_split(),
            "trust_remote_code": True,
        }
        if self.hf_config:
            kwargs["name"] = self.hf_config
        logger.info(
            "Loading '%s' (config=%s, hf_split=%s) for MASE split='%s'",
            self.hf_name, self.hf_config, kwargs["split"], self.split,
        )
        return hf_datasets.load_dataset(**kwargs)

    # ------------------------------------------------------------------
    # MASE dataset lifecycle
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Trigger HuggingFace download / cache. Safe to run on the main process only."""
        self._load_hf()

    def setup(self) -> None:
        """Load data into memory and build the sliding-window index."""
        hf_ds = self._load_hf()

        self._series = []
        self._item_ids = []
        self._windows = []

        for record in hf_ds:
            series = np.asarray(record["target"], dtype=np.float32)
            self._series.append(series)
            self._item_ids.append(str(record["id"]))

        # Build (series_idx, start_pos) window index
        for s_idx, series in enumerate(self._series):
            n = len(series)
            for start in range(0, n - self.window_size + 1, self.stride):
                self._windows.append((s_idx, start))

        logger.info(
            "%s split='%s': %d series → %d windows (ctx=%d, pred=%d, stride=%d)",
            self.__class__.__name__, self.split,
            len(self._series), len(self._windows),
            self.context_length, self.prediction_length, self.stride,
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        """Return a sliding window as a dict of tensors.

        Returns
        -------
        dict with keys:
            ``past_values``   – FloatTensor ``(context_length,)``
            ``future_values`` – FloatTensor ``(prediction_length,)``
        """
        s_idx, start = self._windows[idx]
        window = self._series[s_idx][start : start + self.window_size]
        return {
            "past_values":   torch.from_numpy(window[: self.context_length]),
            "future_values": torch.from_numpy(window[self.context_length :]),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split='{self.split}', "
            f"n_series={len(self._series)}, "
            f"n_windows={len(self._windows)}, "
            f"context={self.context_length}, pred={self.prediction_length})"
        )


# ---------------------------------------------------------------------------
# Concrete dataset classes  (autogluon/chronos_datasets on HuggingFace)
# ---------------------------------------------------------------------------

@add_dataset_info(
    name="chronos_m4_daily",
    dataset_source=DatasetSource.HF_DATASETS,
    available_splits=("train",),
    time_series_forecasting=True,
    prediction_length=14,
)
class ChronosM4DailyDataset(TimeSeriesBaseDataset):
    """M4 Daily — autogluon/chronos_datasets (m4_daily)."""
    hf_name = "autogluon/chronos_datasets"
    hf_config = "m4_daily"


@add_dataset_info(
    name="chronos_m4_weekly",
    dataset_source=DatasetSource.HF_DATASETS,
    available_splits=("train",),
    time_series_forecasting=True,
    prediction_length=13,
)
class ChronosM4WeeklyDataset(TimeSeriesBaseDataset):
    """M4 Weekly — autogluon/chronos_datasets (m4_weekly)."""
    hf_name = "autogluon/chronos_datasets"
    hf_config = "m4_weekly"


@add_dataset_info(
    name="chronos_m4_monthly",
    dataset_source=DatasetSource.HF_DATASETS,
    available_splits=("train",),
    time_series_forecasting=True,
    prediction_length=18,
)
class ChronosM4MonthlyDataset(TimeSeriesBaseDataset):
    """M4 Monthly — autogluon/chronos_datasets (m4_monthly)."""
    hf_name = "autogluon/chronos_datasets"
    hf_config = "m4_monthly"
