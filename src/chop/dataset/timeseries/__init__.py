from .chronos_datasets import (
    TimeSeriesBaseDataset,
    ChronosM4DailyDataset,
    ChronosM4WeeklyDataset,
    ChronosM4MonthlyDataset,
)

# Registry: dataset name -> class
TIMESERIES_DATASET_MAPPING: dict = {
    "chronos_m4_daily":     ChronosM4DailyDataset,
    "chronos_m4_weekly":    ChronosM4WeeklyDataset,
    "chronos_m4_monthly":   ChronosM4MonthlyDataset,
}


def get_timeseries_dataset(
    name: str,
    split: str,
    context_length: int = 512,
    prediction_length: int = None,
    stride: int = 1,
    auto_setup: bool = True,
    custom_path: str = None,  # accepted for API compatibility; unused (data comes from HF)
) -> TimeSeriesBaseDataset:
    """Return an instantiated time-series dataset.

    Parameters
    ----------
    name : str
        Dataset name; must be a key in ``TIMESERIES_DATASET_MAPPING``.
    split : str
        One of ``"train"``, ``"validation"``, ``"test"``, ``"pred"``.
    context_length : int
        Number of past time steps passed as input (default: 512).
    prediction_length : int | None
        Number of future steps to predict. If None, uses the class default
        (i.e. the value set in its ``@add_dataset_info`` decorator).
    stride : int
        Step between successive sliding windows (default: 1).
    auto_setup : bool
        Automatically call ``prepare_data()`` + ``setup()`` at construction.
    custom_path : str | None
        Ignored; present only for compatibility with the generic ``get_dataset``
        dispatcher in ``dataset/__init__.py``.
    """
    assert name in TIMESERIES_DATASET_MAPPING, (
        f"Unknown timeseries dataset '{name}'. "
        f"Available: {list(TIMESERIES_DATASET_MAPPING.keys())}"
    )

    cls = TIMESERIES_DATASET_MAPPING[name]

    # Prefer explicit arg; fall back to class-level info if available
    pred_len = prediction_length
    if pred_len is None and cls.info is not None and cls.info.prediction_length is not None:
        pred_len = cls.info.prediction_length
    if pred_len is None:
        pred_len = 64  # sensible default

    return cls(
        split=split,
        context_length=context_length,
        prediction_length=pred_len,
        stride=stride,
        auto_setup=auto_setup,
    )


def get_timeseries_dataset_cls(name: str) -> type:
    """Return the dataset class (without instantiation)."""
    assert name in TIMESERIES_DATASET_MAPPING, (
        f"Unknown timeseries dataset '{name}'. "
        f"Available: {list(TIMESERIES_DATASET_MAPPING.keys())}"
    )
    return TIMESERIES_DATASET_MAPPING[name]


__all__ = [
    "TIMESERIES_DATASET_MAPPING",
    "get_timeseries_dataset",
    "get_timeseries_dataset_cls",
    "TimeSeriesBaseDataset",
    "ChronosM4DailyDataset",
    "ChronosM4WeeklyDataset",
    "ChronosM4MonthlyDataset",
]