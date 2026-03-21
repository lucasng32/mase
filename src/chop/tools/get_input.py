import inspect
import math
from typing import Literal
from enum import Enum
import torch


class ModelSource(Enum):
    """
    The source of the model, must be one of the following:
    - HF: HuggingFace
    - MANUAL: manually implemented
    - PATCHED: patched HuggingFace
    - TOY: toy model for testing and debugging
    - PHYSICAL: model that perform classification using physical data point vectors
    - NERF: model that estimates neural radiance field (NeRF) of a 3D scene
    """

    HF_TRANSFORMERS = "hf_transformers"
    MANUAL = "manual"
    PATCHED = "patched"
    TOY = "toy"
    TORCHVISION = "torchvision"
    VISION_OTHERS = "vision_others"
    PHYSICAL = "physical"
    NERF = "nerf"


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
    }


def get_cf_args(model_info, task: str, model):
    """Get concrete forward args for freezing dynamic control flow in forward pass"""
    all_forward_kwargs = _get_default_args(model.forward)
    cf_args = {}
    if model_info.model_source == ModelSource.PATCHED:
        cf_args = model.patched_nodes["concrete_forward_args"]
    elif model_info.is_vision_model or model_info.is_physical_model:
        cf_args = {}
    elif model_info.is_nlp_model:
        match task:
            case "classification" | "cls":
                required_input_args = ["input_ids", "attention_mask", "labels"]
            case "language_modeling" | "lm":
                required_input_args = ["input_ids", "attention_mask", "labels"]
            case "translation" | "tran":
                required_input_args = [
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
        for required_input_arg in required_input_args:
            all_forward_kwargs.pop(required_input_arg)
        cf_args = all_forward_kwargs
    elif model_info.is_timeseries_model:
        match task:
            case "forecasting":
                required_input_args = ["context"]
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
        for required_input_arg in required_input_args:
            all_forward_kwargs.pop(required_input_arg, None)
        cf_args = all_forward_kwargs
    else:
        raise RuntimeError(f"Unsupported model+task: {model_info.name}+{task}")
    return cf_args


def get_hf_input_names(model_info, task: str) -> list[str] | None:
    if not model_info.is_timeseries_model:
        return None

    match task:
        case "forecasting":
            # The generic MASE training/transform path should trace the full
            # forecasting signature so quantized GraphModules can still be used
            # in fine-tuning or evaluation code paths.
            return [
                "context",
                "context_mask",
                "group_ids",
                "future_covariates",
                "future_covariates_mask",
                "num_output_patches",
                "future_target",
                "future_target_mask",
            ]
        case _:
            raise ValueError(f"Task {task} is not supported for {model_info.name}")


def _get_timeseries_output_patch_size(model) -> int:
    if model is not None and hasattr(model, "chronos_config"):
        return int(model.chronos_config.output_patch_size)
    return 16


def _format_timeseries_inputs(input_dict: dict, model=None, device: str = "meta") -> dict:
    past_values = input_dict["past_values"].to(device)
    future_values = input_dict["future_values"].to(device)
    output_patch_size = _get_timeseries_output_patch_size(model)
    num_output_patches = max(1, math.ceil(future_values.shape[-1] / output_patch_size))

    return {
        "context": past_values,
        "context_mask": torch.ones_like(past_values, dtype=torch.bool, device=device),
        "group_ids": torch.zeros((past_values.shape[0],), dtype=torch.long, device=device),
        "future_covariates": torch.zeros(
            (past_values.shape[0], num_output_patches * output_patch_size),
            dtype=past_values.dtype,
            device=device,
        ),
        "future_covariates_mask": torch.zeros(
            (past_values.shape[0], num_output_patches * output_patch_size),
            dtype=torch.bool,
            device=device,
        ),
        "future_target": future_values,
        "future_target_mask": torch.ones_like(future_values, dtype=torch.bool, device=device),
        "num_output_patches": num_output_patches,
    }


def get_dummy_input(
    model_info,
    data_module,
    task: str,
    device: str = "meta",
    model=None,
) -> dict:
    """Create a single dummy input for a model. The dummy input is a single sample from the training set.

    Args:
        datamodule (MaseDataModule): a LightningDataModule instance (see machop/chop/dataset/__init__.py). Make sure the datamodule is prepared and setup.

        task (str): task name, one of ["cls", "classification", "lm", "language_modeling", "translation", "tran"]

        is_nlp_model (bool, optional): Whether the task is NLP task or not. Defaults to False.

    Returns:
        dict: a dummy input dict which can be passed to the wrapped lightning model's forward method, like model(**dummy_input)
    """
    assert (
        data_module.train_dataset is not None
    ), "DataModule is not setup. Please call data_module.prepare_data() and .setup()."
    index: int = 0
    train_iter = iter(data_module.train_dataloader())
    n_batches = len(data_module.train_dataloader())
    if index >= n_batches * data_module.batch_size:
        raise ValueError(f"index {index} is out of range.")
    batch_index = index // data_module.batch_size
    sample_index = index % data_module.batch_size
    for _ in range(batch_index):
        next(train_iter)

    if model_info.is_vision_model or model_info.is_physical_model:
        match task:
            case "classification" | "cls":
                x, y = next(train_iter)
                # x = x[[0], ...].to(device)
                x = x.to(device)
                if data_module.name == "mnist" and model_info.is_vision_model:
                    dummy_inputs = {"input_1": x}
                else:
                    dummy_inputs = {"x": x}
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
    elif model_info.is_nerf_model:
        # TODO:
        pass

    elif model_info.is_nlp_model:
        match task:
            case "classification" | "cls":
                input_dict = next(train_iter)
                input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
                attention_mask = input_dict["attention_mask"][[sample_index], ...].to(
                    device
                )

                labels = input_dict["labels"][[sample_index], ...].to(device)
                dummy_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if "token_type_ids" in input_dict:
                    dummy_inputs["token_type_ids"] = input_dict["token_type_ids"][
                        [sample_index], ...
                    ].to(device)

            case "language_modeling" | "lm":
                input_dict = next(train_iter)
                input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
                attention_mask = input_dict["attention_mask"][[sample_index], ...].to(
                    device
                )
                labels = input_dict["labels"][[sample_index], ...].to(device)
                dummy_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            case "translation" | "tran":
                input_dict = next(train_iter)
                input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
                attention_mask = input_dict["attention_mask"][[sample_index], ...].to(
                    device
                )
                decoder_input_ids = input_dict["decoder_input_ids"][
                    [sample_index], ...
                ].to(device)
                decoder_attention_mask = input_dict["decoder_attention_mask"][
                    [sample_index], ...
                ].to(device)
                dummy_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                }
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
    elif model_info.is_timeseries_model:
        match task:
            case "forecasting":
                input_dict = next(train_iter)
                single_input = {
                    "past_values": input_dict["past_values"][[sample_index], ...],
                    "future_values": input_dict["future_values"][[sample_index], ...],
                }
                dummy_inputs = _format_timeseries_inputs(single_input, model=model, device=device)
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
    else:
        raise RuntimeError(f"Unsupported model+task: {model_info.name}+{task}")

    return dummy_inputs


class InputGenerator:
    def __init__(
        self,
        model_info,
        data_module,
        task: str,
        which_dataloader: Literal["train", "val", "test"],
        max_batches: int = None,
        model=None,
    ) -> None:
        """
        Input generator for feeding batches to models. This is used for software passes.

        Args:
            datamodule (MyDataModule): a MyDataModule instance (see machop/chop/dataset/data_module.py). Make sure the datamodule is prepared and setup.
            max_batches (int, optional): Maximum number of batches to generate. Defaults to None will stop when reaching the last batch in dataloader.

        Returns:
            (dict): a dummy input dict which can be passed to the wrapped lightning model's forward method, like model(**dummy_input)
        """
        assert (
            getattr(data_module, f"{which_dataloader}_dataset") is not None
        ), "DataModule is not setup. Please call data_module.prepare_data() and .setup()."
        self.model_info = model_info
        self.task = task
        self.model = model

        self.batch_size = data_module.batch_size
        self.dataloader = getattr(data_module, f"{which_dataloader}_dataloader")()
        self.dataloader_iter = iter(self.dataloader)

        self.max_batches = max_batches
        self.current_batch = 0

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self.max_batches is not None and self.current_batch >= self.max_batches:
            raise StopIteration

        if self.model_info.is_vision_model or self.model_info.is_physical_model:
            match self.task:
                case "classification" | "cls":
                    x, y = next(self.dataloader_iter)
                    inputs = {"x": x}
                case _:
                    raise ValueError(
                        f"Task {self.task} is not supported for {self.model_info.name}"
                    )
        elif self.model_info.is_physical_model:
            match self.task:
                case "classification" | "cls":
                    x, y = next(self.dataloader_iter)
                    inputs = {"x": x}
                case _:
                    raise ValueError(
                        f"Task {self.task} is not supported for {self.model_info.name}"
                    )
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    input_dict = next(self.dataloader_iter)
                    inputs = {
                        "input_ids": input_dict["input_ids"],
                        "attention_mask": input_dict["attention_mask"],
                        "labels": input_dict["labels"],
                    }
                    if "token_type_ids" in input_dict:
                        inputs["token_type_ids"] = input_dict["token_type_ids"]
                case "language_modeling" | "lm":
                    input_dict = next(self.dataloader_iter)
                    inputs = {
                        "input_ids": input_dict["input_ids"],
                        "attention_mask": input_dict["attention_mask"],
                        "labels": input_dict["labels"],
                    }
                case "translation" | "tran":
                    input_dict = next(self.dataloader_iter)
                    inputs = {
                        "input_ids": input_dict["input_ids"],
                        "attention_mask": input_dict["attention_mask"],
                        "decoder_input_ids": input_dict["decoder_input_ids"],
                        "decoder_attention_mask": input_dict["decoder_attention_mask"],
                    }
                case _:
                    raise ValueError(
                        f"Task {self.task} is not supported for {self.model_info.name}"
                    )
        elif self.model_info.is_timeseries_model:
            match self.task:
                case "forecasting":
                    batch = next(self.dataloader_iter)
                    inputs = _format_timeseries_inputs(batch, model=self.model, device=batch["past_values"].device)
                case _:
                    raise ValueError(
                        f"Task {self.task} is not supported for {self.model_info.name}"
                    )
        else:
            raise RuntimeError(
                f"Unsupported model+task: {self.model_info.name}+{self.task}"
            )

        self.current_batch += 1
        return inputs
