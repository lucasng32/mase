import math

import torch
from torchmetrics import MeanMetric

from ..base import WrapperBase


class TimeSeriesForecastingModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info=None,
        learning_rate=1e-4,
        weight_decay=0.0,
        scheduler_args=None,
        epochs=50,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_args=scheduler_args,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.mae_val = MeanMetric()
        self.mae_test = MeanMetric()

    def _output_patch_size(self) -> int:
        if hasattr(self.model, "chronos_config"):
            return int(self.model.chronos_config.output_patch_size)
        return 16

    def _batch_to_model_inputs(self, batch: dict) -> dict:
        context = batch["past_values"]
        future_target = batch["future_values"]
        output_patch_size = self._output_patch_size()
        num_output_patches = max(1, math.ceil(future_target.shape[-1] / output_patch_size))

        return {
            "context": context,
            "context_mask": torch.ones_like(context, dtype=torch.bool),
            "group_ids": torch.zeros((context.shape[0],), dtype=torch.long, device=context.device),
            "future_covariates": torch.zeros(
                (context.shape[0], num_output_patches * output_patch_size),
                dtype=context.dtype,
                device=context.device,
            ),
            "future_covariates_mask": torch.zeros(
                (context.shape[0], num_output_patches * output_patch_size),
                dtype=torch.bool,
                device=context.device,
            ),
            "future_target": future_target,
            "future_target_mask": torch.ones_like(future_target, dtype=torch.bool),
            "num_output_patches": num_output_patches,
        }

    @staticmethod
    def _get_output_value(outputs, key: str):
        if isinstance(outputs, dict):
            return outputs[key]
        return getattr(outputs, key)

    @staticmethod
    def _point_forecast(quantile_preds: torch.Tensor) -> torch.Tensor:
        if quantile_preds.shape[1] == 0:
            raise ValueError("quantile_preds must contain at least one quantile axis")
        median_idx = quantile_preds.shape[1] // 2
        return quantile_preds[:, median_idx]

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**self._batch_to_model_inputs(batch))
        loss = self._get_output_value(outputs, "loss")
        self.log("train_loss_step", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**self._batch_to_model_inputs(batch))
        loss = self._get_output_value(outputs, "loss")
        quantile_preds = self._get_output_value(outputs, "quantile_preds")
        target = batch["future_values"]
        point_forecast = self._point_forecast(quantile_preds)[..., : target.shape[-1]]
        mae = torch.mean(torch.abs(point_forecast - target))
        self.loss_val.update(loss)
        self.mae_val.update(mae)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_loss_epoch", self.loss_val.compute(), prog_bar=True)
        self.log("val_mae_epoch", self.mae_val.compute(), prog_bar=True)
        self.loss_val.reset()
        self.mae_val.reset()

    def test_step(self, batch, batch_idx):
        outputs = self.forward(**self._batch_to_model_inputs(batch))
        loss = self._get_output_value(outputs, "loss")
        quantile_preds = self._get_output_value(outputs, "quantile_preds")
        target = batch["future_values"]
        point_forecast = self._point_forecast(quantile_preds)[..., : target.shape[-1]]
        mae = torch.mean(torch.abs(point_forecast - target))
        self.loss_test.update(loss)
        self.mae_test.update(mae)
        return loss

    def on_test_epoch_end(self):
        self.log("test_loss_epoch", self.loss_test.compute(), prog_bar=True)
        self.log("test_mae_epoch", self.mae_test.compute(), prog_bar=True)
        self.loss_test.reset()
        self.mae_test.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.forward(**self._batch_to_model_inputs(batch))
        quantile_preds = self._get_output_value(outputs, "quantile_preds")
        return {
            "batch_idx": batch_idx,
            "quantile_preds": quantile_preds,
            "point_forecast": self._point_forecast(quantile_preds),
        }
