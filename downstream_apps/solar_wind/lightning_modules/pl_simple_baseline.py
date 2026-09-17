"""
pl_simple_baseline.py

A minimal PyTorch Lightning wrapper for training a solar-wind speed regression model.

The original flare-forecasting version of this file (``FlareLightningModule``) is
preserved in ``pl_simple_baseline_flare_backup.py`` for reference.

This module defines a single LightningModule (SolarWindLightningModule) that:
  - Calls a user-provided PyTorch model on batched inputs (batch["ts"])
  - Computes one or more training/validation losses via a user-provided loss function
  - Logs scalar losses and evaluation metrics using Lightning's built-in logging
  - Configures a simple Adam optimizer

Intended use:
  - Provide a clean, readable baseline training loop in Lightning
  - Separate "model architecture" from "training mechanics"
  - Demonstrate how to log multiple losses/metrics consistently

Key batch contract:
  - batch["ts"]       : torch.Tensor input stack (e.g., [B, C, T, H, W])
  - batch["forecast"] : torch.Tensor target values (e.g., [B] or [B,])

Optional preprocessing:
  - If ``preprocess_fn`` is provided to ``__init__``, it is called on the batch dict
    before every model call. This is the intended hook for input transformations (such
    as inverse-normalizing SDO channels) that should not live inside the model.

Key metrics contract (the `metrics` dict passed to __init__):
  - metrics["train_loss"]    : callable(output, target) -> (loss_dict, weight_list)
        Backpropagated. Logged as "train_loss".
  - metrics["val_loss"]      : callable(output, target) -> (loss_dict, weight_list)
        Optional. Logged as "val_loss" and therefore what ModelCheckpoint monitors.
        Falls back to metrics["train_loss"] when absent.
  - metrics["train_metrics"] : callable(output, target) -> (metric_dict, weight_list)
  - metrics["val_metrics"]   : callable(output, target) -> (metric_dict, weight_list)
        Reported only. These do NOT affect checkpoint selection — "val_loss" does.

Where:
  - loss_dict / metric_dict map string names -> torch scalar tensors
  - weight_list is a list-like of floats (or tensors) aligned with the dict iteration order
    used by this baseline to form a weighted sum loss.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import lightning as L
import torch


# Type aliases for clarity in documentation / teaching.
LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any  # often a list[float] or list[torch.Tensor]


class SolarWindLightningModule(L.LightningModule):
    """
    PyTorch LightningModule for solar-wind speed regression training.

    This class wraps:
      (1) a user-provided PyTorch model (nn.Module-like) and
      (2) a set of loss/metric callables packaged in the `metrics` dictionary.

    Parameters
    ----------
    model:
        A callable model (typically torch.nn.Module) that accepts the batch input tensor
        `x = batch["ts"]` and returns predictions `output`.

    metrics:
        Dictionary containing the training loss function and metric functions.

        Required keys:
          - "train_loss": callable(output, target) -> (losses, weights)
              losses: dict[str, torch.Tensor] scalar losses
              weights: list-like aligned with iteration order of losses.keys()
          - "train_metrics": callable(output, target) -> (metrics, weights)
          - "val_metrics": callable(output, target) -> (metrics, weights)

        Optional key:
          - "val_loss": callable(output, target) -> (losses, weights)
              The validation objective. Defaults to "train_loss" when not supplied, so
              older metrics dicts keep working unchanged.

        The module uses:
          - train_loss in training_step, backpropagated and logged as "train_loss"
          - val_loss in validation_step, logged as "val_loss" — the quantity
            ModelCheckpoint monitors
          - train_metrics logged during training_step (if weights is non-empty)
          - val_metrics logged during validation_step (if weights is non-empty).
            Reported only; they do not influence checkpoint selection.

        Optional accumulation protocol:
          A metrics object may additionally implement ``update(preds, target)``,
          ``compute() -> dict[str, Tensor]`` and ``reset()``. When it does, this module
          feeds it every batch, resets it at each epoch start, and logs ``compute()`` at each
          epoch end under the ``train_epoch_``/``val_epoch_`` prefixes. This is how metrics
          that are not linear in the batch (RMSE, MAE, Pearson r) get correct epoch-level
          values instead of an average of per-batch values. A plain callable without those
          methods is skipped, so the protocol is optional.

    lr:
        Learning rate for the Adam optimizer.

    batch_size:
        Optional batch size passed to Lightning's `self.log(..., batch_size=...)`.
        This improves correct averaging behavior when using distributed settings
        or variable batch sizes.

    preprocess_fn:
        Optional callable applied to the batch dict before every model call.
        Signature: ``(batch: dict) -> dict``. Use this to apply input
        transformations (e.g., ``destandardize_channels``) without
        embedding them in the model itself.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Callable[..., Tuple[Dict[str, torch.Tensor], Weights]]],
        lr: float,
        batch_size: Optional[int] = None,
        preprocess_fn: Optional[Callable[[Dict], Dict]] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.model = model
        self.preprocess_fn = preprocess_fn

        # Loss callables: return (loss_dict, weight_list)
        self.training_loss = metrics["train_loss"]
        # "val_loss" is optional: falling back to train_loss keeps a metrics dict written
        # before this key existed working, with identical behavior.
        self.validation_loss = metrics.get("val_loss", metrics["train_loss"])

        # Metric callables: return (metric_dict, weight_list)
        self.training_evaluation = metrics["train_metrics"]
        self.validation_evaluation = metrics["val_metrics"]

        self.lr = lr

    @staticmethod
    def _combine_losses(loss_dict: LossDict, weights: Weights) -> torch.Tensor:
        """Return a weighted sum of the losses in ``loss_dict``.

        ``weights`` must be aligned with ``loss_dict.keys()`` iteration order.
        Raises ``ValueError`` if ``loss_dict`` is empty.
        """
        loss = None
        for n, key in enumerate(loss_dict.keys()):
            component = loss_dict[key] * weights[n]
            loss = component if loss is None else (loss + component)
        if loss is None:
            raise ValueError("loss_dict is empty; cannot compute a scalar loss.")
        return loss

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass used by Lightning and by explicit calls in steps.

        Parameters
        ----------
        batch:
            Batch dict (at minimum contains ``"ts"`` and ``"forecast"``).

        Returns
        -------
        torch.Tensor
            Model predictions for the batch.
        """
        return self.model(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Runs one training step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets from the batch:
              x = batch["ts"]
              target = batch["forecast"]
        2) Compute model output:
              output = self(x)
        3) Compute per-component losses and combine via provided weights:
              training_losses, training_loss_weights = training_loss(output, target)
        4) Log:
              - total weighted loss as "train_loss" (progress bar)
              - each component loss as "train_loss_<name>"
              - training metrics as "train_metric_<name>" (if any)

        Notes
        -----
        - Targets are reshaped to shape [B, 1] by unsqueeze(1) to match a common
          "single output per sample" convention.
        - The loss combination depends on dict iteration order; ensure loss dict
          insertion order is consistent if that matters.

        Returns
        -------
        torch.Tensor
            The scalar training loss used for backpropagation.
        """
        target = batch["forecast"].unsqueeze(1).float()

        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self(batch)
        training_losses, training_loss_weights = self.training_loss(output, target)
        loss = self._combine_losses(training_losses, training_loss_weights)

        # Log aggregate loss and component losses.
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        for key in training_losses.keys():
            self.log(f"train_loss_{key}", training_losses[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        # Log per-batch evaluation metrics (optional).
        training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(output, target)
        if len(training_evaluation_weights) > 0:
            for key in training_evaluation_metrics.keys():
                self.log(f"train_metric_{key}", training_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        # Feed the epoch-level accumulators, for metrics that cannot be averaged per batch.
        self._update_epoch_metrics(self.training_evaluation, output, target)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Runs one validation step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets
        2) Compute output
        3) Compute validation losses and combine via weights
        4) Log:
              - total weighted loss as "val_loss" (progress bar)
              - each component loss as "val_loss_<name>"
              - validation metrics as "val_metric_<name>" (if any)

        Notes
        -----
        - The loss is computed with `self.validation_loss`, which comes from
          metrics["val_loss"] and falls back to metrics["train_loss"] when that key is
          absent. Supply a distinct "val_loss" callable to monitor something other than
          the training objective.
        - "val_loss" is what ModelCheckpoint monitors. The `val_metrics` logged at the end
          of this method are reported only and do not affect checkpoint selection.
        - No value is returned (Lightning uses logs for validation tracking).
        """
        target = batch["forecast"].unsqueeze(1).float()

        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self(batch)
        val_losses, val_loss_weights = self.validation_loss(output, target)
        loss = self._combine_losses(val_losses, val_loss_weights)

        # Log aggregate loss and component losses.
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        for key in val_losses.keys():
            self.log(f"val_loss_{key}", val_losses[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        # Log per-batch evaluation metrics (optional).
        val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(output, target)
        if len(val_evaluation_weights) > 0:
            for key in val_evaluation_metrics.keys():
                self.log(f"val_metric_{key}", val_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        # Feed the epoch-level accumulators, for metrics that cannot be averaged per batch.
        self._update_epoch_metrics(self.validation_evaluation, output, target)

    # ------------------------------------------------------------------
    # Epoch-level metrics
    #
    # RMSE, MAE and Pearson r are not linear in the batch, so averaging per-batch values
    # over an epoch does not give the epoch's value (and at batch_size=2 a correlation is
    # not even defined). Metric objects that expose update()/compute()/reset() therefore
    # accumulate across the epoch and are read once at the end. Objects that do not expose
    # them — a plain callable, as the metrics contract allows — are skipped, so this is
    # backwards compatible.
    # ------------------------------------------------------------------

    @staticmethod
    def _supports_accumulation(evaluator: Any) -> bool:
        """True if ``evaluator`` implements the update/compute/reset accumulation protocol."""
        return all(callable(getattr(evaluator, name, None)) for name in ("update", "compute", "reset"))

    def _update_epoch_metrics(
        self, evaluator: Any, output: torch.Tensor, target: torch.Tensor
    ) -> None:
        if self._supports_accumulation(evaluator):
            evaluator.update(output.detach(), target.detach())

    def _log_epoch_metrics(self, evaluator: Any, prefix: str) -> None:
        if not self._supports_accumulation(evaluator):
            return
        for key, value in evaluator.compute().items():
            # sync_dist is not needed: torchmetrics already reduces its own state across
            # ranks inside compute(), and syncing an already-reduced scalar would average it
            # a second time.
            self.log(f"{prefix}_{key}", value, prog_bar=False, batch_size=self.batch_size)

    def on_train_epoch_start(self) -> None:
        if self._supports_accumulation(self.training_evaluation):
            self.training_evaluation.reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics(self.training_evaluation, "train_epoch")

    def on_validation_epoch_start(self) -> None:
        if self._supports_accumulation(self.validation_evaluation):
            self.validation_evaluation.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.validation_evaluation, "val_epoch")

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and LR schedule used by Lightning.

        Adam alone, with a fixed ``self.lr`` for the whole run, was found to overshoot:
        the LoRA adapters start as a no-op and the head_* modules start randomly
        initialized, so early steps make large, useful corrections — but nothing tells
        Adam to slow down once it finds a good region, so later epochs bounce to a worse
        val_loss before partially recovering. Cosine-decaying the LR to ~0 over the run
        lets it settle into that region instead of overshooting past it.

        Returns
        -------
        dict
            Adam optimizer plus a per-epoch CosineAnnealingLR schedule.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
