"""
Template metrics for solar-wind speed regression.

The original flare-forecasting version of this file (``FlareMetrics``) is preserved in
``template_metrics_flare_backup.py`` for reference.

SolarWindMetrics defines four metric sets:
- "train_loss"    — differentiable loss that drives backpropagation (MSE).
- "val_loss"      — the quantity logged as `val_loss` and used to select checkpoints.
                    Defaults to the same MSE as "train_loss"; override it when your task
                    needs a different validation objective.
- "train_metrics" — non-differentiable metrics logged during training.
- "val_metrics"   — metrics logged at validation for reporting only.
                    These do NOT influence checkpoint selection — "val_loss" does.

The __call__ method selects the appropriate metric set based on the mode passed at
construction time. The dictionary keys returned by each method become the metric names
propagated to the logger (e.g. WandB, CSV).

PER-BATCH vs PER-EPOCH — why RMSE is not logged per batch
---------------------------------------------------------
``__call__`` returns per-*batch* values, and Lightning averages whatever it is handed over
the epoch. Averaging is only faithful for quantities that are linear in the batch, which
MSE is (with ``drop_last=True`` every batch is the same size, so the mean of per-batch MSE
is the exact global MSE) and RMSE is not. Logging a per-batch RMSE therefore reports the
mean of per-batch square roots, which understates the true global RMSE — it read 49.55
where the real figure was 59.72.

Worse, a *relative* metric is meaningless per batch: ``RelativeSquaredError`` normalizes by
the variance of the targets it is given, and at ``batch_size=2`` that is the variance of
two numbers, so it divides by ~0 and explodes (values of 17-227 were logged). RRSE has been
removed for that reason.

So: ``__call__`` reports only MSE, and RMSE / MAE / Pearson r are accumulated across the
epoch and emitted once by ``compute()``. ``SolarWindLightningModule`` drives that through
``update()`` / ``compute()`` / ``reset()``; a metrics object without those methods still
works exactly as before, so this stays compatible with plain callables.
"""

import torch
import torchmetrics as tm  # Lots of possible metrics in here https://lightning.ai/docs/torchmetrics/stable/all-metrics.html

# Shape contract: predictions arrive as (B,) from HelioSpectformer1D or (B, 1) from the
# linear baseline, while targets are always (B, 1). Every metric below flattens both with
# reshape(-1) rather than squeeze(-1): squeeze is shape-dependent and collapses a
# batch of one to a 0-d scalar, which then fails to broadcast against a (1,) target.
class SolarWindMetrics:
    def __init__(self, mode: str):
        """
        Initialize SolarWindMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. One of "train_loss",
                        "val_loss", "train_metrics", or "val_metrics".
        """
        self.mode = mode

        # Accumulating torchmetrics instances, built once and reused. These hold state
        # across the epoch: update() feeds them, compute() reads the epoch-level value, and
        # reset() clears them at the epoch boundary. Pearson r is included because it is the
        # conventional skill measure for solar wind speed and is only defined over a
        # population, never over a single batch of two.
        self._epoch_metrics: dict[str, tm.Metric] = {
            "mse": tm.MeanSquaredError(squared=True),
            "rmse": tm.MeanSquaredError(squared=False),
            "mae": tm.MeanAbsoluteError(),
            "r": tm.PearsonCorrCoef(),
        }

    def _ensure_device(self, preds: torch.Tensor) -> None:
        """Move the accumulating torchmetrics modules to ``preds``' device, if needed."""
        for name, metric in self._epoch_metrics.items():
            if metric.device != preds.device:
                self._epoch_metrics[name] = metric.to(preds.device)

    # ------------------------------------------------------------------
    # Epoch-level accumulation
    # ------------------------------------------------------------------

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch into the epoch-level metrics.

        Called once per batch by ``SolarWindLightningModule``. Nothing is logged here; the
        accumulated values are read by ``compute()`` at the end of the epoch.
        """
        self._ensure_device(preds)
        flat_preds, flat_target = preds.reshape(-1), target.reshape(-1)
        for metric in self._epoch_metrics.values():
            metric.update(flat_preds, flat_target)

    def compute(self) -> dict[str, torch.Tensor]:
        """Return the epoch-level metrics accumulated since the last ``reset()``.

        Metrics that cannot be computed from what was accumulated are dropped rather than
        raising: ``PearsonCorrCoef`` needs at least two samples and a non-zero variance in
        both series, which a one-batch sanity run does not always provide.
        """
        results = {}
        for name, metric in self._epoch_metrics.items():
            try:
                value = metric.compute()
            except (ValueError, RuntimeError):
                continue
            if torch.isfinite(value).all():
                results[name] = value
        return results

    def reset(self) -> None:
        """Clear the accumulated state. Call at the start of every epoch."""
        for metric in self._epoch_metrics.values():
            metric.reset()

    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated loss metrics.
                                        Keys are metric names (e.g., "mse"), and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """

        output_metrics = {}
        output_weights = []

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds.reshape(-1), target.reshape(-1))
        output_weights.append(1)

        return output_metrics, output_weights

    def val_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate the validation loss — the quantity logged as ``val_loss`` and used by
        ModelCheckpoint to select the best model.

        By default this delegates to ``train_loss``, so the monitored quantity has the same
        form as the training objective and the two cannot drift apart by accident. This is
        the hook to override when your task needs a different validation objective (a
        different weighting, a metric that is meaningful only on held-out data, etc.).

        Note that this is deliberately separate from ``val_metrics``: those are reported
        for information only and do not affect checkpoint selection.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated loss metrics.
                - list[float]: List of weights for each calculated metric.
        """
        return self.train_loss(preds, target)

    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate per-batch evaluation metrics for training.
        IMPORTANT:  These metrics are only for reporting purposes and do not
                    contribute to the training loss. Use only if you want to
                    monitor additional metrics during training.

        Only MSE is returned, because it is the one quantity here that survives Lightning
        averaging it over the epoch. RMSE, MAE and Pearson r come from ``compute()`` at the
        epoch boundary instead — see the PER-BATCH vs PER-EPOCH note in the module docstring.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated evaluation metrics.
                                        Keys are metric names, and values are the corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        output_metrics = {}
        output_weights = []

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds.reshape(-1), target.reshape(-1))
        output_weights.append(1)

        return output_metrics, output_weights

    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate per-batch metrics for validation.

        As with ``train_metrics``, only MSE is reported per batch; the epoch-level RMSE, MAE
        and Pearson r are emitted by ``compute()``.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated metrics.
                                        Keys are metric names (e.g., "mse"), and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """

        output_metrics = {}
        output_weights = []

        output_metrics["mse"] = torch.nn.functional.mse_loss(preds.reshape(-1), target.reshape(-1))
        output_weights.append(1)

        return output_metrics, output_weights

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Evaluate metrics for the mode set at construction time.

        Args:
            preds: Model output tensor. Shape depends on the application.
            target: Ground truth tensor to compare against.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Metric dictionary. Keys become logger metric names; values are
                  scalar tensors aggregated over the batch.
                - List of per-metric weights (used by SolarWindLightningModule to
                  combine multiple loss terms into a single scalar).
        """

        match self.mode.lower():

            case "train_loss":
                return self.train_loss(preds, target)

            # No torch.no_grad() here, matching "train_loss": Lightning already disables
            # gradients during validation, so wrapping it would differ gratuitously from
            # the loss case this mirrors.
            case "val_loss":
                return self.val_loss(preds, target)

            case "train_metrics":
                with torch.no_grad():
                    return self.train_metrics(preds, target)

            case "val_metrics":
                with torch.no_grad():
                    return self.val_metrics(preds, target)

            case _:
                raise NotImplementedError(
                    f"{self.mode} is not implemented as a valid metric case."
                )
