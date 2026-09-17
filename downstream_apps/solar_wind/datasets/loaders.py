"""
Shared dataset / dataloader construction for the solar-wind app.

Three scripts need the *same* solar-wind dataset wiring: ``3_finetune_template_1D.py``
(training), ``4_ensemble_eval.py`` (scoring checkpoints) and ``6_cache_embeddings.py``
(building the embedding cache). Every one of them has to agree on the label alignment
arguments and, crucially, on the target transform — a checkpoint trained on standardized
targets but scored on raw km/s produces a number that looks like a catastrophic regression
and is really just a unit mismatch. That bug was already latent in ``4_ensemble_eval.py``,
which built its own loader and never passed ``label_transform``.

So the wiring lives here once, and the scripts call it. Only the arguments that genuinely
differ between the scripts are parameters.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from downstream_apps.solar_wind.datasets.solar_wind_dataset_01 import SolarWindDSDataset
from workshop_infrastructure.datasets.builders import build_helio_dataloaders


def build_label_transform(cfg) -> Optional[Callable[[pd.Series], pd.Series]]:
    """Return the target standardization implied by the config, or ``None``.

    Maps the raw target to ``(V - data.target_mean) / data.target_std`` when both statistics
    are set, and returns ``None`` when neither is (regress on raw units). The pair is
    validated together in ``SolarWindDataConfig.__post_init__``, so exactly one of them being
    set never reaches here.

    Why this matters: solar wind speed is ~418.6 +/- 93.9 km/s while ``head_unembed`` starts
    near zero, so without this the model must first walk a ~418 km/s offset before it can fit
    any structure — and the 14-parameter linear baseline cannot, at lr=1e-4, walk that far in
    the number of steps a small budget allows.

    Args:
        cfg: A ``TrainingConfig`` whose ``.data`` is a ``SolarWindDataConfig``.

    Returns:
        A callable suitable for ``SolarWindDSDataset(label_transform=...)``, or ``None``.
    """
    mean, std = cfg.data.target_mean, cfg.data.target_std
    if mean is None or std is None:
        return None
    return lambda series: (series - mean) / std


def build_solar_wind_dataloaders(
    cfg,
    scalers,
    *,
    val_index_path: str | None = None,
    return_surya_stack: bool = True,
    num_workers: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create the train and validation DataLoaders for the solar-wind task.

    Everything generic (channels, temporal sampling, S3 access, worker settings) comes from
    ``build_helio_dataloaders()``; only the solar-wind-specific arguments are set here.

    ``label_transform`` is passed as a *shared* task kwarg rather than through
    ``train_kwargs``/``val_kwargs``, so both splits are structurally guaranteed the identical
    transform. ``data.max_samples`` and ``data.train_subsample_seed`` go the other way — into
    ``train_kwargs`` only — so a training-set cap can never shrink or reorder validation.

    Args:
        cfg: A ``TrainingConfig`` from ``load_solar_wind_config()``.
        scalers: Channel normalization statistics, built once by the caller and shared with
            the model so the two cannot disagree.
        val_index_path: Overrides ``data.ds_val_index_path``. Lets one config be pointed at a
            different evaluation split (e.g. a temporal holdout) without a second config file.
        return_surya_stack: ``False`` returns labels only and fetches no SDO frames — the
            cheap way to inspect a split's targets, size and time span.
        num_workers: Overrides ``training.num_workers``.

    Returns:
        ``(train_loader, val_loader)``. Only the training loader shuffles, and both drop a
        final partial batch (``drop_last=True`` in ``build_helio_dataloaders``). Consumers
        that need to line rows up with a split's index should key on each sample's
        ``ds_index`` timestamp rather than on position, since neither property is
        order-preserving.
    """
    return build_helio_dataloaders(
        cfg,
        SolarWindDSDataset,
        scalers=scalers,
        seed=cfg.seed,
        num_workers=num_workers,
        return_surya_stack=return_surya_stack,
        label_transform=build_label_transform(cfg),
        ds_time_column=cfg.data.ds_time_column,
        ds_target_column=cfg.data.ds_target_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
        train_kwargs={
            "ds_index_path": cfg.data.ds_train_index_path,
            "max_number_of_samples": cfg.data.max_samples,
            "train_subsample_seed": cfg.data.train_subsample_seed,
        },
        val_kwargs={
            "ds_index_path": val_index_path or cfg.data.ds_val_index_path,
            "max_number_of_samples": cfg.data.max_val_samples,
        },
    )
