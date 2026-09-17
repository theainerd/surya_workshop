"""
Task-specific configuration for the solar-wind forecasting app.

Everything generic — paths, channels, temporal sampling, S3 settings, the model and
LoRA configs, the training and logging sections, and ``load_config()`` itself — lives in
``workshop_infrastructure/configs.py``. This file holds only what is specific to *this*
task: the OMNI solar-wind index and how its measurements are aligned to the Surya index.

The original flare-forecasting version of this file (``FlareDataConfig`` /
``load_flare_config``, matching ``configs/config_script.yaml``) is preserved in
``configs_flare_backup.py`` for reference.

**This is the pattern to copy when you fork the template.** Subclass ``DataConfig`` with
your task's fields, then bind ``load_config`` to it. You never maintain a copy of the
base config.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Optional

from workshop_infrastructure.configs import (  # re-exported for convenience
    DataConfig,
    LoraAdapterConfig,
    ModelConfig,
    OutputConfig,
    TimeEmbeddingConfig,
    TrainingConfig,
    load_config,
)


@dataclass
class SolarWindDataConfig(DataConfig):
    """DataConfig plus the OMNI solar-wind alignment settings used by ``SolarWindDSDataset``.

    Unlike the flare template's single shared catalog, solar-wind labels are already
    pre-split into train/val files matching the Surya train/val splits, so this config
    carries two index paths (``ds_train_index_path`` / ``ds_val_index_path``) instead of one.
    """
    # Paths to the pre-split OMNI solar-wind CSV indices (relative paths resolve against
    # the config file's dir). Defaults here match the nasa-ibm-ai4science/Surya-bench-
    # solarwind HF dataset (hourly OMNI, pre-cleaned — no fill-value sentinels).
    ds_train_index_path: str = ""
    ds_val_index_path: str = ""
    # Column in the index holding the event timestamp (e.g. "timestamp" — already a
    # point-in-time OMNI observation, not an event start).
    ds_time_column: str = "timestamp"
    # Column in the index to regress on (e.g. "V", solar wind speed in km/s).
    ds_target_column: str = "V"
    # Max allowed gap when matching index timestamps to Surya timesteps.
    ds_time_tolerance: str = "6min"
    # "nearest" is right when ds_time_column is already a point estimate rather than an
    # event start (as with this dataset's hourly "timestamp" column).
    ds_match_direction: str = "nearest"

    # Caps the VALIDATION set to a fixed size (chronological prefix, same every run —
    # never randomized, since "same validation" across comparisons is the whole point).
    # The unmatched val index is thousands of samples (~2.5k for the HF solar-wind
    # benchmark); evaluating all of them every epoch is not affordable on a small GPU-hour
    # budget. Set once and leave alone across every run you want to compare.
    max_val_samples: Optional[int] = None

    # When ``data.max_samples`` caps the training set, draw a random subset seeded by this
    # value instead of taking the chronological prefix. Leave null for a data-amount sweep
    # (each larger max_samples is then a superset of the smaller ones, maximizing S3 cache
    # reuse); set a distinct value per run to build ensemble members from diverse training
    # subsets of the same pool while keeping validation identical.
    train_subsample_seed: Optional[int] = None

    # Standardization applied to the regression target, as (V - target_mean) / target_std.
    # Both must be set together; leave both null to regress on raw km/s.
    #
    # Set them, and set them from the TRAINING split only. Solar wind speed is
    # ~418.6 +/- 93.9 km/s in train.csv while ``head_unembed`` starts near zero, so an
    # un-normalized target makes every model spend its whole budget walking a ~418 km/s
    # offset before it can fit any structure. The linear baseline cannot even do that: at
    # lr=1e-4 Adam shifts its bias by ~lr per step, so 200 steps covers ~0.02 km/s of a
    # 418 km/s gap -- which is exactly why its val RMSE moved 465.67 -> 465.60 over four
    # epochs and looked like a model that could not learn.
    #
    # With them set, predicting 0 means predicting climatology: the correct null, and where
    # a freshly initialized head already sits. Convert reported losses back to physical
    # units with ``RMSE_kms = sqrt(val_loss) * target_std``.
    target_mean: Optional[float] = None
    target_std: Optional[float] = None

    # Both index paths are paths, so they must join the base class's list to get the same
    # relative-to-the-config-file resolution. Extend this whenever you add a path field.
    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS + (
        "ds_train_index_path",
        "ds_val_index_path",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Half a standardization is worse than none: setting only the mean would centre the
        # target while leaving it on a ~94 km/s scale, and setting only the std would divide
        # a ~418 km/s offset without removing it. Either would be reported as if it were a
        # z-score, so reject the pair rather than silently rescaling.
        if (self.target_mean is None) != (self.target_std is None):
            raise ValueError(
                "data.target_mean and data.target_std must be set together (or both left "
                f"null to regress on raw units); got target_mean={self.target_mean!r}, "
                f"target_std={self.target_std!r}."
            )
        if self.target_std is not None and self.target_std <= 0:
            raise ValueError(f"data.target_std must be positive, got {self.target_std!r}.")


# The app's entry point. Identical to load_config() except that the data: section is
# parsed into SolarWindDataConfig, so the keys above are recognized instead of rejected.
load_solar_wind_config = partial(load_config, data_cls=SolarWindDataConfig)


__all__ = [
    "SolarWindDataConfig",
    "load_solar_wind_config",
    # Re-exports so app code can import everything config-related from one place.
    "DataConfig",
    "OutputConfig",
    "TrainingConfig",
    "ModelConfig",
    "LoraAdapterConfig",
    "TimeEmbeddingConfig",
    "load_config",
]
