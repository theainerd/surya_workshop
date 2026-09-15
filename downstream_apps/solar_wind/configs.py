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
from typing import ClassVar

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

    # Both index paths are paths, so they must join the base class's list to get the same
    # relative-to-the-config-file resolution. Extend this whenever you add a path field.
    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS + (
        "ds_train_index_path",
        "ds_val_index_path",
    )


# The app's entry point. Identical to load_config() except that the data: section is
# parsed into SolarWindDataConfig, so the six keys above are recognized instead of rejected.
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
