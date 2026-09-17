#!/usr/bin/env python3
"""
Runnable finetuning script derived from `2_finetune_template_1D.ipynb`.

Design goals
- Config-driven: all hyperparameters live in config_script.yaml
- Minimal CLI: --config plus a handful of per-run overrides
- Multi-GPU capable (DDP) when run as a script

Assumptions
- Assets (`scalers.yaml` + model weights) are downloaded automatically on first run.
- You run this script from the repo root and specify devices via CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.solar_wind.3_finetune_template_1D \
        --config downstream_apps/solar_wind/configs/config_script_01.yaml

All parameters live in the YAML. The CLI overrides only what genuinely varies between
runs of the same config: --max-epochs and --batch-size (sweeps), --s3-cache-dir
(per-machine scratch) and --deterministic (reproducibility, off by default for speed).
Everything else is a config edit.

Forking this script: build_datasets() and build_model() are the only two functions with
task-specific content. build_trainer() and main() should need no changes.
"""

from __future__ import annotations

import argparse
import os

# Must be set BEFORE torch is imported: cuBLAS reads this once, when it initializes, so
# setting it later has no effect. Deterministic cuBLAS on CUDA >= 10.2 requires it, and
# without it every run under training.deterministic warns (or raises, when set to true).
# setdefault so a deliberate ":16:8" (smaller workspace, slightly slower) is respected.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from typing import Tuple

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from downstream_apps.solar_wind.configs import TrainingConfig, load_solar_wind_config
from downstream_apps.solar_wind.datasets.loaders import build_solar_wind_dataloaders
from downstream_apps.solar_wind.lightning_modules.pl_simple_baseline import SolarWindLightningModule
from downstream_apps.solar_wind.metrics.template_metrics import SolarWindMetrics
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.utils import (
    apply_peft_lora,
    build_run_name,
    build_scalers,
    load_pretrained_weights,
    UploadBestCheckpointToS3,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_script_01.yaml"

# --deterministic accepts the same three tokens as the YAML key. argparse hands back a
# string, so the two boolean ones are mapped to real bools -- TrainingConfig validates
# against True/False/"warn", not against their spellings.
_DETERMINISTIC_CLI = {"false": False, "warn": "warn", "true": True}


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to the run config YAML (default: this app's config_script.yaml).")
    # Dev toggles: flip without editing the YAML
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging (useful for local runs).")
    parser.add_argument("--train_baseline", action="store_true",
                        help="Train the simple linear baseline instead of HelioSpectformer.")
    # Per-job / per-machine overrides: vary across runs without touching the YAML
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override training.max_epochs from the config YAML.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.batch_size from the config YAML.")
    # These two are what distinguish the three comparison categories from one another, so
    # they vary run-to-run over a single config rather than justifying a config file each.
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override data.max_samples: cap the TRAINING set only "
                             "(validation is never touched). This is the axis of the "
                             "data-amount comparison. Pair it with --train-subsample-seed, "
                             "or the cap takes a chronological prefix and the amount of data "
                             "is confounded with which years it covers.")
    parser.add_argument("--train-subsample-seed", type=int, default=None,
                        help="Override data.train_subsample_seed: draw the training subset at "
                             "random (spread over the whole index) instead of taking a "
                             "chronological prefix. Vary it at fixed --max-samples to build "
                             "ensemble members that differ in composition but not in size.")
    parser.add_argument("--s3-cache-dir", type=str, default=None,
                        help="Override data.s3_cache_dir (the local cache for S3 reads). "
                             "Handy when the same config runs on machines with different scratch.")
    parser.add_argument("--deterministic", choices=tuple(_DETERMINISTIC_CLI), default=None,
                        help="Override training.deterministic. The config default is 'false', "
                             "which trades reproducibility for roughly 20%% throughput. Pass "
                             "'warn' when you need to tell whether a change in your results came "
                             "from your edit or from run-to-run drift.")
    return parser.parse_args()


def build_datasets(cfg: TrainingConfig, scalers) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from config.

    The wiring itself lives in ``datasets/loaders.py`` because the eval and embedding-cache
    scripts must construct these splits identically — including the target standardization,
    which is what keeps ``val_loss`` in the same units a checkpoint was trained in.

    ``scalers`` is built once in main() and shared with build_model(), so the two paths
    cannot end up with different normalization statistics.
    """
    return build_solar_wind_dataloaders(cfg, scalers)


def build_model(cfg: TrainingConfig, scalers, train_baseline: bool = False) -> L.LightningModule:
    """Instantiate the model and wrap it in a LightningModule.

    ``scalers`` is only needed by the linear baseline, which consumes its inputs in
    signum-log space; the HelioSpectformer path works directly on normalized inputs.
    """
    metrics = {
        "train_loss": SolarWindMetrics("train_loss"),
        # val_loss is what ModelCheckpoint monitors; val_metrics are reported only.
        "val_loss": SolarWindMetrics("val_loss"),
        "train_metrics": SolarWindMetrics("train_metrics"),
        "val_metrics": SolarWindMetrics("val_metrics"),
    }

    if train_baseline:
        from functools import partial
        from downstream_apps.solar_wind.models.simple_baseline import (
            RegressionSolarWindModel,
            destandardize_channels,
        )
        n_input_timestamps = cfg.model.time_embedding.time_dim
        n_channels = len(cfg.data.channels)
        model = RegressionSolarWindModel(n_input_timestamps * n_channels)
        preprocess_fn = partial(destandardize_channels, channel_order=cfg.data.channels, scalers=scalers)
        return SolarWindLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size, preprocess_fn=preprocess_fn)
    else:
        from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
        model = HelioSpectformer1D.from_config(
            cfg.model,
            num_outputs=1,
            dtype=cfg.dtype,
            use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
        )
        load_pretrained_weights(model, cfg.model.pretrained_path)

        # Three fine-tuning regimes, selected from the model: section of the YAML:
        #   use_lora: true                        -> LoRA adapters (default)
        #   use_lora: false, freeze_backbone: true  -> linear probe (head only)
        #   use_lora: false, freeze_backbone: false -> full fine-tuning
        if cfg.model.freeze_backbone:
            for name, param in model.named_parameters():
                if name.startswith("backbone."):
                    param.requires_grad = False
        if cfg.model.use_lora:
            model = apply_peft_lora(model, cfg.model.lora_config)

        _log_trainable_parameters(model)

    return SolarWindLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size)


def _log_trainable_parameters(model) -> None:
    """Print the trainable/total parameter counts, so the chosen regime is visible in the log."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def build_trainer(
    cfg: TrainingConfig,
    run_name: str,
    no_wandb: bool = False,
    max_epochs_override: int | None = None,
) -> Tuple[L.Trainer, ModelCheckpoint]:
    """Configure loggers, callbacks, and the Lightning Trainer.

    ``run_name`` (from ``build_run_name()``) names both loggers instead of the bare
    ``cfg.job_id``, so runs that vary training-set size, composition, or fine-tuning mode
    land on distinct WandB names and distinct ``runs/<run_name>`` CSV folders rather than
    colliding into the same name with an opaque version number.
    """
    max_epochs = max_epochs_override if max_epochs_override is not None else cfg.max_epochs

    loggers = []
    if not no_wandb:
        loggers.append(WandbLogger(
            entity=cfg.wandb_entity,  # None = personal account; set in YAML for team runs
            project=cfg.wandb_project,
            name=run_name,
            log_model=False,
            save_dir=os.environ.get("TMPDIR", "./wandb/wandb_tmp"),
        ))
    loggers.append(CSVLogger("runs", name=run_name))

    Path(cfg.output.ckpt_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.output.ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    upload_cb = UploadBestCheckpointToS3(
        checkpoint_cb=checkpoint_cb,
        bucket=cfg.output.s3_bucket,
        prefix=cfg.output.s3_prefix,
        fixed_key_name=(cfg.output.s3_best_key or None),
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        # Reproducibility. "warn" (the default) gives bit-identical runs wherever a
        # deterministic kernel exists and names the op where one does not, instead of
        # killing the run. benchmark is pinned rather than inherited: cuDNN autotuning
        # picks algorithms by timing, so leaving it on would reintroduce run-to-run drift.
        deterministic=cfg.deterministic,
        benchmark=False,
        logger=loggers,
        callbacks=[checkpoint_cb, upload_cb],
        log_every_n_steps=2,
        # batch_size=2 makes every step high-variance; clip to guard against a single
        # bad batch derailing the LoRA adapters / freshly-initialized head.
        gradient_clip_val=1.0,
    )
    return trainer, checkpoint_cb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_solar_wind_config(args.config)
    # Seeding comes after the config load, so the seed is a configured value rather than
    # a constant buried in the code. Seeds Python, NumPy and torch in this process;
    # workers=True extends it to DataLoader workers.
    L.seed_everything(cfg.seed, workers=True)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_samples is not None:
        cfg.data.max_samples = args.max_samples
    if args.train_subsample_seed is not None:
        cfg.data.train_subsample_seed = args.train_subsample_seed
    if args.s3_cache_dir is not None:
        cfg.data.s3_cache_dir = args.s3_cache_dir
    if args.deterministic is not None:
        cfg.deterministic = _DETERMINISTIC_CLI[args.deterministic]
    # Fetch scalers, and the backbone weights unless we are training the baseline.
    ensure_assets(cfg, which=["scalers"] if args.train_baseline else ["scalers", "weights"])

    # Built once and shared: the dataset normalizes with these, and the linear baseline
    # de-standardizes with them. Two separate builds could silently disagree.
    scalers = build_scalers(info=cfg.data.scalers_path)

    train_loader, val_loader = build_datasets(cfg, scalers)
    lit_model = build_model(cfg, scalers, train_baseline=args.train_baseline)

    if args.train_baseline:
        mode = "baseline"
    elif cfg.model.use_lora:
        mode = "lora"
    elif cfg.model.freeze_backbone:
        mode = "probe"
    else:
        mode = "full"
    run_name = build_run_name(
        job_id=cfg.job_id,
        mode=mode,
        n_train=cfg.data.max_samples if cfg.data.max_samples is not None else "all",
        max_epochs=args.max_epochs if args.max_epochs is not None else cfg.max_epochs,
        subsample_seed=cfg.data.train_subsample_seed,
    )
    print(f"[RUN] {run_name}")
    trainer, checkpoint_cb = build_trainer(
        cfg, run_name, no_wandb=args.no_wandb, max_epochs_override=args.max_epochs
    )

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_cb.best_model_path:
        print(f"[CKPT] Best checkpoint: {checkpoint_cb.best_model_path}")
        if checkpoint_cb.best_model_score is not None:
            print(f"[CKPT] Best val_loss: {float(checkpoint_cb.best_model_score):.6f}")
    else:
        print("[CKPT] No best checkpoint was saved.")


if __name__ == "__main__":
    main()
