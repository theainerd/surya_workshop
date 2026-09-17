#!/usr/bin/env python3
"""
Evaluate an ensemble of solar-wind LoRA finetunes on the shared validation set.

This is what makes "build your own ensemble" (training several members on different
random subsets of the training pool via ``data.train_subsample_seed``, see
``configs/config_script_01.yaml``) actually meaningful: N separate checkpoints only
become an ensemble once their predictions are averaged and scored together.

Usage (from the repo root):
    python -m downstream_apps.solar_wind.4_ensemble_eval \
        --config downstream_apps/solar_wind/configs/config_script_01.yaml \
        --checkpoints checkpoints/best-epoch=05-val_loss=0.1234.ckpt \
                      checkpoints/best-epoch=06-val_loss=0.1180.ckpt \
                      checkpoints/best-epoch=04-val_loss=0.1201.ckpt

Every checkpoint must have been trained from the same ``--config`` (same model/LoRA
section, same channels) so predictions line up on the same validation samples; only
``data.max_samples``/``data.train_subsample_seed`` are expected to differ between the
runs that produced them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from downstream_apps.solar_wind.configs import load_solar_wind_config
from downstream_apps.solar_wind.datasets.loaders import build_solar_wind_dataloaders
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.utils import apply_peft_lora, build_scalers, load_pretrained_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config the checkpoints were trained from (same model/LoRA section).")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="One or more Lightning checkpoint paths, one per ensemble member.")
    return parser.parse_args()


def build_val_loader(cfg, scalers):
    """Same val split every member was scored against.

    Shares ``datasets/loaders.py`` with the training script rather than rebuilding the
    wiring. That sharing is the point: this function previously omitted ``label_transform``,
    so a checkpoint trained on standardized targets would have been scored against raw km/s
    and reported an RMSE ~400 too large for reasons that had nothing to do with the model.
    """
    _, val_loader = build_solar_wind_dataloaders(cfg, scalers)
    return val_loader


def load_member(cfg, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Rebuild the exact architecture build_model() would have trained, then load weights.

    Loading via the raw state_dict (rather than LightningModule.load_from_checkpoint)
    avoids re-deriving LightningModule __init__ args (metrics, lr, ...) that aren't needed
    for inference — only the wrapped nn.Module's weights are.
    """
    from workshop_infrastructure.models.finetune_models import HelioSpectformer1D

    model = HelioSpectformer1D.from_config(
        cfg.model,
        num_outputs=1,
        dtype=cfg.dtype,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    )
    load_pretrained_weights(model, cfg.model.pretrained_path)
    if cfg.model.freeze_backbone:
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
    if cfg.model.use_lora:
        model = apply_peft_lora(model, cfg.model.lora_config)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    # LightningModule saves the wrapped model under "model.<name>"; strip that prefix.
    prefix = "model."
    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    model.load_state_dict(stripped, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, val_loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    preds, targets = [], []
    for batch in val_loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        preds.append(model(batch).reshape(-1).float().cpu())
        targets.append(batch["forecast"].reshape(-1).float().cpu())
    return torch.cat(preds), torch.cat(targets)


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def main() -> None:
    args = parse_args()
    cfg = load_solar_wind_config(args.config)
    ensure_assets(cfg, which=["scalers", "weights"])
    scalers = build_scalers(info=cfg.data.scalers_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = build_val_loader(cfg, scalers)

    member_preds = []
    targets = None
    for ckpt_path in args.checkpoints:
        model = load_member(cfg, ckpt_path, device)
        preds, batch_targets = collect_predictions(model, val_loader, device)
        if targets is None:
            targets = batch_targets
        member_rmse = rmse(preds, batch_targets)
        print(f"[MEMBER] {Path(ckpt_path).name}: val RMSE = {member_rmse:.3f} km/s")
        member_preds.append(preds)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ensemble_pred = torch.stack(member_preds, dim=0).mean(dim=0)
    ensemble_rmse = rmse(ensemble_pred, targets)
    print(f"[ENSEMBLE] {len(args.checkpoints)} members, averaged prediction: val RMSE = {ensemble_rmse:.3f} km/s")


if __name__ == "__main__":
    main()
