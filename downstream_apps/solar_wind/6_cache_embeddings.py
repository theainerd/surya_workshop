#!/usr/bin/env python3
"""
Cache frozen-backbone embeddings so that head-only experiments cost microseconds, not hours.

WHY THIS EXISTS
---------------
Wall-clock in this app is ~100% I/O, not compute. Measured on completed runs: the
14-parameter linear baseline (64.2 min / 800 sample-passes) and the 366M-parameter LoRA
model (79.6 min / 1000 sample-passes) both ran at **4.8 s per sample-pass**. Every second
goes into reading a ~545 MB NetCDF frame and normalizing a 13x4096x4096 stack. No
architecture change can speed that up — only not re-reading the frames can.

A frozen backbone re-reads 1 GB per sample per epoch to recompute a value that never
changes. This script computes it once and writes 1280 floats (~5 KB) per sample instead:
a ~200,000x reduction in bytes touched per epoch, which turns a 40-minute run into ~2
seconds and makes lag sweeps, cross-validation and bootstrap CIs affordable.

WHY THE CACHE IS MATHEMATICALLY EXACT (not an approximation)
-----------------------------------------------------------
Requires ``model.pooling: global_average``. The full head is

    head_unembed( dropout( mean_over_tokens( head_linear(tokens) ) ) )

and because ``head_linear`` is linear and the pooling is a mean,

    mean(head_linear(tokens)) == head_linear(mean(tokens))

so caching ``mean(tokens)`` loses nothing — the head can still be trained afterwards, in
full, on the cached vector. ``--verify`` checks this identity numerically against a real
forward pass before you trust any downstream number.

Under ``pooling: class_token`` this does NOT hold: the CLS token is a *trainable input* to
the backbone (``forward_with_cls_token``), so the embedding depends on head weights and is
not cacheable at all. The script refuses to run in that configuration.

Caching is also only valid because the input pipeline is deterministic as configured:
``drop_hmi_probability: 0.0``, ``drop_rate: 0.0``, ``learned_flow: false``, and
``_base_dataset_kwargs()`` never passes ``num_mask_aia_channels`` or ``random_vert_flip``,
so both keep their defaults (0, False). The script re-checks these and refuses otherwise.

WHAT IT WRITES
--------------
One ``.npz`` per split containing, for every sample:
  emb_d4, emb_d6, emb_d8, emb_d10  (N, 1280)  mean-pooled tokens after 4/6/8/10 blocks.
                                              Four depths from one pass, so "does solar wind
                                              speed need all 10 blocks?" is answerable for
                                              free — and if d6 ties d10, 40% of the backbone
                                              forward is deletable.
  feat13                           (N, 13)    the linear baseline's own features, so the
                                              matched baseline is free to evaluate too.
  target                           (N,)       the label, in whatever units the config sets.
  ts                               (N,)       ISO timestamp, the join key for relabelling at
                                              a different Sun-Earth propagation lag.

Usage (from the repo root):
    # Tier 1 (train) + Tier 2 (val), and check the cache is exact
    python -m downstream_apps.solar_wind.6_cache_embeddings \
        --config downstream_apps/solar_wind/configs/config_experiment.yaml --verify

    # Tier 3 (temporal holdout), reusing the same config
    python -m downstream_apps.solar_wind.6_cache_embeddings \
        --config downstream_apps/solar_wind/configs/config_experiment.yaml \
        --only val --val-index downstream_apps/solar_wind/data/hf_solar_wind/holdout_late_designed.csv \
        --name holdout_late
"""

from __future__ import annotations

import argparse
import os

# Must precede the torch import; see the note in 3_finetune_template_1D.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
from functools import partial
from itertools import chain
from pathlib import Path

import lightning as L
import numpy as np
import torch

from downstream_apps.solar_wind.configs import load_solar_wind_config
from downstream_apps.solar_wind.datasets.loaders import build_solar_wind_dataloaders
from downstream_apps.solar_wind.models.simple_baseline import destandardize_channels
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
from workshop_infrastructure.utils import build_scalers, load_pretrained_weights

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_experiment.yaml"

# Depths to capture, as 1-based block counts over chain(spectral_blocks, attention_blocks).
# Block 10 is the backbone's final output, i.e. what the head normally consumes.
CAPTURE_DEPTHS = (4, 6, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--only", choices=("train", "val", "both"), default="both",
                        help="Which split(s) to cache. Use 'val' with --val-index to cache "
                             "an extra evaluation split without a second config file.")
    parser.add_argument("--val-index", type=str, default=None,
                        help="Override data.ds_val_index_path (e.g. a temporal holdout CSV).")
    parser.add_argument("--name", type=str, default=None,
                        help="Output stem for the val split (default: 'val'). Set this when "
                             "using --val-index so the file is not overwritten.")
    parser.add_argument("--out-dir", type=str, default="cache/embeddings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many batches. For smoke tests.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override training.num_workers. This is the one knob that moves "
                             "wall-clock here: the pass is I/O-bound, so workers fetching "
                             "frames in parallel is what hides the ~70 s per cold S3 frame. "
                             "Each worker holds a ~873 MB stack and prefetches 2, so raising "
                             "it trades RAM for throughput.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.batch_size.")
    parser.add_argument("--no-save", action="store_true",
                        help="Run the pass but write nothing. For timing measurements.")
    parser.add_argument("--s3-concurrency", type=int, default=None,
                        help="Override data.s3_boto3_max_concurrency (parallel part downloads "
                             "per file). At the config default of 4 this pass sustained only "
                             "~16 MB/s, so this is worth tuning before a long run.")
    parser.add_argument("--verify", action="store_true",
                        help="Check the cached embedding reproduces a full forward pass "
                             "exactly. Run this once before trusting any downstream result.")
    return parser.parse_args()


def check_cacheable(cfg) -> None:
    """Refuse to build a cache that would silently be wrong.

    Each condition below is one under which a cached embedding would not be a fixed function
    of the input sample, making every downstream probe fit noise.
    """
    problems = []
    if cfg.model.pooling != "global_average":
        problems.append(
            f"model.pooling is {cfg.model.pooling!r}; only 'global_average' is cacheable. "
            "Under 'class_token' the CLS token is a trainable input to the backbone, so the "
            "embedding depends on head weights."
        )
    if cfg.model.learned_flow:
        problems.append("model.learned_flow is true; the flow module perturbs the input stack.")
    if cfg.drop_hmi_probability:
        problems.append(
            f"training.drop_hmi_probability is {cfg.drop_hmi_probability}; random channel "
            "dropping makes the embedding non-deterministic."
        )
    if cfg.model.use_lora:
        # Not fatal: LoRA params exist but we never load or apply adapter weights here, so
        # the backbone is the pretrained one. Worth saying out loud, though.
        print("[CACHE] note: model.use_lora is true in the config, but no adapters are "
              "applied here — the cache is of the PRETRAINED backbone, which is what a "
              "frozen-backbone probe needs. LoRA cannot use this cache.")
    if problems:
        raise SystemExit("[CACHE] refusing to build a cache:\n  - " + "\n  - ".join(problems))


def build_backbone(cfg, device: torch.device) -> HelioSpectformer1D:
    """Rebuild the architecture the training script would use, with pretrained weights, frozen."""
    model = HelioSpectformer1D.from_config(
        cfg.model,
        num_outputs=1,
        dtype=cfg.dtype,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    )
    load_pretrained_weights(model, cfg.model.pretrained_path)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def register_depth_hooks(model: HelioSpectformer1D, captured: dict) -> list:
    """Hook the blocks at CAPTURE_DEPTHS, storing each one's mean-pooled output in ``captured``.

    ``HelioSpectFormer`` iterates ``chain(backbone.blocks_spectral_gating,
    backbone.blocks_attention)`` (2 spectral + 8 attention = 10), so a depth of N means the
    output of chained block index N-1. Pooling inside the hook is what keeps this cheap: the
    token tensor is (B, 65536, 1280) and would be ~336 MB per sample to keep, while its mean
    over tokens is 1280 floats.
    """
    inner = model.backbone.backbone
    blocks = list(chain(inner.blocks_spectral_gating, inner.blocks_attention))

    def hook(_module, _inputs, output, depth: int):
        # Blocks return the token tensor (B, N, D); mean over the token axis.
        captured[f"emb_d{depth}"] = output.mean(dim=1).float().cpu()

    handles = []
    for depth in CAPTURE_DEPTHS:
        handles.append(blocks[depth - 1].register_forward_hook(partial(hook, depth=depth)))
    return handles


def baseline_features(batch: dict, cfg, scalers) -> torch.Tensor:
    """The linear baseline's own 13-d features, computed on CPU alongside the embedding.

    Mirrors ``RegressionSolarWindModel.forward``: de-standardize to signum-log space, take
    the absolute spatial mean per channel, and flatten channel x time. Computing it here
    means the matched baseline never needs its own pass over the data.
    """
    destandardized = destandardize_channels(batch, cfg.data.channels, scalers)
    per_channel = destandardized["ts"].abs().mean(dim=[3, 4])  # (B, C, T)
    return per_channel.reshape(per_channel.shape[0], -1).float()


@torch.no_grad()
def cache_split(loader, model, cfg, scalers, device, label: str, limit: int | None) -> dict:
    """Run one forward pass over ``loader`` and return the stacked cache arrays."""
    captured: dict[str, torch.Tensor] = {}
    handles = register_depth_hooks(model, captured)
    collected: dict[str, list] = {f"emb_d{d}": [] for d in CAPTURE_DEPTHS}
    collected["feat13"], collected["target"], collected["ts"] = [], [], []

    n_batches = len(loader) if limit is None else min(limit, len(loader))
    started = last_tick = time.time()
    recent: list[float] = []
    try:
        for i, batch in enumerate(loader):
            if limit is not None and i >= limit:
                break

            # CPU first: destandardize_channels clones the (B,C,T,H,W) stack, and doing that
            # on the GPU would cost ~1.7 GB of VRAM per batch for nothing.
            collected["feat13"].append(baseline_features(batch, cfg, scalers))
            collected["target"].append(batch["forecast"].float().cpu())
            collected["ts"].extend(batch["ds_index"])

            gpu_batch = {
                "ts": batch["ts"].to(device, non_blocking=True),
                "time_delta_input": batch["time_delta_input"].to(device, non_blocking=True),
            }
            captured.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                # Call the backbone directly, not the model: the head is what we train later.
                model.backbone(gpu_batch)

            missing = [f"emb_d{d}" for d in CAPTURE_DEPTHS if f"emb_d{d}" not in captured]
            if missing:
                raise RuntimeError(f"hooks did not fire for {missing}; block indexing is wrong")
            for key, value in captured.items():
                collected[key].append(value)

            # Report the MARGINAL time per batch, not the cumulative average. The average is
            # actively misleading here: worker startup dominates the first batch, the
            # prefetch buffer filled during that startup then empties in ~3 s/batch, and only
            # after it drains does the number reflect the sustained cold-fetch rate. A
            # cumulative average was still falling through all of that and read 34 s/batch
            # where the sustained rate was 68.
            now = time.time()
            done = i + 1
            marginal = now - last_tick
            last_tick = now
            recent.append(marginal)
            sustained = sum(recent[-10:]) / len(recent[-10:])
            print(f"[CACHE:{label}] batch {done}/{n_batches}  "
                  f"{marginal:5.1f} s  (last-10 mean {sustained:5.1f} s)  "
                  f"eta {sustained * (n_batches - done) / 60:5.1f} min", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    out = {k: torch.cat(v).numpy() for k, v in collected.items() if k != "ts"}
    out["ts"] = np.array(collected["ts"])
    return out


@torch.no_grad()
def verify_exactness(loader, model, cfg, device) -> None:
    """Assert head(cached_mean) == full forward, the identity the whole cache rests on.

    If this fails, ``mean(head_linear(t)) == head_linear(mean(t))`` does not hold for this
    configuration and every downstream probe result would be meaningless.
    """
    batch = next(iter(loader))
    gpu_batch = {
        "ts": batch["ts"].to(device),
        "time_delta_input": batch["time_delta_input"].to(device),
    }
    captured: dict[str, torch.Tensor] = {}
    handles = register_depth_hooks(model, captured)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            direct = model(gpu_batch)  # full model: backbone + pooling + head
            cached_mean = captured["emb_d10"].to(device)
            # Reproduce the head from the cached vector. dropout is identity in eval mode.
            hidden = model.head_linear(cached_mean) if cfg.model.penultimate_linear_layer else cached_mean
            from_cache = model.head_unembed(hidden).squeeze(dim=1)
    finally:
        for handle in handles:
            handle.remove()

    direct_f, cache_f = direct.float().cpu(), from_cache.float().cpu()
    abs_err = (direct_f - cache_f).abs().max().item()
    scale = direct_f.abs().max().clamp(min=1e-6).item()
    print(f"\n[VERIFY] full forward     : {direct_f.tolist()}")
    print(f"[VERIFY] from cached emb  : {cache_f.tolist()}")
    print(f"[VERIFY] max abs error    : {abs_err:.3e}  (relative {abs_err / scale:.3e})")
    # bf16 carries ~3 decimal digits, and the two paths sum 65536 tokens in different orders,
    # so exact equality is not expected; 1% relative is comfortably within that noise.
    if abs_err / scale > 1e-2:
        raise SystemExit("[VERIFY] FAILED — cached embeddings do not reproduce the forward "
                         "pass. Do not trust any result computed from this cache.")
    print("[VERIFY] PASSED — the cache reproduces the full forward pass.\n")


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_solar_wind_config(args.config)
    L.seed_everything(cfg.seed, workers=True)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.s3_concurrency is not None:
        cfg.data.s3_boto3_max_concurrency = args.s3_concurrency
    check_cacheable(cfg)
    ensure_assets(cfg, which=["scalers", "weights"])
    scalers = build_scalers(info=cfg.data.scalers_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_solar_wind_dataloaders(
        cfg, scalers, val_index_path=args.val_index, num_workers=args.num_workers
    )
    for label, loader in (("train", train_loader), ("val", val_loader)):
        dropped = len(loader.dataset) - len(loader) * cfg.batch_size
        note = f"  ({dropped} dropped by drop_last)" if dropped else ""
        print(f"[CACHE] {label}: {len(loader.dataset)} samples -> {len(loader)} batches{note}")

    model = build_backbone(cfg, device)
    if args.verify:
        verify_exactness(val_loader, model, cfg, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    if args.only in ("train", "both"):
        targets.append(("train", train_loader, "train"))
    if args.only in ("val", "both"):
        targets.append(("val", val_loader, args.name or "val"))

    for label, loader, stem in targets:
        arrays = cache_split(loader, model, cfg, scalers, device, label, args.limit)
        if args.no_save:
            print(f"[CACHE] --no-save: discarding {len(arrays['ts'])} {label} rows")
            continue
        path = out_dir / f"{stem}.npz"
        np.savez_compressed(
            path,
            target_mean=np.array(cfg.data.target_mean if cfg.data.target_mean is not None else np.nan),
            target_std=np.array(cfg.data.target_std if cfg.data.target_std is not None else np.nan),
            **arrays,
        )
        size_mb = path.stat().st_size / 1e6
        print(f"[CACHE] wrote {path}  n={len(arrays['ts'])}  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
