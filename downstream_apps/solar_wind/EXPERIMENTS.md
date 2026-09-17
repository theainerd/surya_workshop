# Solar wind speed (V) regression — experiment record

**Bottom line: neither model predicts solar wind speed. LoRA reaches the no-information floor;
the linear baseline falls ~25 km/s below it. The ~26 km/s gap between them is real and
reproducible, but it measures calibration, not extracted signal.**

All numbers reproduced from `runs/*/version_0/metrics.csv`.

---

## Dataset

Two sources, joined on time. Nothing is generated or simulated.

| | source | size | span |
|---|---|---|---|
| **Inputs** — SDO images | `s3://nasa-surya-bench` (public, anon), indexed by `data/indices/surya_aws_s3_{train,val}.csv` | 467,400 train / 21,600 val frames; **561 MB each** on disk, 13x4096x4096 float32 (873 MB in memory) | 12-min cadence |
| **Labels** — solar wind speed | `nasa-ibm-ai4science/Surya-bench-solarwind` (HF), at `data/hf_solar_wind/{train,validation}.csv` | 69,957 train / 3,429 val hourly rows | train 2010-05..2019-12; val 2011-01..2019-01 |

Each frame is 13 channels: 8 AIA EUV wavelengths (94, 131, 171, 193, 211, 304, 335, 1600 A) +
5 HMI magnetic/velocity components. Label columns are `timestamp, V, BX_GSE, BY_GSM, BZ_GSM, N`;
only **V** (speed, km/s) is used. Train V = 418.6 +/- 93.9 km/s, range 240-800.

### How a training sample is built

1. **Join.** `SolarWindDSDataset` aligns the two indices with `pd.merge_asof(direction="nearest")`
   and a **6-minute tolerance** (half the 12-min SDO cadence, so the closest possible frame).
   Rows with no frame in tolerance are dropped — **only 74% of `validation.csv` rows survive**.
2. **Normalize inputs.** Per channel: signum-log `sign(x)*log(1+|x|)`, then z-score using
   `assets/scalers.yaml`.
3. **Normalize the target.** `(V - 418.6) / 93.9`, statistics from the **full training split
   only**. Applied identically to train and val as a shared kwarg, so the two can never end up in
   different units.
4. No augmentation is active (`drop_hmi_probability: 0`, `num_mask_aia_channels: 0`,
   `random_vert_flip: False`), so the pipeline is deterministic given a sample.

Full uncapped training is not affordable: ~70k samples x 561 MB is ~39 TB of reads. Hence the
designed subsets below.

### How the designed subsets are built

Row selections over the label CSVs using pandas only — no code changes, so the sampling design
is auditable in the files themselves (`data/hf_solar_wind/*_designed.csv`).

- **Tier 1 (train, 250).** Index stride over `train.csv` (`iloc[::233]` -> 301 candidates), giving
  **>=9.7 d spacing** across 2010-2019; 250 survive the 6-min match. Spacing is set well above V's
  ~1.8 d decorrelation time so **n_eff ~= n**, which is precisely what the original runs got wrong.
- **Tiers 2 and 3 (val 32, holdout 12).** **Match first, then thin** — the reverse order costs 40%
  of the samples, because the 74% that match are not evenly spread. Construct the dataset over the
  full CSV, take the matched timestamps, then greedily keep samples **>=4 d apart** (Tier 2) or
  **>=1 d** (Tier 3), trimming to an even count since `drop_last=True` at `batch_size=2`.
- `max_samples` / `max_val_samples` / `train_subsample_seed` are left **null** in the config: the
  CSVs *are* the design, and capping would re-impose the chronological prefix that caused the
  original defect. `--max-samples` + `--train-subsample-seed` vary it per run for categories 1-2,
  the seed drawing a random subset spread over all years rather than a prefix.

## Results — 6 runs

4 epochs, `batch_size=2`, standardized target, same 32 independent validation samples
(2011-2019, ≥4 d apart). RMSE/MAE in km/s. `skill = 1 - MSE/MSE_climatology`.

| run | n | ep | RMSE | MAE | r | skill | trajectory |
|---|---|---|---|---|---|---|---|
| `baseline_n50_s101` | 50 | 3 | 101.2 | 85.4 | −0.129 | −0.880 | 102→102→101→101 |
| `baseline_n50_s102` | 50 | 3 | 101.3 | 85.6 | −0.134 | −0.886 | 102→102→101→101 |
| `baseline_nall` | 250 | 3 | 96.9 | 81.4 | −0.130 | −0.723 | 100→98→97→97 |
| `lora_n50_s101` | 50 | 3 | 75.0 | 61.8 | +0.014 | −0.034 | 85→80→102→75 |
| `lora_n50_s102` | 50 | 3 | **68.9** | 60.0 | **+0.241** | **+0.127** | 173→86→71→69 |
| `lora_nall` | 250 | 3 | 70.9 | 57.9 | +0.125 | +0.078 | 112→73→123→71 |

**Reference lines:** climatology **73.8** · oracle constant **70.3** km/s.
**Significance:** |r| must exceed **0.364** at n=32.

## The three categories

| category | verdict | evidence |
|---|---|---|
| **3. baseline vs LoRA**, identical data | **solid** | LoRA wins by **29.3 km/s** (n=50) and **26.0 km/s** (n=250) — ~180x the baseline's own noise |
| **2. training composition** | **solid** | Seed spread at n=50: baseline **0.16 km/s**, LoRA **6.10 km/s**. LoRA is ~38x more sensitive to *which* samples it sees |
| **1. amount of data** | **not resolvable** | LoRA effect 1.1 km/s vs 6.10 km/s composition noise. `lora_n50_s102` (68.9) **beats** `lora_nall` (70.9) with 5x less data |

Two-member LoRA ensemble: **72.0 km/s — worse than its better member (68.9)**. With members
that scattered, averaging drags toward the weaker one.

## Why "no skill" despite LoRA's 68.9 km/s

- Every |r| is below 0.364. The largest (+0.241) is from the wildest trajectory and is not
  reproduced by seed 101 (+0.014) or n=250 (+0.125).
- 68.9 km/s merely matches the **oracle constant** (70.3) — a predictor handed the validation
  window's own mean. That is calibration, not prediction.
- The baseline sits ~25 km/s *below* climatology with r ~ -0.13, and train loss ~= val loss
  throughout: a genuine null, not overfitting. Whole-disk mean brightness does not linearly
  predict V, which is physically unsurprising.

## Why the nulls are measurements, not a broken pipeline

Optimizer descends and train tracks val · target transform works (`train_loss` 5.08 -> 0.77 in
ten steps, vs the old baseline moving 0.07 km/s in four epochs) · `sqrt(val_loss)` ==
`val_epoch_rmse` to four decimals across independent code paths · 66 tests pass, including a
harness that provably finds planted signal and rejects noise.

## Caveats

1. **`deterministic: false` forced** — `warn` disables memory-efficient attention and the LoRA
   backward pass OOMs (5 GiB needed, 43.9/44.4 GiB held, 65,536 tokens/sample). Not
   bit-reproducible.
2. **n=32 validation** is the official split's hard ceiling (9 blocks, 143 days). Demonstrated:
   32 samples miss a real moderate effect that 250 detect.
3. **LoRA instability** — "best of 4 epochs on 32 samples" is a selection effect about the size
   of the effects reported.
4. **One run per configuration** except the n=50 pairs — which is what exposed category 1.

## Next, in priority order

1. **Test the physics.** Every run pairs an SDO image with the *simultaneous* in-situ V, but wind
   at L1 left the Sun 2-6 days earlier. The lag sweep needs relabelling only, no retraining, and
   separates real coronal signal (skill peaking at 2-5 d, bump at 27 d) from persistence.
2. **Score on Tier 1 blocked CV** (250 independent out-of-fold predictions, ~8x the power)
   instead of the 32-sample holdout. `7_probe_from_cache.py` is built and validated.
3. **Repeat seeds before trusting small effects.** Category 1 is the cautionary example.

---

## Appendix — why the first 8 runs were void

Best result ever (40.2 km/s) **lost to a constant predictor (32.4 km/s)**. Four defects:

1. **n_eff 3-8, not 50-250.** `train_subsample_seed: null` takes a chronological prefix, so
   "n=250" was 250 *consecutive hours*. V decorrelates in ~1.8 d (r = 0.719/0.418/0.203/0.082 at
   1/2/3/4 d; 0.481 at 27 d, the Carrington rotation).
2. **Validation had no signal** — 100 consecutive hours, std 32.4 km/s, n_eff ~ 2.
3. **Target never normalized.** V is 418.6 +/- 93.9 km/s while `head_unembed` starts at ~0. The
   baseline shifts its bias ~lr/step, so 200 steps covered 0.02 km/s of a 418 km/s gap — it
   needed ~4M steps. Log confirms: RMSE moved 465.67 -> 465.60 in four epochs.
4. **No baseline reference logged**, so "worse than a constant" went unnoticed for 8 runs.

Also: `val_metric_rmse` was the mean of per-batch RMSEs (49.55 logged vs 59.72 true), and
`val_metric_rrse` divided by the variance of 2 samples (read 17-227). Both fixed.

## Appendix — the redesigned apparatus

**Designed splits** (`data/hf_solar_wind/*_designed.csv`), spaced against the 1.8 d
decorrelation time, matched-first-then-thinned:

| tier | n | spacing | span | V (km/s) | use |
|---|---|---|---|---|---|
| 1 `train_designed` | 250 | >=9.7 d | 2010-05..2019-12 | 418.2 +/- 93.6 | training; blocked CV |
| 2 `validation_designed` | 32 | >=4.0 d | 2011-01..2019-01 | 396.2 +/- 70.3 | held-out scoring |
| 3 `holdout_late_designed` | 12 | >=1 d | Jan 2019 | 409.5 +/- 103.8 | qualitative only |

**Fixes:** target standardization via the dataset's existing unused `label_transform` hook ·
`rrse` removed, MAE + Pearson r added, epoch-level accumulation · one shared loader (fixed a
latent bug where `4_ensemble_eval.py` would have scored a z-unit checkpoint in raw km/s) ·
`s3_boto3_max_concurrency: 4->16` gave an **8x I/O speedup** (68 s -> 8.6 s per batch), the only
change that moved wall-clock, since the 14-param baseline and 366M LoRA both ran at 4.8 s per
sample-pass.

**Built and validated but unused** (deferred, worth returning to): embedding cache
(`6_cache_embeddings.py`, verified **bit-identical** to a full forward pass) and the probe
(`7_probe_from_cache.py` — controls, lag sweep, blocked CV; finds planted signal, rejects noise).
A ridge fit on the pooled embedding is the *optimal* version of what `head_linear`+`head_unembed`
can express, so it settles the frozen-backbone question far faster than training runs.

## Status

- [x] Diagnosis of the 8 prior runs
- [x] Designed 3-tier splits; target normalization; metric fixes; shared loader — 66 tests pass
- [x] Cache + probe built and validated (deferred, not used for these results)
- [x] **6 runs complete** — all three categories, baseline and LoRA in each
- [x] Results recorded above
- [ ] Lag sweep (physics test)
- [ ] Tier 1 blocked-CV scoring
- [ ] Nothing committed to git — all changes are uncommitted working-tree edits
