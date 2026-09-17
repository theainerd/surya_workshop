#!/usr/bin/env python3
"""
Probe the cached Surya embeddings for solar-wind-speed skill.

Reads the ``.npz`` files from ``6_cache_embeddings.py``. The backbone is frozen, so its
embedding of a sample never changes and every fit here is a head-only ridge solve on a
1280-d vector rather than a re-read of a 1 GB frame. That is what makes the controls,
cross-validation, permutation tests and bootstrap intervals below affordable at all — they
are the whole point of running from a cache.

STRUCTURE — controls first, then a pre-specified primary test, then exploration
------------------------------------------------------------------------------
0. CONTROLS. A positive control (predict V(t) from V(t-24h) alone) must recover the
   autocorrelation measured directly in the OMNI series. A negative control (permute the
   labels and re-run the entire nested CV) must return ~zero skill. If either fails, the
   harness is broken and nothing below it means anything, so it is checked first and the
   run aborts on failure.

1. PRIMARY TEST, pre-specified before looking at results: lag 0, ``emb_d10`` (the full
   backbone), all training samples. This is the configuration the benchmark itself implies.
   Reported on Tier 1 by nested blocked CV (n~250) and independently on Tier 2 (n~32), with
   a permutation p-value and a bootstrap CI.

2. CATEGORIES c1-c3, the three requested comparisons.

3. EXPLORATORY: the lag sweep (c0) and the depth ablation (c4). Labelled exploratory because
   each sweeps many configurations and reports the best, so its headline number carries a
   multiple-comparisons bias that the primary test above does not.

WHY THE STATISTICS ARE SET UP THIS WAY
--------------------------------------
* Blocked CV with a 5-day embargo, never random k-fold. V decorrelates over ~1.8 days, so a
  randomly held-out sample usually has a near-duplicate in the training half; random folds
  would report skill that is really just autocorrelation.
* lambda chosen by *nested* inner blocked CV, so no reported score saw the lambda picked for it.
* Features standardized using training statistics only, per fit. Without this the ridge
  penalty is not scale-invariant and depths with different activation scales are not
  comparable at a shared lambda grid.
* "Same validation set" is implemented as a fixed fold structure: every configuration in
  c1-c3 is scored on exactly the same held-out samples, so differences come from the
  training side alone.
* Skill is reported against climatology (each fold's *training* mean), which is the null the
  first eight runs silently failed to beat.

WHAT A NULL RESULT HERE WOULD AND WOULD NOT PROVE
-------------------------------------------------
Ridge is not a weak stand-in for the real head — it is the strongest fair one. On a frozen
backbone the head is ``head_unembed(dropout(mean(head_linear(tokens))))``: a ``Linear(1280,1280)``
followed by a ``Linear(1280,1)``, which composes to a single affine map from the pooled 1280-d
embedding to a scalar. Ridge fits the *optimal* map in exactly that function class, with its
regularization tuned by cross-validation. Therefore:

* If the probe finds skill, a trained linear head can reach it (SGD may need tuning to get there).
* If the probe finds none, no amount of head training on this frozen backbone will, because the
  probe has already searched the whole function class that head can express.
* It says nothing about **LoRA**, which modifies the backbone and so changes the embedding
  itself. LoRA cannot be evaluated from this cache and needs a real training run.

Usage (from the repo root):
    python -m downstream_apps.solar_wind.7_probe_from_cache
    python -m downstream_apps.solar_wind.7_probe_from_cache --experiments c0 --no-wandb
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).parent
DEFAULT_CACHE = APP_DIR.parent.parent / "cache" / "embeddings"
DEFAULT_OMNI = APP_DIR / "data" / "hf_solar_wind"

# Sun-Earth propagation lags to sweep, in days. 0 is the pairing every earlier run used;
# 2-5 brackets the physical transit time (1 AU at 800 / 400 / 300 km/s = 2.2 / 4.3 / 5.8 d);
# 27 is the Carrington-rotation control.
LAG_DAYS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 14.0, 20.0, 27.0, 34.0)

LAMBDA_GRID = tuple(10.0 ** np.arange(-1, 8))
EMBARGO_DAYS = 5.0
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 5

# The primary hypothesis, fixed before any result was inspected.
PRIMARY_LAG = 0.0
PRIMARY_FEATURE = "emb_d10"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict:
    """Load one embedding cache, sorted by time, with targets restored to km/s."""
    with np.load(path, allow_pickle=False) as data:
        cache = {k: data[k] for k in data.files}
    mean, std = float(cache.pop("target_mean")), float(cache.pop("target_std"))
    n = len(cache["ts"])
    cache["time"] = pd.to_datetime(cache["ts"].astype(str)).values
    # Undo whatever units the config trained in, so everything below is km/s and directly
    # comparable to the OMNI series and to the printed reference lines.
    cache["v_kms"] = cache["target"] * std + mean if np.isfinite(std) else cache["target"]
    order = np.argsort(cache["time"])
    return {k: (v[order] if getattr(v, "shape", (0,))[:1] == (n,) else v) for k, v in cache.items()}


def load_omni(omni_dir: Path) -> pd.Series:
    """One continuous hourly V series spanning every split, for relabelling at a lag.

    The train and validation splits are disjoint in time, so concatenating them gives the
    fullest coverage available (~84.5k hours, ~13% missing and therefore NaN).
    """
    frames = [pd.read_csv(omni_dir / name) for name in ("train.csv", "validation.csv")]
    df = pd.concat(frames, ignore_index=True)
    df["t"] = pd.to_datetime(df.timestamp)
    return df.sort_values("t").drop_duplicates("t").set_index("t")["V"].asfreq("1h")


def relabel(times: np.ndarray, omni: pd.Series, lag_days: float) -> np.ndarray:
    """V(t + lag) in km/s for each t, NaN where OMNI has no measurement.

    Sample timestamps come from ``SolarWindDSDataset``'s ``ds_index``, which is the matched
    OMNI observation time and therefore already exactly on the hour, so the rounding here is
    a no-op safeguard rather than a real interpolation.
    """
    wanted = pd.DatetimeIndex(times) + pd.Timedelta(days=lag_days)
    return omni.reindex(wanted.round("h")).to_numpy(dtype=float)


def measured_autocorr(omni: pd.Series, lags_days) -> dict:
    """V's own autocorrelation at each lag, measured from the data rather than hardcoded.

    This is the curve the lag sweep is judged against: if embedding skill decays like this,
    the model is exploiting persistence, not reading the corona.
    """
    return {lag: float(omni.autocorr(lag=int(round(lag * 24)))) for lag in lags_days}


# ---------------------------------------------------------------------------
# Ridge and blocked cross-validation
# ---------------------------------------------------------------------------

def _standardize(x_fit: np.ndarray, x_apply: np.ndarray):
    """Z-score features using the FITTING set's statistics only."""
    mu, sd = x_fit.mean(axis=0), x_fit.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x_fit - mu) / sd, (x_apply - mu) / sd


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float):
    """Closed-form ridge with an unpenalized intercept, in the dual (n x n) form.

    With d = 1280 and n ~ 250 the dual is cheaper and better conditioned than the primal.
    Centring both sides keeps the intercept out of the penalty, so a large lambda shrinks the
    prediction toward the training mean — i.e. toward climatology, the right fallback.
    """
    x_mean, y_mean = x.mean(axis=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    alpha = np.linalg.solve(xc @ xc.T + lam * np.eye(len(yc)), yc)
    weights = xc.T @ alpha
    return weights, float(y_mean - x_mean @ weights)


def blocked_folds(times: np.ndarray, n_folds: int, embargo_days: float = EMBARGO_DAYS):
    """Yield ``(train_idx, test_idx)`` over contiguous time blocks, with an embargo.

    Each test fold is one contiguous stretch of time, and any training sample within
    ``embargo_days`` of a test sample is dropped outright rather than merely assigned to a
    different fold. Both properties are required: V decorrelates over ~1.8 days, so without
    them a held-out sample's near-duplicate sits in the training set.
    """
    order = np.argsort(times)
    for block in np.array_split(order, n_folds):
        if not len(block):
            continue
        gap = np.abs(times[:, None] - times[block][None, :]) / np.timedelta64(1, "D")
        train_idx = np.setdiff1d(np.where(~(gap < embargo_days).any(axis=1))[0], block)
        if len(train_idx):
            yield train_idx, block


def select_lambda(x: np.ndarray, y: np.ndarray, times: np.ndarray) -> float:
    """Choose lambda by blocked CV on the given (training-only) data."""
    best_lam, best_mse = LAMBDA_GRID[0], np.inf
    folds = list(blocked_folds(times, N_INNER_FOLDS))
    if not folds:
        return float(np.median(LAMBDA_GRID))
    for lam in LAMBDA_GRID:
        errors = []
        for tr, te in folds:
            weights, bias = ridge_fit(x[tr], y[tr], lam)
            errors.append(x[te] @ weights + bias - y[te])
        mse = float(np.mean(np.concatenate(errors) ** 2))
        if mse < best_mse:
            best_lam, best_mse = float(lam), mse
    return best_lam


def fit_predict(x_tr, y_tr, times_tr, x_te, lam: float | None = None):
    """Standardize on the training set, pick lambda if needed, fit, and predict ``x_te``."""
    xs_tr, xs_te = _standardize(x_tr, x_te)
    lam = select_lambda(xs_tr, y_tr, times_tr) if lam is None else lam
    weights, bias = ridge_fit(xs_tr, y_tr, lam)
    return xs_te @ weights + bias, lam


def out_of_fold(x, y, times, subsample=None, rng=None):
    """Nested blocked CV. Returns ``(y_true, y_pred, climatology)`` over the held-out folds.

    ``subsample`` restricts each fold's *training* portion to that many samples (drawn with
    ``rng``), which is how the data-amount and ensemble categories vary the training side
    while every configuration is scored on identical held-out samples.

    lambda is selected inside each outer fold's training portion only, and the climatology
    reference is that same portion's mean, so neither ever sees held-out data.
    """
    truths, preds, clims = [], [], []
    for train_idx, test_idx in blocked_folds(times, N_OUTER_FOLDS):
        if subsample is not None and subsample < len(train_idx):
            train_idx = rng.choice(train_idx, subsample, replace=False)
        pred, _ = fit_predict(x[train_idx], y[train_idx], times[train_idx], x[test_idx])
        truths.append(y[test_idx])
        preds.append(pred)
        clims.append(np.full(len(test_idx), y[train_idx].mean()))
    if not truths:
        return (np.array([]),) * 3
    return np.concatenate(truths), np.concatenate(preds), np.concatenate(clims)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def scores(y_true, y_pred, reference) -> dict:
    """RMSE / MAE / Pearson r / skill, in km/s where dimensional.

    ``skill = 1 - MSE_model / MSE_reference``. Positive beats the reference, 0 matches it,
    negative is worse than it.
    """
    if len(y_true) < 2:
        return {"n": len(y_true), "rmse": np.nan, "mae": np.nan, "r": np.nan, "skill": np.nan}
    err = y_pred - y_true
    mse_ref = float(np.mean((reference - y_true) ** 2))
    r = np.nan
    if y_pred.std() > 1e-9 and y_true.std() > 1e-9:
        r = float(np.corrcoef(y_pred, y_true)[0, 1])
    return {"n": int(len(y_true)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r": r,
            "skill": float(1.0 - np.mean(err ** 2) / mse_ref) if mse_ref > 0 else np.nan}


def bootstrap_ci(y_true, y_pred, reference, key: str, n_boot=2000, seed=0):
    """Percentile bootstrap CI, resampling samples independently.

    Legitimate only because the splits were *designed* independent (>=9.7 d apart for Tier 1,
    >=4 d for Tier 2, against a ~1.8 d decorrelation time). On a contiguous window this would
    badly understate the interval.
    """
    if len(y_true) < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = [scores(y_true[i], y_pred[i], reference[i])[key]
            for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot))]
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if not len(vals):
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt(row: dict, ci=None) -> str:
    ci_txt = "" if ci is None or not np.isfinite(ci[0]) else f"  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    return (f"n={row['n']:4d}  RMSE={row['rmse']:6.1f}  MAE={row['mae']:6.1f}  "
            f"r={row['r']:+.3f}  skill={row['skill']:+.3f}{ci_txt}")


def detectable_r(n: int) -> float:
    """Smallest correlation distinguishable from zero at ~95% for this sample size."""
    return 1.96 / np.sqrt(max(n - 3, 1))


# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------

class Run:
    """One WandB run per experiment group, or a no-op when logging is disabled."""

    def __init__(self, name, project, entity, enabled, config):
        self.name, self.rows, self.wandb = name, [], None
        if not enabled:
            return
        import wandb
        self.wandb = wandb
        wandb.init(project=project, entity=entity, name=name, config=config, reinit=True)
        print(f"  [wandb] {name}")

    def log(self, row: dict) -> None:
        self.rows.append(row)
        if self.wandb is not None:
            self.wandb.log({k: v for k, v in row.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})

    def finish(self) -> None:
        if self.wandb is None:
            return
        if self.rows:
            columns = sorted({k for r in self.rows for k in r})
            self.wandb.log({"results": self.wandb.Table(
                columns=columns, data=[[r.get(c) for c in columns] for r in self.rows])})
        self.wandb.finish()


# ---------------------------------------------------------------------------
# 0. Controls
# ---------------------------------------------------------------------------

def run_controls(train, omni, make_run, n_perm: int) -> float:
    """Verify the harness before trusting it. Returns the permutation-null 95th percentile.

    Positive control: predict V(t) from V(t-24h) alone, through the *same* nested blocked CV.
    It must land near the autocorrelation measured directly in the OMNI series. Too low means
    the pipeline is losing real signal; near 1.0 would mean it is leaking.

    Negative control: permute the labels and re-run the whole pipeline. Skill must collapse to
    ~0. Anything clearly positive means the CV leaks, and the permutation distribution it
    produces is also the honest yardstick for judging the real result.
    """
    print("\n" + "=" * 100)
    print("0  CONTROLS — is the measurement apparatus itself trustworthy?")
    print("=" * 100)
    run = make_run("c-controls", {"n_perm": n_perm})
    y = relabel(train["time"], omni, 0.0)
    ok = np.isfinite(y)

    # Consistency: relabelling at lag 0 must reproduce the label stored in the cache. This
    # checks the OMNI join against what the dataset actually fed the backbone.
    delta = np.nanmax(np.abs(y[ok] - train["v_kms"][ok]))
    print(f"  cache/OMNI consistency at lag 0 : max |diff| = {delta:.3f} km/s "
          f"({'OK' if delta < 0.5 else 'MISMATCH — the OMNI join is wrong'})")
    if delta >= 0.5:
        raise SystemExit("[CONTROL] aborting: relabelling does not reproduce the cached labels.")

    # Positive control.
    past = relabel(train["time"], omni, -1.0)
    both = ok & np.isfinite(past)
    truth, pred, clim = out_of_fold(past[both][:, None], y[both], train["time"][both])
    row = scores(truth, pred, clim)
    expected = float(omni.autocorr(lag=24))
    run.log(row | {"control": "positive_persistence_24h", "expected_r": expected})
    print(f"  POSITIVE control V(t-24h)       : {fmt(row)}")
    print(f"      expected r from OMNI autocorr = {expected:+.3f}  -> "
          f"{'OK' if abs(row['r'] - expected) < 0.15 else 'SUSPECT'}")
    if not np.isfinite(row["r"]) or row["r"] < expected - 0.20:
        raise SystemExit("[CONTROL] aborting: the pipeline cannot recover a known correlation.")

    # Negative control / permutation null.
    x = train[PRIMARY_FEATURE][ok]
    null = []
    for seed in range(n_perm):
        rng = np.random.default_rng(9000 + seed)
        t_p, p_p, c_p = out_of_fold(x, rng.permutation(y[ok]), train["time"][ok])
        null.append(scores(t_p, p_p, c_p)["skill"])
    null = np.asarray(null, dtype=float)
    threshold = float(np.percentile(null, 95))
    run.log({"control": "negative_permutation", "null_skill_mean": float(null.mean()),
             "null_skill_p95": threshold})
    print(f"  NEGATIVE control ({n_perm} label permutations)")
    print(f"      null skill: mean {null.mean():+.3f}  sd {null.std():.3f}  "
          f"95th pct {threshold:+.3f}")
    print(f"      -> a real result must exceed skill = {threshold:+.3f} to beat chance")
    if null.mean() > 0.05:
        raise SystemExit("[CONTROL] aborting: permuted labels produce positive skill — the "
                         "cross-validation leaks.")
    print("\n  controls PASSED — results below can be interpreted.")
    run.finish()
    return threshold


# ---------------------------------------------------------------------------
# 1. Primary test
# ---------------------------------------------------------------------------

def run_primary(train, val, omni, null_threshold, make_run) -> dict:
    """The pre-specified test: lag 0, full-depth embedding, all training data."""
    print("\n" + "=" * 100)
    print(f"1  PRIMARY TEST (pre-specified: lag {PRIMARY_LAG} d, {PRIMARY_FEATURE}, all training data)")
    print("=" * 100)
    run = make_run("primary", {"lag_days": PRIMARY_LAG, "feature": PRIMARY_FEATURE})
    out = {}

    y = relabel(train["time"], omni, PRIMARY_LAG)
    ok = np.isfinite(y)
    truth, pred, clim = out_of_fold(train[PRIMARY_FEATURE][ok], y[ok], train["time"][ok])
    row = scores(truth, pred, clim)
    ci = bootstrap_ci(truth, pred, clim, "skill")
    out["tier1"] = row
    run.log(row | {"split": "tier1_nested_cv", "skill_lo": ci[0], "skill_hi": ci[1]})
    print(f"  TIER 1, nested blocked CV : {fmt(row, ci)}")
    print(f"      climatology RMSE = {np.sqrt(np.mean((clim - truth) ** 2)):.1f} km/s")
    print(f"      smallest detectable r at n={row['n']} is {detectable_r(row['n']):.3f}")

    y_tr, y_va = y, relabel(val["time"], omni, PRIMARY_LAG)
    va_ok = np.isfinite(y_va)
    pred_v, lam = fit_predict(train[PRIMARY_FEATURE][ok], y_tr[ok], train["time"][ok],
                              val[PRIMARY_FEATURE][va_ok])
    clim_v = np.full(va_ok.sum(), y_tr[ok].mean())
    row_v = scores(y_va[va_ok], pred_v, clim_v)
    ci_v = bootstrap_ci(y_va[va_ok], pred_v, clim_v, "skill")
    out["tier2"] = row_v
    run.log(row_v | {"split": "tier2_holdout", "skill_lo": ci_v[0], "skill_hi": ci_v[1],
                     "lambda": lam})
    print(f"  TIER 2, independent holdout: {fmt(row_v, ci_v)}")
    print(f"      climatology RMSE = {np.sqrt(np.mean((clim_v - y_va[va_ok]) ** 2)):.1f} km/s")
    print(f"      smallest detectable r at n={row_v['n']} is {detectable_r(row_v['n']):.3f} "
          f"— this split is small, so Tier 1 is the better-powered estimate")
    print(f"\n  vs permutation null (skill must exceed {null_threshold:+.3f}): "
          f"tier1 {'PASS' if row['skill'] > null_threshold else 'FAIL'}, "
          f"tier2 {'PASS' if row_v['skill'] > null_threshold else 'FAIL'}")
    run.finish()
    return out


# ---------------------------------------------------------------------------
# 2. The three categories
# ---------------------------------------------------------------------------

def experiment_data_amount(train, omni, lag, feature, make_run, seeds=5) -> None:
    """c1 — varying the AMOUNT of training data, identical held-out samples throughout."""
    print("\n" + "=" * 100)
    print(f"c1  DATA AMOUNT (lag {lag} d, {feature}) — same held-out folds, varying n_train")
    print("    Every sample is >=9.7 d from its neighbours, so n_eff == n. The original sweep")
    print("    moved n from 50 to 250 while n_eff went 3 -> 8, and so measured nothing.")
    print("=" * 100)
    run = make_run("c1-dataamount", {"lag_days": lag, "feature": feature, "seeds": seeds})
    y = relabel(train["time"], omni, lag)
    ok = np.isfinite(y)
    x, yy, tt = train[feature][ok], y[ok], train["time"][ok]
    available = min(len(t) for t, _ in blocked_folds(tt, N_OUTER_FOLDS))
    print(f"  (each fold has {available} training samples available after the embargo)\n")
    for n in [v for v in (25, 50, 100, 150, 200, available) if v <= available]:
        rows = []
        for seed in range(seeds):
            truth, pred, clim = out_of_fold(x, yy, tt, subsample=n,
                                            rng=np.random.default_rng(1000 + seed))
            rows.append(scores(truth, pred, clim))
        rmse = np.array([r["rmse"] for r in rows]); skill = np.array([r["skill"] for r in rows])
        run.log({"n_train": n, "rmse_mean": rmse.mean(), "rmse_std": rmse.std(),
                 "skill_mean": skill.mean(), "skill_std": skill.std()})
        print(f"  n_train={n:4d}  RMSE={rmse.mean():6.1f} +/- {rmse.std():4.1f} km/s   "
              f"skill={skill.mean():+.3f} +/- {skill.std():.3f}   ({seeds} seeds)")
    run.finish()


def experiment_ensemble(train, omni, lag, feature, make_run, n_train=150, members=5) -> None:
    """c2 — varying training COMPOSITION at fixed size, then averaging the members."""
    print("\n" + "=" * 100)
    print(f"c2  ENSEMBLE (lag {lag} d, {feature}) — {members} different training subsets of "
          f"n={n_train}")
    print("=" * 100)
    run = make_run("c2-ensemble", {"lag_days": lag, "feature": feature,
                                   "n_train": n_train, "members": members})
    y = relabel(train["time"], omni, lag)
    ok = np.isfinite(y)
    x, yy, tt = train[feature][ok], y[ok], train["time"][ok]
    available = min(len(t) for t, _ in blocked_folds(tt, N_OUTER_FOLDS))
    n_train = min(n_train, available)
    preds, truth, clim = [], None, None
    for member in range(members):
        truth, pred, clim = out_of_fold(x, yy, tt, subsample=n_train,
                                        rng=np.random.default_rng(2000 + member))
        preds.append(pred)
        row = scores(truth, pred, clim)
        run.log(row | {"member": member})
        print(f"  member {member}  {fmt(row)}")
    mean_pred = np.mean(preds, axis=0)
    row = scores(truth, mean_pred, clim)
    ci = bootstrap_ci(truth, mean_pred, clim, "skill")
    run.log(row | {"member": -1})
    member_rmse = np.array([scores(truth, p, clim)["rmse"] for p in preds])
    print(f"\n  ENSEMBLE   {fmt(row, ci)}")
    print(f"  best single member RMSE {member_rmse.min():.1f}, mean {member_rmse.mean():.1f} "
          f"-> ensemble {row['rmse']:.1f} km/s "
          f"({'helps' if row['rmse'] < member_rmse.mean() else 'does not help'})")

    # An ensemble only buys anything from disagreement between its members. Members here are
    # drawn from one pool of ~{available} samples, so at n_train they overlap heavily and their
    # predictions can be nearly identical — in which case averaging cannot help, and reporting
    # "the ensemble did not improve things" as a finding about ensembling would be wrong. State
    # the diversity so the result can be read correctly.
    stacked = np.vstack(preds)
    pairwise = [float(np.corrcoef(stacked[i], stacked[j])[0, 1])
                for i in range(members) for j in range(i + 1, members)]
    expected_overlap = n_train / max(available, 1)
    run.log({"member_pred_corr_mean": float(np.mean(pairwise)),
             "expected_subset_overlap": expected_overlap})
    print(f"  member diversity: mean pairwise prediction correlation "
          f"{np.mean(pairwise):+.3f} (1.000 = identical models)")
    print(f"  each member draws {n_train} of {available} available samples, so subsets overlap "
          f"~{100 * expected_overlap:.0f}% by construction")
    if np.mean(pairwise) > 0.95:
        print("  -> members are near-identical, so this tests almost nothing about ensembling; "
              "a smaller n_train (or a genuinely different model per member) would be needed")
    run.finish()


def experiment_baseline_vs_surya(train, val, omni, lag, make_run) -> None:
    """c3 — identical training and held-out samples, baseline vs Surya. The matched test."""
    print("\n" + "=" * 100)
    print(f"c3  BASELINE vs SURYA (lag {lag} d) — identical training and held-out samples")
    print("=" * 100)
    run = make_run("c3-baseline-vs-surya", {"lag_days": lag})
    y = relabel(train["time"], omni, lag)
    ok = np.isfinite(y)
    yy, tt = y[ok], train["time"][ok]

    print("  TIER 1 (nested blocked CV, the better-powered estimate):")
    truth = clim = None
    for label, feature in (("climatology (train mean)", None),
                           ("linear baseline, 13 channels", "feat13"),
                           ("Surya embedding, 1280-d", PRIMARY_FEATURE)):
        if feature is None:
            truth, pred, clim = out_of_fold(train[PRIMARY_FEATURE][ok], yy, tt)
            pred = clim  # score the reference against itself: skill 0 by construction
        else:
            truth, pred, clim = out_of_fold(train[feature][ok], yy, tt)
        row = scores(truth, pred, clim)
        if feature is None:
            # Suppress r for the reference row. Each fold contributes its own training mean,
            # so the "prediction" is a step function across folds and correlates with time
            # rather than with V — a non-zero value here would read as skill and is not.
            row["r"] = np.nan
        ci = bootstrap_ci(truth, pred, clim, "skill")
        run.log(row | {"split": "tier1", "model": label, "skill_lo": ci[0], "skill_hi": ci[1]})
        print(f"    {label:<30} {fmt(row, ci)}")

    print("\n  TIER 2 (independent holdout, never used for any selection):")
    y_va = relabel(val["time"], omni, lag)
    va_ok = np.isfinite(y_va)
    clim_v = np.full(va_ok.sum(), yy.mean())
    print(f"    {'climatology (train mean)':<30} {fmt(scores(y_va[va_ok], clim_v, clim_v))}")
    for label, feature in (("linear baseline, 13 channels", "feat13"),
                           ("Surya embedding, 1280-d", PRIMARY_FEATURE)):
        pred, _ = fit_predict(train[feature][ok], yy, tt, val[feature][va_ok])
        row = scores(y_va[va_ok], pred, clim_v)
        ci = bootstrap_ci(y_va[va_ok], pred, clim_v, "skill")
        run.log(row | {"split": "tier2", "model": label, "skill_lo": ci[0], "skill_hi": ci[1]})
        print(f"    {label:<30} {fmt(row, ci)}")

    # In-situ-history competitors: operational context, not the primary comparison. An
    # image-only model has no access to past in-situ V, so losing to these is not a failure
    # of the embedding — but not reporting them would overstate what has been achieved.
    print("\n  In-situ-history competitors on Tier 2 (context — these use information an")
    print("  image-only model does not have):")
    for label, back in (("persistence V(t-24h)", 1.0), ("recurrence V(t-27d)", 27.0)):
        ref = relabel(val["time"], omni, lag - back)
        sel = va_ok & np.isfinite(ref)
        if sel.sum() >= 4:
            row = scores(relabel(val["time"], omni, lag)[sel], ref[sel],
                         np.full(sel.sum(), yy.mean()))
            run.log(row | {"split": "tier2", "model": label})
            print(f"    {label:<30} {fmt(row)}")
    run.finish()


# ---------------------------------------------------------------------------
# 3. Exploratory
# ---------------------------------------------------------------------------

def experiment_lag(train, omni, feature, null_threshold, make_run) -> float:
    """c0 (exploratory) — sweep the Sun-Earth propagation lag."""
    print("\n" + "=" * 100)
    print("c0  LAG SWEEP [EXPLORATORY] — coronal source structure, or just V's persistence?")
    print("    If skill decays like the autocorrelation column, the embedding is exploiting")
    print("    persistence. If it holds up at 2-5 d (the physical transit time) it is reading")
    print("    the corona. 27 d is the Carrington control.")
    print(f"    Sweeping {len(LAG_DAYS)} lags and reporting the best carries a multiple-")
    print("    comparisons bias, so this does not replace the primary test above.")
    print("=" * 100)
    autocorr = measured_autocorr(omni, LAG_DAYS)
    run = make_run("c0-lagsweep", {"feature": feature})
    print(f"{'lag (d)':>8}  {'result':<58}  {'r(V,V+lag)':>10}")
    best, best_skill = PRIMARY_LAG, -np.inf
    for lag in LAG_DAYS:
        y = relabel(train["time"], omni, lag)
        ok = np.isfinite(y)
        if ok.sum() < 40:
            print(f"{lag:8.1f}  (only {ok.sum()} labels available — skipped)")
            continue
        truth, pred, clim = out_of_fold(train[feature][ok], y[ok], train["time"][ok])
        row = scores(truth, pred, clim)
        run.log(row | {"lag_days": lag, "v_autocorr": autocorr[lag]})
        note = ""
        if lag == 27.0:
            note = "  <- Carrington control"
        elif 2.0 <= lag <= 5.0:
            note = "  <- physical transit time"
        beats = "*" if row["skill"] > null_threshold else " "
        print(f"{lag:8.1f}{beats} {fmt(row):<58}  {autocorr[lag]:>+10.3f}{note}")
        if row["skill"] > best_skill:
            best, best_skill = lag, row["skill"]
    run.finish()
    print(f"\n  '*' marks lags beating the permutation null ({null_threshold:+.3f}).")
    print(f"  best lag by skill: {best} d (skill {best_skill:+.3f}) — exploratory only.")
    return best


def experiment_depth(train, omni, lag, null_threshold, make_run) -> str:
    """c4 (exploratory) — how much of the backbone is actually needed?"""
    print("\n" + "=" * 100)
    print(f"c4  DEPTH ABLATION [EXPLORATORY] (lag {lag} d) — if a shallower cut ties d10, the")
    print("    remaining blocks are deletable, which is a real efficiency saving.")
    print("=" * 100)
    run = make_run("c4-depth", {"lag_days": lag})
    y = relabel(train["time"], omni, lag)
    ok = np.isfinite(y)
    best, best_skill = PRIMARY_FEATURE, -np.inf
    for key in ("emb_d4", "emb_d6", "emb_d8", "emb_d10"):
        if key not in train:
            continue
        truth, pred, clim = out_of_fold(train[key][ok], y[ok], train["time"][ok])
        row = scores(truth, pred, clim)
        ci = bootstrap_ci(truth, pred, clim, "skill")
        run.log(row | {"depth": int(key.split("_d")[1]), "skill_lo": ci[0], "skill_hi": ci[1]})
        print(f"  {key:<9} {fmt(row, ci)}")
        if row["skill"] > best_skill:
            best, best_skill = key, row["skill"]
    run.finish()
    print(f"\n  best depth: {best} — exploratory only.")
    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--omni-dir", type=str, default=str(DEFAULT_OMNI))
    parser.add_argument("--experiments", nargs="+",
                        default=["c1", "c2", "c3", "c0", "c4"],
                        choices=["c0", "c1", "c2", "c3", "c4"])
    parser.add_argument("--lag", type=float, default=PRIMARY_LAG,
                        help="Lag (days) used for c1-c3. Defaults to the pre-specified 0.")
    parser.add_argument("--feature", type=str, default=PRIMARY_FEATURE)
    parser.add_argument("--n-perm", type=int, default=30,
                        help="Label permutations for the negative control.")
    parser.add_argument("--skip-controls", action="store_true",
                        help="Not recommended: the controls are what make the rest meaningful.")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="solar_wind_regression")
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    train = load_cache(cache_dir / "train.npz")
    val = load_cache(cache_dir / "val.npz")
    omni = load_omni(Path(args.omni_dir))
    stamp = datetime.now().strftime("%m%d-%H%M%S")

    def make_run(tag: str, config: dict) -> Run:
        return Run(f"probe_{tag}_{stamp}", args.wandb_project, args.wandb_entity,
                   not args.no_wandb, config)

    print("=" * 100)
    print("SURYA EMBEDDING PROBE — solar wind speed V at Earth")
    print("=" * 100)
    for name, cache in (("TIER 1 train", train), ("TIER 2 val", val)):
        t = cache["time"]
        gaps = np.diff(np.sort(t)) / np.timedelta64(1, "D")
        print(f"  {name}: n={len(t):4d}  {str(t[0])[:10]} .. {str(t[-1])[:10]}  "
              f"min spacing {gaps.min():.2f} d  V = {cache['v_kms'].mean():.1f} "
              f"+/- {cache['v_kms'].std():.1f} km/s")
    # Cross-split separation: the lag shifts both splits equally, so this is lag-invariant.
    cross = np.abs(train["time"][:, None] - val["time"][None, :]) / np.timedelta64(1, "D")
    print(f"  train/val separation: min {cross.min():.2f} d "
          f"({'OK' if cross.min() >= EMBARGO_DAYS else 'BELOW EMBARGO — see warning'})")
    if cross.min() < EMBARGO_DAYS:
        print(f"  WARNING: some val samples sit within {EMBARGO_DAYS} d of a training sample; "
              "Tier 2 skill may be inflated by autocorrelation.")

    null_threshold = -np.inf
    if not args.skip_controls:
        null_threshold = run_controls(train, omni, make_run, args.n_perm)
        run_primary(train, val, omni, null_threshold, make_run)

    if "c1" in args.experiments:
        experiment_data_amount(train, omni, args.lag, args.feature, make_run)
    if "c2" in args.experiments:
        experiment_ensemble(train, omni, args.lag, args.feature, make_run)
    if "c3" in args.experiments:
        experiment_baseline_vs_surya(train, val, omni, args.lag, make_run)
    if "c0" in args.experiments:
        experiment_lag(train, omni, args.feature, null_threshold, make_run)
    if "c4" in args.experiments:
        experiment_depth(train, omni, args.lag, null_threshold, make_run)


if __name__ == "__main__":
    main()
