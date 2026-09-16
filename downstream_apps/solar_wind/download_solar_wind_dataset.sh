#!/usr/bin/env bash
# Download the full nasa-ibm-ai4science/Surya-bench-solarwind HF dataset (OMNI solar-wind
# labels: train/validation/test/leaky_validation CSVs) into this app's data/ directory.
#
# Thin wrapper: the implementation lives in workshop_infrastructure/assets.py.
# Only train.csv / validation.csv are wired into configs/config_script_01.yaml
# (ds_train_index_path / ds_val_index_path) — test.csv and leaky_validation.csv are fetched
# for completeness but this template has no test/eval path for them.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
exec python -m workshop_infrastructure.assets \
  --dataset-repo nasa-ibm-ai4science/Surya-bench-solarwind \
  --dataset-files train.csv validation.csv test.csv leaky_validation.csv \
  --dest "${SCRIPT_DIR}/data/hf_solar_wind"
