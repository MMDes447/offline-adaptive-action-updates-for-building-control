#!/usr/bin/env bash
# Batch 1: settle which checkpoint the paper reports.
#   0. 2-day smoke test (pipeline check only, not a result)
#   1. beta3 epoch 100, learned sample selector  -> should reproduce the headline numbers
#   2. beta3 epoch 500, learned sample selector  -> the true epoch-500 policy (never evaluated so far)
#
# Usage (from anywhere):
#   bash aamas2027/scripts/run_batch1_provenance.sh   (from the repository root)
# Optional: CONDA_ENV=offrl_5zone_mo  CUDA_VISIBLE_DEVICES=1  bash ...run_batch1_provenance.sh
set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAMAS="$(dirname "$SCRIPTS")"
LOGS="$AAMAS/logs"; mkdir -p "$LOGS"
CONDA_ENV="${CONDA_ENV:-offrl_5zone}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

B3="checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3"
E100="$B3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch100.pt"
E500="$B3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch500.pt"
SHA_E100="c723f82a473474a11d483133afa5b61cdbe822c4a1550e121686afd90f05c750"
SHA_E500="56d0b61cfb161529638c4f74b231fe92ebbd1c6caa3d44cc582e9421f2695425"
STAMP="$(date +%Y%m%d_%H%M)"

run() {  # run <run-name> <args...>
  local name="$1"; shift
  echo "=== $(date '+%F %T')  start $name"
  python "$SCRIPTS/run_eval.py" --run-name "$name" "$@" > "$LOGS/$name.log" 2>&1 \
    && echo "=== $(date '+%F %T')  OK    $name" \
    || { echo "=== $(date '+%F %T')  FAIL  $name  (see $LOGS/$name.log)"; return 1; }
}

python "$SCRIPTS/../../check_env.py" > "$LOGS/env_check_$STAMP.txt" 2>&1 || true

run "smoke_${STAMP}_e100_sample_2days" --checkpoint "$E100" --expect-checkpoint-sha "$SHA_E100" \
    --mode sample --selector-seed 20260728 --smoke-days 2

run "b3e100_sample_s20260728" --checkpoint "$E100" --expect-checkpoint-sha "$SHA_E100" \
    --mode sample --selector-seed 20260728
run "b3e500_sample_s20260728" --checkpoint "$E500" --expect-checkpoint-sha "$SHA_E500" \
    --mode sample --selector-seed 20260728

echo "Batch 1 finished. Results in $AAMAS/runs/ ."
