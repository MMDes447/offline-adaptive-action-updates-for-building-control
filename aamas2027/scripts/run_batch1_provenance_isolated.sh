#!/usr/bin/env bash
# Batch 1 launcher with all batch-level checks/logs isolated under BATCH_REPORT_DIR.
set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAMAS="$(dirname "$SCRIPTS")"
PROJECT_ROOT="$(dirname "$AAMAS")"
CONDA_ENV="${CONDA_ENV:-offrl_5zone}"
BATCH_STAMP="${BATCH_STAMP:?set BATCH_STAMP to YYYYMMDD_HHMM}"
BATCH_REPORT_DIR="${BATCH_REPORT_DIR:?set BATCH_REPORT_DIR inside $AAMAS/runs}"
BATCH_REPORT_DIR="$(realpath -m "$BATCH_REPORT_DIR")"

case "$BATCH_REPORT_DIR" in
  "$AAMAS"/runs/batch1_*) ;;
  *) echo "ERROR: BATCH_REPORT_DIR must be $AAMAS/runs/batch1_<timestamp>" >&2; exit 1 ;;
esac
if [[ ! -d "$BATCH_REPORT_DIR" ]]; then
  echo "ERROR: batch report directory does not exist: $BATCH_REPORT_DIR" >&2
  exit 1
fi

B3="checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3"
E100="$B3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch100.pt"
E500="$B3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch500.pt"
SHA_E100="c723f82a473474a11d483133afa5b61cdbe822c4a1550e121686afd90f05c750"
SHA_E500="56d0b61cfb161529638c4f74b231fe92ebbd1c6caa3d44cc582e9421f2695425"
SMOKE_NAME="smoke_${BATCH_STAMP}_e100_sample_2days"
E100_NAME="b3e100_sample_s20260728"
E500_NAME="b3e500_sample_s20260728"
LOGS="$BATCH_REPORT_DIR/per_run_logs"

# Enforce the handoff's stronger no-clobber rule before creating any run output.
for name in "$SMOKE_NAME" "$E100_NAME" "$E500_NAME"; do
  if [[ -e "$AAMAS/runs/$name" ]]; then
    echo "ERROR: run path already exists: $AAMAS/runs/$name" >&2
    exit 1
  fi
done
for path in "$BATCH_REPORT_DIR/env_check.json" "$BATCH_REPORT_DIR/env_check.txt" \
            "$LOGS/$SMOKE_NAME.log" "$LOGS/$E100_NAME.log" "$LOGS/$E500_NAME.log"; do
  if [[ -e "$path" ]]; then
    echo "ERROR: refusing to overwrite existing batch file: $path" >&2
    exit 1
  fi
done

mkdir -p "$LOGS"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: failed to activate conda environment $CONDA_ENV" >&2
  exit 1
fi
cd "$PROJECT_ROOT"

python "$SCRIPTS/check_env_isolated.py" \
  --expected-conda-env "$CONDA_ENV" \
  --output "$BATCH_REPORT_DIR/env_check.json" \
  > "$BATCH_REPORT_DIR/env_check.txt" 2>&1

run() {
  local name="$1"
  shift
  echo "=== $(date '+%F %T')  start $name"
  python "$SCRIPTS/run_eval.py" --run-name "$name" "$@" > "$LOGS/$name.log" 2>&1 \
    && echo "=== $(date '+%F %T')  OK    $name" \
    || { echo "=== $(date '+%F %T')  FAIL  $name  (see $LOGS/$name.log)"; return 1; }
}

run "$SMOKE_NAME" --checkpoint "$E100" --expect-checkpoint-sha "$SHA_E100" \
  --mode sample --selector-seed 20260728 --smoke-days 2
run "$E100_NAME" --checkpoint "$E100" --expect-checkpoint-sha "$SHA_E100" \
  --mode sample --selector-seed 20260728
run "$E500_NAME" --checkpoint "$E500" --expect-checkpoint-sha "$SHA_E500" \
  --mode sample --selector-seed 20260728

echo "Batch 1 finished. Results in $AAMAS/runs/"
