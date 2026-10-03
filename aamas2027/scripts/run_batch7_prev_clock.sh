#!/usr/bin/env bash
# Batch 7: (A) full-action IQL whose actor also receives the previously executed action,
#          (B) a time-of-day (clock) update selector with the learned selector's budget and
#              average hour-of-day profile, on the reported checkpoint.
#
#   JOBS=2 TRAIN_JOBS=2 bash aamas2027/scripts/run_batch7_prev_clock.sh
#
# Stage 0: hour-of-day table from the reference rollout (make_hourly_rates.py)
#          -> aamas2027/batch7_inputs/hourly_rates_b3e100.json (kept if it already exists)
# Stage 1: train iql_flat_prev seeds $PREV_SEEDS (100 epochs; train_baselines.py, frozen beta3 cell
#          + one documented substitution: the proposal loss receives a_{t-1})
# Stage 2: annual evaluations (run_eval.py)
#          iql_flat_prev_t<s>_e100_always_prev   --mode always_prev (update all, proposal sees a_{t-1})
#          b3e100_clock_s<seed>                  --mode clock, reported checkpoint, seeds $CLOCK_SEEDS
# Stage 3: make_results.py --main-epoch 100, make_baselines.py, make_batch5.py -> aamas2027/paper
#
# New folders only: checkpoints/aamas2027/aamas_iql_flat_prev_seed<s>/, aamas2027/batch7_runs/<stamp>/.
# Finished trainings/evaluations are skipped (restartable). Nothing existing is overwritten.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAMAS="$(dirname "$SCRIPTS")"; ROOT="$(dirname "$AAMAS")"
CONDA_ENV="${CONDA_ENV:-offrl_5zone}"
JOBS="${JOBS:-2}"; TRAIN_JOBS="${TRAIN_JOBS:-2}"
PREV_SEEDS="${PREV_SEEDS:-1 2 3}"
CLOCK_SEEDS="${CLOCK_SEEDS:-20260728 20260729 20260730 20260731 20260732}"
EPOCHS=100
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
OUT="$AAMAS/batch7_runs/$STAMP"; RUNS="$OUT/runs"; LOGS="$OUT/logs"
mkdir -p "$RUNS" "$LOGS"

CONDA_SH="${CONDA_SH:-}"
if [ -z "$CONDA_SH" ]; then
  if command -v conda >/dev/null 2>&1; then CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  else CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"; fi
fi
set +u; source "$CONDA_SH"; conda activate "$CONDA_ENV"; set -u
[ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ] || { echo "STOP: could not activate $CONDA_ENV"; exit 1; }
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_$USER}"
cd "$ROOT"
python "$SCRIPTS/faithful/extract_cells.py" --check || { echo "STOP: frozen notebook cells differ from the notebooks"; exit 1; }
# the patched evaluation function must build (each substitution matches exactly once)
python -c "import sys; sys.path[:0]=['$ROOT','$SCRIPTS/sdar_eval']; import policy_ext; print('policy_ext OK', policy_ext.PATCHED_LOAD_AND_ACT_SHA256[:12])" \
  || { echo "STOP: policy_ext could not patch sdar_iql_train_4_updated.load_and_act"; exit 1; }

CKPT="checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch100.pt"
[ -f "$CKPT" ] || { echo "STOP: missing checkpoint $CKPT"; exit 1; }
REF_ROLLOUT="aamas2027/runs/b3e100_sample_s20260728/rollout.npz"
HOURLY="aamas2027/batch7_inputs/hourly_rates_b3e100.json"

# ---------------- stage 0: hour-of-day table ----------------
if [ -f "$HOURLY" ]; then echo "hourly table exists: $HOURLY"
else python "$SCRIPTS/make_hourly_rates.py" --rollout "$REF_ROLLOUT" --out "$HOURLY" | tee "$LOGS/hourly_rates.txt"; fi

# ---------------- stage 1: training ----------------
TCMDS="$LOGS/train_cmds.txt"; : > "$TCMDS"
for s in $PREV_SEEDS; do
  run="aamas_iql_flat_prev_seed$s"; man="checkpoints/aamas2027/$run/train_manifest.json"
  if [ -f "$man" ] && grep -q '"status": "ok"' "$man"; then echo "skip training $run (done)"; continue; fi
  if [ -f "$man" ]; then echo "STOP: $man exists but is not status ok; inspect it (nothing is deleted automatically)"; exit 1; fi
  echo "python $SCRIPTS/train_baselines.py --algo iql_flat_prev --seed $s --epochs $EPOCHS > $LOGS/train_$run.log 2>&1 && echo OK train_$run || echo FAIL train_$run" >> "$TCMDS"
done
echo "Stage 1: $(wc -l < "$TCMDS") trainings, $TRAIN_JOBS in parallel"
if [ -s "$TCMDS" ]; then
  xargs -P "$TRAIN_JOBS" -I{} bash -c "{}" < "$TCMDS" | tee "$LOGS/train_status.txt"
  if grep -q "^FAIL" "$LOGS/train_status.txt"; then echo "STOP: training failed:"; grep "^FAIL" "$LOGS/train_status.txt"; exit 1; fi
fi

# ---------------- stage 2: evaluation ----------------
ECMDS="$LOGS/eval_cmds.txt"; : > "$ECMDS"
done_ok() { for mf in "$AAMAS"/batch*_runs/*/runs/"$1"/manifest.json "$AAMAS"/runs/"$1"/manifest.json; do
              [ -f "$mf" ] && grep -q '"status": "ok"' "$mf" && return 0; done; return 1; }
ev() {  # ev <name> <checkpoint> <args...>
  local name="$1" c="$2"; shift 2
  [ -f "$ROOT/$c" ] || { echo "STOP: missing checkpoint $c"; exit 1; }
  if done_ok "$name"; then echo "skip $name (done)"; return; fi
  echo "python $SCRIPTS/run_eval.py --runs-dir $RUNS --run-name $name --checkpoint $c --overwrite $* > $LOGS/$name.log 2>&1 && echo OK $name || echo FAIL $name" >> "$ECMDS"
}
for s in $PREV_SEEDS; do
  ev "iql_flat_prev_t${s}_e${EPOCHS}_always_prev" \
     "checkpoints/aamas2027/aamas_iql_flat_prev_seed${s}/sdar_iql_aamas_iql_flat_prev_seed${s}_checkpoint_epoch${EPOCHS}.pt" \
     --mode always_prev --selector-seed 20260728
done
for es in $CLOCK_SEEDS; do
  ev "b3e100_clock_s${es}" "$CKPT" --mode clock --selector-seed "$es" --hourly-rates "$ROOT/$HOURLY"
done
echo "Stage 2: $(wc -l < "$ECMDS") annual evaluations, $JOBS in parallel"
if [ -s "$ECMDS" ]; then
  xargs -P "$JOBS" -I{} bash -c "{}" < "$ECMDS" | tee "$LOGS/eval_status.txt"
  if grep -q "^FAIL" "$LOGS/eval_status.txt"; then echo "STOP: evaluation failed:"; grep "^FAIL" "$LOGS/eval_status.txt"; exit 1; fi
fi

# ---------------- stage 3: tables and macros ----------------
python "$SCRIPTS/make_results.py" --main-epoch 100 | tee "$LOGS/make_results.txt"
python "$SCRIPTS/make_baselines.py" | tee "$LOGS/make_baselines.txt"
python "$SCRIPTS/make_batch5.py"    | tee "$LOGS/make_batch5.txt"
date --iso-8601=seconds > "$OUT/COMPLETE"
echo "Batch 7 finished. Outputs: $OUT ; tables/macros in $AAMAS/paper."
