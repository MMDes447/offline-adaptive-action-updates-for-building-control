#!/usr/bin/env bash
# Batch 4: offline baselines + extra training seeds of SDAR-IQL, trained with the
# exact beta3 training code (train_baselines.py) and evaluated with the exact
# evaluation code (run_eval.py, new_dataset_gen1 cells).
#
#   JOBS=2 TRAIN_JOBS=2 bash aamas2027/scripts/run_batch4_baselines.sh
#
# Stage 1 (training, GPU, 100 epochs each, like the reported beta3 checkpoint):
#   iql_flat  seed(s) $BASELINE_SEEDS   IQL without action repetition (full action every step)
#   bc_flat   seed(s) $BASELINE_SEEDS   behaviour cloning, full action every step
#   bc_sdar   seed(s) $BASELINE_SEEDS   behaviour cloning with the act/repeat structure (no critic)
#   sdar_iql  seed(s) $SDAR_SEEDS       our method, new training seeds (beta3 itself = seed 0)
# Stage 2 (annual evaluation of each epoch-100 checkpoint):
#   flat baselines: 1 run each (their selector is constant, so selector seeds do not matter)
#   bc_sdar, sdar_iql: selector seeds $EVAL_SEEDS
# Stage 3: make_baselines.py -> paper/tab_baselines.tex, numbers_baselines.tex
#
# Everything goes to new folders: checkpoints/aamas2027/<run>/ and
# aamas2027/batch4_runs/<stamp>/. Finished trainings/evaluations are skipped, so the
# script can be restarted. Nothing existing is overwritten.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAMAS="$(dirname "$SCRIPTS")"; ROOT="$(dirname "$AAMAS")"
CONDA_ENV="${CONDA_ENV:-offrl_5zone}"
JOBS="${JOBS:-2}"; TRAIN_JOBS="${TRAIN_JOBS:-2}"
BASELINE_SEEDS="${BASELINE_SEEDS:-1}"
SDAR_SEEDS="${SDAR_SEEDS:-1 2}"
EVAL_SEEDS="${EVAL_SEEDS:-20260728 20260729 20260730}"
EPOCHS="${EPOCHS:-100}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
OUT="$AAMAS/batch4_runs/$STAMP"; RUNS="$OUT/runs"; LOGS="$OUT/logs"
mkdir -p "$RUNS" "$LOGS"

CONDA_SH="${CONDA_SH:-}"
if [ -z "$CONDA_SH" ]; then
  if command -v conda >/dev/null 2>&1; then CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  else CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"; fi
fi
set +u; source "$CONDA_SH"; conda activate "$CONDA_ENV"; set -u
[ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ] || { echo "STOP: could not activate $CONDA_ENV"; exit 1; }
export PYTHONUNBUFFERED=1
cd "$ROOT"
python "$SCRIPTS/faithful/extract_cells.py" --check || { echo "STOP: frozen notebook cells differ from the notebooks"; exit 1; }

# ---------------- stage 1: training ----------------
TCMDS="$LOGS/train_cmds.txt"; : > "$TCMDS"
addt() {  # addt <algo> <seed>
  local algo="$1" seed="$2" run="aamas_${1}_seed${2}"
  local man="checkpoints/aamas2027/$run/train_manifest.json"
  if [ -f "$man" ] && grep -q '"status": "ok"' "$man"; then echo "skip training $run (done)"; return; fi
  echo "python $SCRIPTS/train_baselines.py --algo $algo --seed $seed --epochs $EPOCHS > $LOGS/train_$run.log 2>&1 && echo OK train_$run || echo FAIL train_$run" >> "$TCMDS"
}
for s in $BASELINE_SEEDS; do addt iql_flat "$s"; addt bc_flat "$s"; addt bc_sdar "$s"; done
for s in $SDAR_SEEDS; do addt sdar_iql "$s"; done
echo "Stage 1: $(wc -l < "$TCMDS") trainings, $TRAIN_JOBS in parallel"
if [ -s "$TCMDS" ]; then
  xargs -P "$TRAIN_JOBS" -I{} bash -c "{}" < "$TCMDS" | tee "$LOGS/train_status.txt"
  if grep -q "^FAIL" "$LOGS/train_status.txt"; then echo "STOP: training failed:"; grep "^FAIL" "$LOGS/train_status.txt"; exit 1; fi
fi

# ---------------- stage 2: evaluation ----------------
ECMDS="$LOGS/eval_cmds.txt"; : > "$ECMDS"
adde() {  # adde <algo> <train-seed> <eval-seed>
  local algo="$1" ts="$2" es="$3"
  local ck="checkpoints/aamas2027/aamas_${algo}_seed${ts}/sdar_iql_aamas_${algo}_seed${ts}_checkpoint_epoch${EPOCHS}.pt"
  local name="${algo}_t${ts}_e${EPOCHS}_sample_s${es}"
  [ -f "$ROOT/$ck" ] || { echo "MISSING checkpoint $ck"; return; }
  for mf in "$AAMAS"/batch4_runs/*/runs/"$name"/manifest.json; do
    if [ -f "$mf" ] && grep -q '"status": "ok"' "$mf"; then echo "skip $name (done: $mf)"; return; fi
  done
  echo "python $SCRIPTS/run_eval.py --runs-dir $RUNS --run-name $name --checkpoint $ck --mode sample --selector-seed $es --overwrite > $LOGS/$name.log 2>&1 && echo OK $name || echo FAIL $name" >> "$ECMDS"
}
FIRST_SEED="$(echo $EVAL_SEEDS | awk '{print $1}')"
for s in $BASELINE_SEEDS; do
  adde iql_flat "$s" "$FIRST_SEED"
  adde bc_flat "$s" "$FIRST_SEED"
  for es in $EVAL_SEEDS; do adde bc_sdar "$s" "$es"; done
done
for s in $SDAR_SEEDS; do for es in $EVAL_SEEDS; do adde sdar_iql "$s" "$es"; done; done
echo "Stage 2: $(wc -l < "$ECMDS") annual evaluations, $JOBS in parallel"
if [ -s "$ECMDS" ]; then
  xargs -P "$JOBS" -I{} bash -c "{}" < "$ECMDS" | tee "$LOGS/eval_status.txt"
  if grep -q "^FAIL" "$LOGS/eval_status.txt"; then echo "STOP: evaluation failed:"; grep "^FAIL" "$LOGS/eval_status.txt"; exit 1; fi
fi

# ---------------- stage 3: tables ----------------
python "$SCRIPTS/make_baselines.py" && echo "baseline table written to $AAMAS/paper"
date --iso-8601=seconds > "$OUT/COMPLETE"
echo "Batch 4 finished."
