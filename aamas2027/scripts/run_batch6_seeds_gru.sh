#!/usr/bin/env bash
# Batch 6: two more training runs of our agent (5 in total) and the observation-history
# (GRU window) ablation. Same training code (train_baselines.py -> frozen beta3 cell) and
# same evaluation code (run_eval.py -> frozen notebook cells) as batches 4 and 5.
#
#   JOBS=2 TRAIN_JOBS=2 bash aamas2027/scripts/run_batch6_seeds_gru.sh
#
# Stage 1 (training, GPU, 100 epochs each, like the reported checkpoint):
#   sdar_iql     seeds $SDAR_SEEDS (default 3 4)  our method, new training seeds
#                (seed 0 = reported beta3 checkpoint, seeds 1-2 = batch 4)
#   sdar_iql_k1  seed 1                           our method with SEQ_LEN 36 -> 1:
#                the GRU sees only the current observation (no history)
# Stage 2 (annual evaluations):
#   sdar_iql_t<s>_e100_sample_s<es>           San Francisco, selector seeds $EVAL_SEEDS
#   sdar_iql_t<s>_e100_fresno_sample_s20260728   held-out climate (as batch 5 did for t1, t2)
#   k1_e100_sample_s<es>                       San Francisco, selector seeds $EVAL_SEEDS
# Stage 3: make_baselines.py, make_components.py (no --seeds), make_batch5.py
#   -> aamas2027/paper/ (tables and macros now count 5 training runs, the K=1 row and the
#      Fresno results of all training runs). They are then integrated into the paper.
#
# New folders only: checkpoints/aamas2027/<run>/ and aamas2027/batch6_runs/<stamp>/.
# Finished trainings/evaluations are skipped (restartable). Nothing existing is overwritten.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAMAS="$(dirname "$SCRIPTS")"; ROOT="$(dirname "$AAMAS")"
CONDA_ENV="${CONDA_ENV:-offrl_5zone}"
JOBS="${JOBS:-2}"; TRAIN_JOBS="${TRAIN_JOBS:-2}"
SDAR_SEEDS="${SDAR_SEEDS:-3 4}"
EVAL_SEEDS="${EVAL_SEEDS:-20260728 20260729 20260730}"
FRESNO_SEED="${FRESNO_SEED:-20260728}"
EPOCHS=100
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
OUT="$AAMAS/batch6_runs/$STAMP"; RUNS="$OUT/runs"; LOGS="$OUT/logs"
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
python - <<'PY' || { echo "STOP: usa_ca_fresno weather not available"; exit 1; }
from pyenergyplus.dataset import weather_files
assert "usa_ca_fresno" in weather_files
PY

# ---------------- stage 1: training ----------------
TCMDS="$LOGS/train_cmds.txt"; : > "$TCMDS"
addt() {  # addt <algo> <seed>
  local algo="$1" seed="$2" run="aamas_${1}_seed${2}"
  local man="checkpoints/aamas2027/$run/train_manifest.json"
  if [ -f "$man" ] && grep -q '"status": "ok"' "$man"; then echo "skip training $run (done)"; return; fi
  if [ -f "$man" ]; then echo "STOP: $man exists but is not status ok; inspect it (nothing is deleted automatically)"; exit 1; fi
  echo "python $SCRIPTS/train_baselines.py --algo $algo --seed $seed --epochs $EPOCHS > $LOGS/train_$run.log 2>&1 && echo OK train_$run || echo FAIL train_$run" >> "$TCMDS"
}
for s in $SDAR_SEEDS; do addt sdar_iql "$s"; done
addt sdar_iql_k1 1
echo "Stage 1: $(wc -l < "$TCMDS") trainings, $TRAIN_JOBS in parallel"
if [ -s "$TCMDS" ]; then
  xargs -P "$TRAIN_JOBS" -I{} bash -c "{}" < "$TCMDS" | tee "$LOGS/train_status.txt"
  if grep -q "^FAIL" "$LOGS/train_status.txt"; then echo "STOP: training failed:"; grep "^FAIL" "$LOGS/train_status.txt"; exit 1; fi
fi

# ---------------- stage 2: evaluation ----------------
ECMDS="$LOGS/eval_cmds.txt"; : > "$ECMDS"
done_ok() { for mf in "$AAMAS"/batch*_runs/*/runs/"$1"/manifest.json "$AAMAS"/runs/"$1"/manifest.json; do
              [ -f "$mf" ] && grep -q '"status": "ok"' "$mf" && return 0; done; return 1; }
ck() { echo "checkpoints/aamas2027/aamas_${1}_seed${2}/sdar_iql_aamas_${1}_seed${2}_checkpoint_epoch${EPOCHS}.pt"; }
ev() {  # ev <name> <checkpoint> <args...>
  local name="$1" c="$2"; shift 2
  [ -f "$ROOT/$c" ] || { echo "STOP: missing checkpoint $c"; exit 1; }
  if done_ok "$name"; then echo "skip $name (done)"; return; fi
  echo "python $SCRIPTS/run_eval.py --runs-dir $RUNS --run-name $name --checkpoint $c --overwrite $* > $LOGS/$name.log 2>&1 && echo OK $name || echo FAIL $name" >> "$ECMDS"
}
for s in $SDAR_SEEDS; do
  for es in $EVAL_SEEDS; do ev "sdar_iql_t${s}_e${EPOCHS}_sample_s${es}" "$(ck sdar_iql "$s")" --mode sample --selector-seed "$es"; done
  ev "sdar_iql_t${s}_e${EPOCHS}_fresno_sample_s${FRESNO_SEED}" "$(ck sdar_iql "$s")" --mode sample --selector-seed "$FRESNO_SEED" --weather usa_ca_fresno
done
for es in $EVAL_SEEDS; do ev "k1_e${EPOCHS}_sample_s${es}" "$(ck sdar_iql_k1 1)" --mode sample --selector-seed "$es"; done
echo "Stage 2: $(wc -l < "$ECMDS") annual evaluations, $JOBS in parallel"
if [ -s "$ECMDS" ]; then
  xargs -P "$JOBS" -I{} bash -c "{}" < "$ECMDS" | tee "$LOGS/eval_status.txt"
  if grep -q "^FAIL" "$LOGS/eval_status.txt"; then echo "STOP: evaluation failed:"; grep "^FAIL" "$LOGS/eval_status.txt"; exit 1; fi
fi

# ---------------- stage 3: tables and macros ----------------
python "$SCRIPTS/make_baselines.py"  | tee "$LOGS/make_baselines.txt"
python "$SCRIPTS/make_components.py" | tee "$LOGS/make_components.txt"
python "$SCRIPTS/make_batch5.py"     | tee "$LOGS/make_batch5.txt"
date --iso-8601=seconds > "$OUT/COMPLETE"
echo "Batch 6 finished. Outputs: $OUT ; tables/macros in $AAMAS/paper."
