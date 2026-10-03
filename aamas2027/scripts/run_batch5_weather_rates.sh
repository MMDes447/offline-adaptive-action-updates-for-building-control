#!/usr/bin/env bash
# Batch 5: held-out weather (California) + update-rate sweep of the state-independent
# selectors + baseline training-length check. No training; every run is an annual
# EnergyPlus-Radiance simulation with the project's own notebook code (run_eval.py,
# run_rbc.py). Building, glazing (your SageGlass SR2 BSDFs), lighting and HVAC are
# unchanged; only the weather file changes in part A.
#
#   JOBS=2 bash aamas2027/scripts/run_batch5_weather_rates.sh
#
# Part A  held-out weather (default WEATHERS="usa_ca_fresno"; an .epw path also works,
#         e.g. WEATHERS="usa_ca_fresno /path/USA_CA_Los.Angeles.Intl.AP.722950_TMY3.epw")
#   rbc_clean_sf                       RBC under San Francisco = reproduction check of
#                                      offline_smooth_clean_episode_000 (must match, else stop)
#   rbc_clean_<tag>                    RBC under the new weather (runs after the check)
#   rbc_b2225_{sf,<tag>}               retuned RBC: the 22-25 C thermostat band now in the
#                                      notebook cell (episodes 0-12 used 21-24 C; see run_rbc.py)
#   b3e100_<tag>_sample_s<seed>        reported checkpoint, selector seeds $SEEDS
#   sdar_iql_t{1,2}_e100_<tag>_sample_s20260728   the two extra training runs
# Part B  update-rate sweep on the reported checkpoint (San Francisco)
#   b3e100_{periodic,constant}_{x2,x4,rbcrate}_s<seed>, seeds $RATE_SEEDS
#   x2 / x4: 2x and 4x the learned per-subsystem rates (capped at 1);
#   rbcrate: the behaviour controller's own per-subsystem update rates.
#   (1x rate-matched and always-update already exist from batch 2.)
# Part C  baselines at epoch 50 (San Francisco), to rule out under/over-training:
#   {iql_flat,bc_flat,bc_sdar}_t1_e50_sample_s20260728
# Finally make_batch5.py -> paper/tab_weather.tex, numbers_batch5.tex, fig_rate_sweep.pdf.
#
# Outputs go to aamas2027/batch5_runs/<stamp>/. Finished runs are skipped (restartable).
# Nothing existing is overwritten.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAMAS="$(dirname "$SCRIPTS")"; ROOT="$(dirname "$AAMAS")"
CONDA_ENV="${CONDA_ENV:-offrl_5zone}"; JOBS="${JOBS:-2}"
WEATHERS="${WEATHERS:-usa_ca_fresno}"
SEEDS="${SEEDS:-20260728 20260729 20260730}"
RATE_SEEDS="${RATE_SEEDS:-20260728 20260729}"
PART_A="${PART_A:-1}"; PART_B="${PART_B:-1}"; PART_C="${PART_C:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
OUT="$AAMAS/batch5_runs/$STAMP"; RUNS="$OUT/runs"; LOGS="$OUT/logs"; mkdir -p "$RUNS" "$LOGS"

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

B3="checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3"
CKPT="$B3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch100.pt"
ck_seed() { echo "checkpoints/aamas2027/aamas_${1}_seed${2}/sdar_iql_aamas_${1}_seed${2}_checkpoint_epoch${3}.pt"; }
for f in "$CKPT" "$(ck_seed sdar_iql 1 100)" "$(ck_seed sdar_iql 2 100)" \
         "$(ck_seed iql_flat 1 50)" "$(ck_seed bc_flat 1 50)" "$(ck_seed bc_sdar 1 50)"; do
  [ -f "$ROOT/$f" ] || { echo "STOP: missing checkpoint $f"; exit 1; }
done

# weather keys -> tags (and fail early on unknown keys)
python - $WEATHERS <<'PY' > "$LOGS/weather_tags.txt" || { echo "STOP: unknown weather (see above)"; exit 1; }
import sys, os
from pyenergyplus.dataset import weather_files
for w in sys.argv[1:]:
    if w in weather_files: tag = w.replace("usa_ca_", "").replace("usa_", "")
    elif os.path.isfile(w): tag = os.path.basename(w).split(".")[0].split("_")[-1].lower() or "custom"
    else:
        sys.stderr.write(f"unknown weather {w!r}; bundled keys: {sorted(weather_files)}\n"); sys.exit(1)
    print(w, tag)
PY
echo "weathers: $(tr '\n' ';' < "$LOGS/weather_tags.txt")"

done_ok() { for mf in "$AAMAS"/batch5_runs/*/runs/"$1"/manifest.json; do
              [ -f "$mf" ] && grep -q '"status": "ok"' "$mf" && return 0; done; return 1; }
CMDS="$LOGS/cmds.txt"; : > "$CMDS"
ev() {  # ev <name> <checkpoint> <args...>
  local name="$1" ck="$2"; shift 2
  if done_ok "$name"; then echo "skip $name (done)"; return; fi
  echo "python $SCRIPTS/run_eval.py --runs-dir $RUNS --run-name $name --checkpoint $ck --overwrite $* > $LOGS/$name.log 2>&1 && echo OK $name || echo FAIL $name" >> "$CMDS"
}

# ---------------- part A: RBC chain first (reproduction check gates the new-weather RBC) ----------------
if [ "$PART_A" = "1" ]; then
  chain=""
  if ! done_ok rbc_clean_sf; then
    chain="python $SCRIPTS/run_rbc.py --runs-dir $RUNS --run-name rbc_clean_sf --check-reference --overwrite > $LOGS/rbc_clean_sf.log 2>&1 && echo OK rbc_clean_sf"
  fi
  while read -r w tag; do
    done_ok "rbc_clean_$tag" && { echo "skip rbc_clean_$tag (done)"; continue; }
    step="python $SCRIPTS/run_rbc.py --runs-dir $RUNS --run-name rbc_clean_$tag --weather $w --overwrite > $LOGS/rbc_clean_$tag.log 2>&1 && echo OK rbc_clean_$tag"
    chain="${chain:+$chain && }$step"
  done < "$LOGS/weather_tags.txt"
  [ -n "$chain" ] && echo "( $chain ) || echo FAIL rbc_chain" >> "$CMDS"
  # retuned RBC: the 22-25 C thermostat band currently in the notebook cell (a stronger rule-based
  # baseline; no reference episode exists for it, so no reproduction check)
  chain2=""
  for pair in "usa_ca_san_francisco|sf" $(awk '{print $1"|"$2}' "$LOGS/weather_tags.txt"); do
    w="${pair%%|*}"; tag="${pair##*|}"
    done_ok "rbc_b2225_$tag" && { echo "skip rbc_b2225_$tag (done)"; continue; }
    step="python $SCRIPTS/run_rbc.py --runs-dir $RUNS --run-name rbc_b2225_$tag --weather $w --band notebook --overwrite > $LOGS/rbc_b2225_$tag.log 2>&1 && echo OK rbc_b2225_$tag"
    chain2="${chain2:+$chain2 && }$step"
  done
  [ -n "$chain2" ] && echo "( $chain2 ) || echo FAIL rbc_b2225_chain" >> "$CMDS"
  while read -r w tag; do
    for s in $SEEDS; do ev "b3e100_${tag}_sample_s$s" "$CKPT" --mode sample --selector-seed "$s" --weather "$w"; done
    for ts in 1 2; do ev "sdar_iql_t${ts}_e100_${tag}_sample_s20260728" "$(ck_seed sdar_iql $ts 100)" \
                         --mode sample --selector-seed 20260728 --weather "$w"; done
  done < "$LOGS/weather_tags.txt"
fi

# ---------------- part B: update-rate sweep (same checkpoint, San Francisco) ----------------
if [ "$PART_B" = "1" ]; then
  REF_RUN="$AAMAS/runs/b3e100_sample_s20260728"
  RATES1="$(python - "$REF_RUN/rollout.npz" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
key = "learned_selection_probabilities" if "learned_selection_probabilities" in d.files else "selection_probabilities"
p = d[key][1:]
print(",".join(f"{x:.5f}" for x in (p[:, 0:4].mean(), p[:, 4:9].mean(), p[:, 9:14].mean(), p[:, 14:19].mean())))
PY
)"
  RATES_RBC="$(python - "$ROOT/offline_smooth_clean_episode_000_stats.json" <<'PY'
import sys, json
s = json.load(open(sys.argv[1]))
print(",".join(f"{s[k]:.5f}" for k in ("mean_selection_mask_glazing", "mean_selection_mask_lighting",
                                       "mean_selection_mask_heating", "mean_selection_mask_cooling")))
PY
)"
  scale() { python -c "import sys; print(','.join(f'{min(1.0, float(x)*$2):.5f}' for x in '$1'.split(',')))"; }
  RATES2="$(scale "$RATES1" 2)"; RATES4="$(scale "$RATES1" 4)"
  { echo "x1 (learned, = batch 2) $RATES1"; echo "x2 $RATES2"; echo "x4 $RATES4"; echo "rbcrate $RATES_RBC"; } | tee "$LOGS/rates.txt"
  for s in $RATE_SEEDS; do
    for pair in "x2:$RATES2" "x4:$RATES4" "rbcrate:$RATES_RBC"; do
      lab="${pair%%:*}"; r="${pair#*:}"
      ev "b3e100_periodic_${lab}_s$s" "$CKPT" --mode periodic --selector-seed "$s" --phase-seed "$s" --rates "$r"
      ev "b3e100_constant_${lab}_s$s" "$CKPT" --mode constant --selector-seed "$s" --rates "$r"
    done
  done
fi

# ---------------- part C: baselines at epoch 50 ----------------
if [ "$PART_C" = "1" ]; then
  for algo in iql_flat bc_flat bc_sdar; do
    ev "${algo}_t1_e50_sample_s20260728" "$(ck_seed $algo 1 50)" --mode sample --selector-seed 20260728
  done
fi

echo "$(wc -l < "$CMDS") jobs queued, $JOBS in parallel (the RBC chain is one job). Logs: $LOGS/<run>.log"
if [ -s "$CMDS" ]; then
  xargs -P "$JOBS" -I{} bash -c "{}" < "$CMDS" | tee "$LOGS/status.txt"
  if grep -q "^FAIL" "$LOGS/status.txt"; then echo "STOP: some runs failed:"; grep "^FAIL" "$LOGS/status.txt"; exit 1; fi
fi
python "$SCRIPTS/make_batch5.py" && echo "batch-5 tables/figure written to $AAMAS/paper"
date --iso-8601=seconds > "$OUT/COMPLETE"
echo "Batch 5 finished."
