#!/usr/bin/env bash
# Batch 3: design-component ablation of SDAR-IQL, plus a training-length sweep.
#
#   JOBS=2 bash aamas2027/scripts/run_batch3_components.sh
#
# No retraining. Every variant below was already trained on the SAME 14-episode
# dataset, reward and architecture (GRU 2x128, K=36, twin Q with mask, tau=0.7,
# gamma=0.99, w_max=20). The checkpoint configs differ in exactly these fields
# (checked field by field on 2026-10-01):
#
#   variant  checkpoint dir                                   V input        policy weight            critic
#   plain    ..._gru2_seq36                                   rho            exp(A_std / T), T=1      single twin-Q
#   augv     ..._gru2_seq36_augmented_value                   rho, a_prev    exp(A_std / T), T=1      single twin-Q
#   b3       ..._augmented_value_standard_iql_beta3  (FULL)   rho, a_prev    exp(3 A), no std.        single twin-Q
#   v11      sdar_iql_v11_dualcritic_..._decomnposed          rho, a_prev    exp(A_std / T), T=1      task + SDAR critics
#
# plain -> augv -> b3 is a clean one-change-at-a-time ladder at matched epoch 100.
# v11 is the decomposed-critic framework (epoch 100 = matched, epoch 250 = its best).
#
# Part A  component ladder: 5 variants x SEEDS selector seeds, all at the same
#         evaluation protocol (learned sampled selector, deterministic proposal,
#         safeguard on, SF TMY3).
# Part B  training length: beta3 epochs 50..500, selector seed 20260728.
#
# Run names are shared with batches 1 and 2 (b3e100_sample_s*, b3e500_sample_s20260728),
# so runs that already finished are skipped. Safe to restart.
# Finishes by calling make_components.py.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAMAS="$(dirname "$SCRIPTS")"; ROOT="$(dirname "$AAMAS")"
LOGS="$AAMAS/logs"; mkdir -p "$LOGS"
JOBS="${JOBS:-2}"
CONDA_ENV="${CONDA_ENV:-offrl_5zone}"
SEEDS="${SEEDS:-20260728 20260729 20260730}"
PART_A="${PART_A:-1}"
PART_B="${PART_B:-1}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-50 100 150 200 250 300 350 400 450 500}"

# shellcheck disable=SC1091
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

C="checkpoints"
PLAIN="$C/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_checkpoint_epoch"
AUGV="$C/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_checkpoint_epoch"
B3="$C/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch"
V11="$C/sdar_iql_v11_dualcritic_preserve_old_total_thermal3_tclip50_gru2_seq36_decomnposed/sdar_iql_sdar_iql_v11_dualcritic_preserve_old_total_thermal3_tclip50_gru2_seq36_decomnposed_checkpoint_epoch"

CMDS="$LOGS/batch3_cmds.txt"; : > "$CMDS"
add() {  # add <run-name> <checkpoint> <extra args...>
  local name="$1" ckpt="$2"; shift 2
  [ -f "$ROOT/$ckpt" ] || { echo "MISSING checkpoint for $name: $ckpt"; exit 1; }
  for mf in "$AAMAS/runs/$name/manifest.json" "$AAMAS"/batch*_runs/*/runs/"$name"/manifest.json; do
    if [ -f "$mf" ] && grep -q '"status": "ok"' "$mf"; then
      echo "skip $name (done: $mf)"; return
    fi
  done
  echo "python -u $SCRIPTS/run_eval.py --run-name $name --checkpoint $ckpt $* > $LOGS/$name.log 2>&1 && echo OK $name || { echo FAIL $name; exit 255; }" >> "$CMDS"
}

if [ "$PART_A" = "1" ]; then
  for s in $SEEDS; do
    add "b3e100_sample_s$s"    "${B3}100.pt"    --mode sample --selector-seed "$s"   # full model
    add "augv_e100_sample_s$s" "${AUGV}100.pt"  --mode sample --selector-seed "$s"   # - standard IQL weighting
    add "plain_e100_sample_s$s" "${PLAIN}100.pt" --mode sample --selector-seed "$s"  # - augmented value
    add "v11_e100_sample_s$s"  "${V11}100.pt"   --mode sample --selector-seed "$s"   # decomposed critic, matched epoch
    add "v11_e250_sample_s$s"  "${V11}250.pt"   --mode sample --selector-seed "$s"   # decomposed critic, best epoch
  done
fi
if [ "$PART_B" = "1" ]; then
  for e in $SWEEP_EPOCHS; do
    add "b3e${e}_sample_s20260728" "${B3}${e}.pt" --mode sample --selector-seed 20260728
  done
fi

N="$(wc -l < "$CMDS")"
echo "$N runs queued, $JOBS in parallel. Commands: $CMDS  Progress: $LOGS/<run>.log"
if [ "$N" -gt 0 ]; then
  xargs -P "$JOBS" -I{} bash -c "{}" < "$CMDS" | tee "$LOGS/batch3_status.txt"
fi
python "$SCRIPTS/make_components.py" --seeds $SEEDS && echo "component tables/figure written to $AAMAS/paper"
echo "Batch 3 finished. FAIL lines above (if any) name the log to check."
