#!/usr/bin/env bash
# Batch 2: fair update-selector ablation on ONE fixed checkpoint (Table 2 of the paper).
#
#   EPOCH=100 JOBS=2 bash aamas2027/scripts/run_batch2_ablation.sh
#
# Prerequisite: batch 1 finished and produced runs/b3e${EPOCH}_sample_s20260728 (status ok).
# The rate-matched selectors use that run's mean learned update probability per subsystem,
# so random/periodic get the same update budget as the learned selector.
#
# The 17 runs, all with the same checkpoint, deterministic proposal, safeguard and weather:
#   learned sample x 4 more seeds  (seed 20260728 comes from batch 1)
#   learned threshold (p >= 0.5)
#   rate-matched constant (random) x 5 seeds
#   rate-matched periodic x 5 phase seeds
#   always update
#   learned sample with the unoccupied-lights safeguard OFF (diagnostic)
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAMAS="$(dirname "$SCRIPTS")"; ROOT="$(dirname "$AAMAS")"
LOGS="$AAMAS/logs"; mkdir -p "$LOGS"
EPOCH="${EPOCH:-100}"; JOBS="${JOBS:-2}"; CONDA_ENV="${CONDA_ENV:-offrl_5zone}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"

B3="checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3"
CKPT="$B3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch${EPOCH}.pt"
REF_RUN="$AAMAS/runs/b3e${EPOCH}_sample_s20260728"
[ -f "$ROOT/$CKPT" ] || { echo "missing checkpoint $CKPT"; exit 1; }
python - "$REF_RUN" <<'PY' || { echo "batch-1 reference run missing or not ok: $REF_RUN"; exit 1; }
import json, sys; m = json.load(open(sys.argv[1] + "/manifest.json")); assert m["status"] == "ok", m["status"]
PY

# rate matching: mean learned update probability per subsystem (forced first step excluded)
RATES="$(python - "$REF_RUN/rollout.npz" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
key = "learned_selection_probabilities" if "learned_selection_probabilities" in d.files else "selection_probabilities"
p = d[key][1:]
g = [p[:, 0:4].mean(), p[:, 4:9].mean(), p[:, 9:14].mean(), p[:, 14:19].mean()]
print(",".join(f"{x:.5f}" for x in g))
PY
)"
echo "rate-matched targets (glazing,lighting,heating,cooling) from $REF_RUN: $RATES" | tee "$LOGS/batch2_rates_e${EPOCH}.txt"

CMDS="$LOGS/batch2_cmds_e${EPOCH}.txt"; : > "$CMDS"
add() { local name="$1"; shift
  [ -f "$AAMAS/runs/$name/manifest.json" ] && grep -q '"status": "ok"' "$AAMAS/runs/$name/manifest.json" && { echo "skip $name (done)"; return; }
  echo "python $SCRIPTS/run_eval.py --run-name $name --checkpoint $CKPT --overwrite $* > $LOGS/$name.log 2>&1 && echo OK $name || echo FAIL $name" >> "$CMDS"; }
for s in 20260729 20260730 20260731 20260732; do add "b3e${EPOCH}_sample_s$s" --mode sample --selector-seed $s; done
add "b3e${EPOCH}_threshold" --mode threshold
for s in 20260728 20260729 20260730 20260731 20260732; do
  add "b3e${EPOCH}_constant_s$s" --mode constant --selector-seed $s --rates "$RATES"
  add "b3e${EPOCH}_periodic_s$s" --mode periodic --selector-seed $s --phase-seed $s --rates "$RATES"
done
add "b3e${EPOCH}_always" --mode always
add "b3e${EPOCH}_sample_s20260728_sgoff" --mode sample --selector-seed 20260728 --safeguard off

echo "$(wc -l < "$CMDS") runs queued, $JOBS in parallel. Progress: $LOGS/<run>.log"
xargs -P "$JOBS" -I{} bash -c "{}" < "$CMDS"
python "$SCRIPTS/make_results.py" --main-epoch "$EPOCH" && echo "paper tables/figure regenerated in $AAMAS/paper"
echo "Batch 2 finished."
