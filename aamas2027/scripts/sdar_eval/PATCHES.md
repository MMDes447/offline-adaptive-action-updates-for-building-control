# Patches relative to Ablation_sdar_.ipynb

Notebook sha256: `5e2e2d0c7b1f4bef5f76116f65a14adfc279990fcafd710d1947c9f8672c7edd`

1. callback: SELECTOR_MODE default read from env (runner always passes it explicitly to setup())
2. builder: accept sample/threshold/always in addition to constant/periodic
3. builder: selector_stochastic expected True for sample and constant
4. builder: fixed-rate target checks only for constant/periodic; always -> all-ones mask check; learned -> active==learned probability check
5. builder: mask description for every mode (print only)
6. builder: define EVAL_SELECTOR_MODE default (was commented out in the saved notebook -> NameError)
7. builder: disable stray SELECTOR_MODE assignment

These two patched files are used only for the constant, periodic and always selector modes.

Cells 0-4 (model, glazing, EnergyPlusSetup) are no longer re-implemented here. `run_eval.py` executes the notebook cells verbatim from `../faithful/cells/`. The sample and threshold modes use `new_dataset_gen1.ipynb` cells 14 and 46 unpatched; see `../faithful/README.md`.

## Batch 7 additions (2026-10-03)
8. callback: setup() accepts hourly_update_rates; modes "clock" and "always_prev" load the agent through
   policy_ext.load_and_act, which is sdar_iql_train_4_updated.load_and_act with exact text substitutions
   (listed in policy_ext.py; each must match once). All other modes still use the original function.
9. builder: accepts "clock" and "always_prev"; clock rollouts must log, at every step, the hour-of-day row of
   HOURLY_UPDATE_RATES (checked against the observed time of day); always_prev uses the always-update checks.
10. builder: in mode always_prev the stored proposal input (action_mixes) is the previous executed action,
   the input the proposal policy actually received (policy_ext substitution 6). Without this the NPZ stored
   the -2 update markers and the weight-level actor provenance check failed (heating setpoints reproduced
   1.6% with the markers vs 99.9% with the previous action, checked offline on iql_flat_prev seed 1).
   The three batch-7 always_prev evaluations of 2026-10-03_0223 are left as they are (status failed).
