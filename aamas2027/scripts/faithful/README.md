# Faithful evaluation: which code runs, and where it comes from

I searched the whole project on 2026-10-01 for code that runs a trained agent in EnergyPlus. Four notebooks contain evaluation code: `new_dataset_gen1.ipynb`, `Ablation_sdar_.ipynb`, `off_data_rl_2.ipynb` and `train_decomposed_critic.ipynb`.

**The beta3 rollouts came from `new_dataset_gen1.ipynb`.** Its saved cell outputs show the following, in order:

1. **Callback, cell 14.** `EVAL_CHECKPOINT` is the beta3 epoch-100 file. The selector is `SELECTOR_MODE = "sample"` with seed 20260728. It imports `load_and_act` from `sdar_iql_train_3_updated.py`.
2. **`eps.set_callback`, cell 16.**
3. **`eps.run(annual=True)`, cell 17.** The saved output reads "EnergyPlus 25.2.0 … YMD=2026.08.07 01:45 … Run Time 00hr 08min 47s".
4. **Builder, cell 46.** Its output reads "Saved CSV: agent_eval_dataset_decomposed_epoch500_…_s20260728.csv (52559 rows, 151 cols)". That file carries the beta3 epoch-100 weights; its label says epoch 500 because the builder's `EVAL_CHECKPOINT_PATH` was not updated.

**The selector-ablation rollouts (constant, periodic, always) came from `Ablation_sdar_.ipynb`,** cells 5 and 9. That code uses `sdar_iql_train_4_updated.py`.

**The simulation setup is the same in both notebooks.** Cells 0–4 are identical byte for byte:

- the frads `medium_office` reference model;
- 4 SageGlass SR2 electrochromic glazing systems (clear, light, medium and full tint, each paired with `igsdb_product_14028`, 90 % argon);
- lighting levels of 2231, 2231, 1412, 1412 and 10586 W;
- `EnergyPlusSetup(epmodel, weather_files["usa_ca_san_francisco"], enable_radiance=True)`.

## What `run_eval.py` executes

| Mode | Cells executed | Policy module |
|---|---|---|
| `sample`, `threshold` | `new_dataset_gen1.ipynb` 0–4, 14, 46 | `sdar_iql_train_3_updated.py` |
| `constant`, `periodic`, `always` | `Ablation_sdar_.ipynb` 0–4, 5, 9, with the patches in `../sdar_eval/PATCHES.md` | `sdar_iql_train_4_updated.py` |

The cells are copied **byte for byte** into `cells/`, and `CELLS.json` records the SHA-256 of each one.

- `run_eval.py` refuses to run a cell whose hash no longer matches.
- If you change a notebook on purpose, run `python aamas2027/scripts/faithful/extract_cells.py`.
- `--check` compares the copies with the notebooks without writing anything.

## The only deviations from running the notebook by hand

1. **Parameters.** The checkpoint, selector mode, selector seed and safeguard switch are set on the executed namespace, not typed into the cell. The callback's `setup()` reads them when it is called, so the effect is identical.
2. **EnergyPlus output.** It goes to `runs/<name>/eplus/` instead of the project root. This changes where the files are written, not what is simulated.
3. **Builder output.** The builder is called with `prefix=runs/<name>/rollout`, so it never overwrites the `agent_eval_dataset_*` files in the project root.
4. **Provenance check.** After every run, the logged selector logits are compared with the checkpoint named in the run. A mismatch marks the run as `failed`.
5. **`--smoke-days N` (tests only).** A separate hook stops EnergyPlus after the first N days.

`sdar_eval/sim_setup.py`, my earlier re-implementation of cells 0–4 with a pickled glazing cache, is **no longer used**. The notebook cells now run as they are.

## Consistency of the two policy modules

`sdar_iql_train_3_updated.py` (used for beta3) and `sdar_iql_train_4_updated.py` (used for the ablation) differ only inside `load_and_act`:

- version 4 adds the constant, periodic and always modes;
- version 4 adds extra logging.

The sample and threshold paths draw from the same seeded `torch.Generator`, in the same order. The network classes are identical to the beta3 training notebook, apart from docstrings.

## Verification

`verify_replay.py` runs this pipeline for the first N days of a year. It replaces each selector draw with the mask that the reference rollout logged, and then compares observations and executed actions step by step.

The replay is needed because the notebook drew its masks from a CUDA generator. A CPU run gets a different random stream from the same seed.

On the workstation, run it without `--no-replay-masks` for the masked comparison. With `--no-replay-masks` on the same GPU, it checks the full chain, random draws included.

## Behaviour controller (RBC) runs: `run_rbc.py` (added 2 Oct)

The clean behaviour controller is rerun with the code that generated the 14 data episodes, all from `new_dataset_gen1.ipynb`:

- cells 0–4: the same model, glazing and lighting cells as the agent runs (weather swapped for non-SF runs, exactly as in `run_eval.py`);
- **cell 8** (`gen1_08_expert_v12.py`): the v12 smooth expert with callback-side exploration;
- cells 16/17: `set_callback` and `eps.run(annual=True)`;
- **cell 33** (`gen1_33_expert_builder.py`): the episode builder whose default prefix is `offline_smooth_<profile>_episode_<id>`.

Deviations:

1. Two literal lines of cell 8 are substituted, as one would edit them by hand for the clean episode: `EPISODE_ID = 13` → `0` and `NOISE_PROFILE = "high"` → `"clean"`. With `"clean"`, `EXPLORATION_ENABLED` is false, so there is no action noise and no forced repetition; the controller is deterministic.
2. Outputs go to `runs/<name>/` (EnergyPlus in `eplus/`, the builder writes `rollout.*`). The builder's reward scales are dataset-local, as in the original episode; only the physical KPIs are used.
3. `--check-reference` (San Francisco only) compares the annual KPIs with `offline_smooth_clean_episode_000.csv` and marks the run failed outside 1 % electricity / 0.3 pp violations / 0.5 pp visual / 0.3 pp update rate. Batch 5 runs this check before any new-weather RBC run.

**Thermostat band (found 2 Oct by the reproduction check).** The notebook's cell 8 now sets `COMFORT_BAND_LOW/HIGH = 22/25 C` and `COMFORT_OVERRIDE_SP = 24 C`. The first batch-5 run used it unchanged and failed the check: 396.9 kWh/day and 6.95 % violations instead of 401.5 and 19.55 %, with identical glazing and lighting update rates but 18 % instead of 46 % thermal updates. Replaying the logged zone temperatures through the cell's thermal rule:

- the clean episode's heating setpoints are reproduced at 100 % with a 21–24 C band and at 49 % with 22–25 C;
- for episodes 1–12, the share of idle setpoints at 21–22 C matches each noise profile's idle probability (0.85 low, 0.70 medium, 0.56 high) only with a 21–24 C band;
- episode 13 (high noise, collected last) has no idle setpoints at 21–22 C, i.e. it used the 22–25 C band.

`run_rbc.py --band data` (default) therefore restores 21/24/22 C, which is what episodes 0–12 were collected with. `--band notebook` keeps the cell as it is; batch 5 runs it as a second, retuned rule-based baseline (`rbc_b2225_*`).
