# Adaptive Action Updates in Offline Reinforcement Learning for Integrated Energy-Efficient Building Control

Code, trained models and run records for AAMAS 2027 submission #2458
(anonymous for double-blind review).

The agent learns, from the logs of a rule-based controller alone, **when to change each
of 19 actuator commands** (4 electrochromic windows, 5 lighting zones, 10 heating/cooling
setpoints) of an EnergyPlus–Radiance office model. It is an offline extension of SDAR
(spatially decoupled action repetition) trained with implicit Q-learning (IQL).

## Contents

| Path | What it is |
|---|---|
| `new_dataset_gen1.ipynb` | Data collection (behaviour controller, cell 8; episode builder, cell 33), dataset pooling (cell 35) and reward re-weighting (cell 43), agent evaluation callback (cell 14) and rollout builder (cell 46) |
| `new_new_removed_batch_normalizatrion.ipynb` | Training script of the reported agent (single cell) |
| `Ablation_sdar_.ipynb` | Evaluation code for the state-independent selector ablations (cells 5 and 9) |
| `sdar_iql_train_3_updated.py`, `sdar_iql_train_4_updated.py` | Network classes and `load_and_act` used at evaluation (identical networks; v4 adds the ablation selector modes) |
| `aamas2027/scripts/` | Reproduction pipeline: training of extra runs and baselines, annual evaluations, tables and figures (see below) |
| `aamas2027/scripts/faithful/cells/` | The notebook cells above, frozen byte for byte; `CELLS.json` holds their SHA-256 |
| `aamas2027/scripts/sdar_eval/` | Evaluation wrappers, KPI definitions, weight-level provenance check; `PATCHES.md` lists every deviation from the notebook code |
| `glaizng_2/`, `glaizngs/` | Optical data of the SageGlass SR2 electrochromic tints and the inner pane (IGSDB 14028) used to build the glazing systems |
| `checkpoints/` | The five trained agents of the paper (100 epochs each) |
| `aamas2027/batch7_inputs/` | Hour-of-day update table of the time-of-day selector |
| `results/run_manifests/` | One record per annual evaluation used in the paper (105 runs): code hashes, checkpoint SHA-256, library versions, EnergyPlus version, provenance check |
| `results/summaries/` | Machine-readable values behind every table and figure |
| `results/paper_tables/` | The generated LaTeX tables and macros as they appear in the paper |
| `supplementary/` | Supplementary material (PDF) |

Folder and file names (e.g. `glaizng_2`) are kept exactly as in the original project,
because the frozen notebook cells reference them.

## Environment

Python 3.11 with EnergyPlus 25.2 (via `pyenergyplus`) and Radiance (via `frads`/`pyradiance`).
The experiments ran on Linux with one NVIDIA RTX 5090; versions recorded in every run manifest:

```
python 3.11.15   torch 2.11.0+cu128   numpy 2.4.6   pandas 3.0.2
frads 2.1.15     pyradiance 1.1.5     EnergyPlus 25.2.0
```

```bash
conda create -n offrl_5zone python=3.11
conda activate offrl_5zone
pip install -r requirements.txt
```

The building model (`frads` reference *medium office*) and the TMY3 weather files
(San Francisco for training and evaluation, Fresno as held-out climate) ship with
`pyenergyplus`; no other inputs are needed for the simulations.

All commands below run from the repository root.

## Pipeline

### 1. Offline dataset (14 annual episodes)
`new_dataset_gen1.ipynb`, cell 8, is the behaviour controller with its exploration knobs
(`EPISODE_ID`, `NOISE_PROFILE` and the repetition probabilities; per-episode settings in
the supplementary material, Tables S1–S2). Cell 33 writes one episode
(`offline_smooth_<profile>_episode_<id>.npz`), cell 35 pools the 14 episodes into
`pooled_sdar_experts.npz`, and cell 43 re-weights the reward components
(thermal weight 3, thermal clip 50) into `pooled_sdar_experts_thermal3_tclip50.npz`,
the training set (SHA-256 `5c6c255e3b7791c4c31b76ea865a36a1933edb3b443a1406970056c7582cadfb`).
The pooled dataset (148 MB) is not part of this repository; it can be regenerated as above.

The clean behaviour controller is rerun under any weather with
```bash
python aamas2027/scripts/run_rbc.py --run-name rbc_clean_sf                       # 21-24 C band (data)
python aamas2027/scripts/run_rbc.py --band notebook --run-name rbc_b2225_sf        # retuned 22-25 C band
python aamas2027/scripts/run_rbc.py --weather usa_ca_fresno --run-name rbc_clean_fresno
```

### 2. Training
The reported agent is the single cell of `new_new_removed_batch_normalizatrion.ipynb`
(500 epochs; the paper uses the 100-epoch checkpoint). Further training runs and all
baselines use the same cell through
```bash
python aamas2027/scripts/train_baselines.py --algo sdar_iql      --seed 1   # our method, new training seed
python aamas2027/scripts/train_baselines.py --algo iql_flat      --seed 1   # IQL (full action)
python aamas2027/scripts/train_baselines.py --algo iql_flat_prev --seed 1   # IQL (full action) + a_{t-1}
python aamas2027/scripts/train_baselines.py --algo bc_flat       --seed 1   # BC (full action)
python aamas2027/scripts/train_baselines.py --algo bc_sdar       --seed 1   # BC (act/repeat)
python aamas2027/scripts/train_baselines.py --algo sdar_iql_k1   --seed 1   # one-step history (K=1)
```
Each run writes `checkpoints/aamas2027/<run>/` and a `train_manifest.json` with code and dataset hashes.

### 3. Annual evaluation
```bash
python aamas2027/scripts/run_eval.py --run-name b3e100_sample_s20260728 \
  --checkpoint checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch100.pt \
  --mode sample --selector-seed 20260728
```
Selector modes: `sample` (learned, reported), `threshold`, `constant` (random, rate-matched),
`periodic`, `always`, `clock` (time-of-day, needs `--hourly-rates`), `always_prev`
(full-action baseline whose policy sees the previous action). `--weather usa_ca_fresno`
selects the held-out climate. Every run executes the frozen notebook cells, refuses to run
if a cell no longer matches its recorded hash, and finishes with a weight-level provenance
check (`sdar_eval/provenance.py`): the logged selector logits and proposals are recomputed
from the checkpoint by teacher forcing and must match.

One annual evaluation takes about 15 minutes.

### 4. Tables and figures
| Script | Paper |
|---|---|
| `make_results.py --main-epoch 100` | Table 1, Table 4, Figure 1 |
| `make_baselines.py` | Table 2 |
| `make_batch5.py` | Table 3, Figure 2, supplementary Table S3 |
| `make_timing_figure.py` | Figure 3 |
| `make_components.py` | Table 5, Figure 4, supplementary Table S4 |
| `make_hourly_rates.py` | time-of-day selector table (`aamas2027/batch7_inputs/`) |

Statistics: day-paired differences with 95 % intervals from a 30-day moving-block bootstrap.

### 5. Exact experiment sequence
`aamas2027/scripts/run_batch1_provenance.sh` … `run_batch7_prev_clock.sh` are the scripts
that produced every run in `results/run_manifests/` (batch 1: provenance of the reported
checkpoint; 2: selector ablation; 3: component ablation and training length;
4: offline baselines; 5: retuned controller, held-out climate, update-rate sweep;
6: further training runs and one-step history; 7: time-of-day selector and the
full-action baseline with the previous action). They skip finished runs and never overwrite results.

## Checkpoints

| Training run | File |
|---|---|
| 0 (reported) | `checkpoints/sdar_iql_v10_..._beta3/..._checkpoint_epoch100.pt` |
| 1, 2, 5, 6 | `checkpoints/aamas2027/aamas_sdar_iql_seed{1,2,5,6}/..._checkpoint_epoch100.pt` |

`CHECKSUMS.sha256` lists their SHA-256; the same values appear in the run manifests of
every evaluation that used them.

## Notes on this release
* Notebook outputs were cleared; cell sources are unchanged
  (`python aamas2027/scripts/faithful/extract_cells.py --check` passes).
* Absolute paths, user and host names were removed from the run records; the
  `conda.sh` fall-back path in the batch scripts was replaced by `$HOME/miniconda3`.
  No other file was edited.
* Raw annual rollouts (100–240 MB each) and the pooled dataset are not included
  because of their size; every number in the paper can be traced to a run record in
  `results/run_manifests/` and recomputed with the scripts above.

## License
MIT (see `LICENSE`). The glazing optical data come from the International Glazing
Database (IGSDB) and keep their original terms.
