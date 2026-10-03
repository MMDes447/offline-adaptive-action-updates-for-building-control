#!/usr/bin/env python3
"""Freeze the evaluation code exactly as it is in the project notebooks.

    python aamas2027/scripts/faithful/extract_cells.py          # extract + write CELLS.json
    python aamas2027/scripts/faithful/extract_cells.py --check  # verify frozen cells still match

Sources (found by searching the whole repo on 2026-10-01; see README.md here):

new_dataset_gen1.ipynb  -- the notebook that produced every beta3 rollout
  cells 0-4   frads model, SageGlass SR2 glazing systems, lighting, EnergyPlusSetup
  cell 14     evaluation callback (imports sdar_iql_train_3_updated.load_and_act)
  cell 46     agent-rollout dataset builder (the cell whose saved output is the
              7 Aug 01:54 beta3 rollout, run right after EnergyPlus finished)
  cell 8      v12 smooth expert = the behaviour controller (RBC), with the
              EPISODE_ID / NOISE_PROFILE knobs that produced the 14 data episodes
  cell 33     expert-episode builder (default prefix offline_<expert>_<profile>_episode_<id>,
              the code that wrote offline_smooth_clean_episode_000.*); used by run_rbc.py

Ablation_sdar_.ipynb    -- the notebook that produced the constant/periodic/always
                           selector-ablation rollouts (5-6 Aug)
  cells 0-4   identical to new_dataset_gen1 cells 0-4 (checked byte for byte)
  cell 5      ablation callback (imports sdar_iql_train_4_updated.load_and_act)
  cell 9      fixed-rate dataset builder

new_new_removed_batch_normalizatrion.ipynb -- the notebook that trained beta3
  cell 0      the complete SDAR-IQL training script (used by train_baselines.py)

Cells are written byte for byte. Nothing is edited here; parameters are set by
run_eval.py on the executed namespace, and every deviation is listed in README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ROOT = HERE.parent.parent.parent
CELLS = HERE / "cells"
SOURCES = {
    "new_dataset_gen1.ipynb": {
        "gen1_00_imports.py": 0,
        "gen1_01_model.py": 1,
        "gen1_02_glazing.py": 2,
        "gen1_03_add_glazing_lighting.py": 3,
        "gen1_04_energyplus_setup.py": 4,
        "gen1_14_eval_callback.py": 14,
        "gen1_46_rollout_builder.py": 46,
        "gen1_08_expert_v12.py": 8,
        "gen1_33_expert_builder.py": 33,
    },
    "Ablation_sdar_.ipynb": {
        "abl_05_ablation_callback.py": 5,
        "abl_09_fixed_rate_builder.py": 9,
    },
    "new_new_removed_batch_normalizatrion.ipynb": {
        "train_beta3_cell00.py": 0,
    },
}
SETUP_TWINS = {0: "gen1_00_imports.py", 1: "gen1_01_model.py", 2: "gen1_02_glazing.py",
               3: "gen1_03_add_glazing_lighting.py", 4: "gen1_04_energyplus_setup.py"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cell_source(nb, idx) -> str:
    return "".join(nb["cells"][idx]["source"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify, do not write")
    a = ap.parse_args()
    manifest = {"notebooks": {}, "cells": {}}
    problems = []
    for nb_name, cells in SOURCES.items():
        raw = (ROOT / nb_name).read_bytes()
        nb = json.loads(raw)
        manifest["notebooks"][nb_name] = {"sha256": sha(raw), "n_cells": len(nb["cells"])}
        for fname, idx in cells.items():
            src = cell_source(nb, idx)
            data = src.encode("utf-8")
            manifest["cells"][fname] = {"notebook": nb_name, "cell": idx, "sha256": sha(data),
                                        "lines": len(src.splitlines())}
            target = CELLS / fname
            if a.check:
                if not target.is_file() or sha(target.read_bytes()) != sha(data):
                    problems.append(f"{fname}: frozen copy differs from {nb_name} cell {idx}")
            else:
                CELLS.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        if nb_name == "Ablation_sdar_.ipynb":
            gen1 = json.loads((ROOT / "new_dataset_gen1.ipynb").read_bytes())
            for i, twin in SETUP_TWINS.items():
                if cell_source(nb, i) != cell_source(gen1, i):
                    problems.append(f"setup cell {i} differs between Ablation_sdar_ and new_dataset_gen1")
    if a.check:
        old = json.loads((HERE / "CELLS.json").read_text())
        if old["cells"] != manifest["cells"]:
            problems.append("CELLS.json does not match the notebooks any more")
    else:
        (HERE / "CELLS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for p in problems:
        print("MISMATCH:", p)
    print("OK: frozen cells match the notebooks" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
