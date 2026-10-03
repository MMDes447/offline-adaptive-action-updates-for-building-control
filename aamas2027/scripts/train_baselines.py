#!/usr/bin/env python3
"""Train SDAR-IQL (extra training seeds) and the offline baselines with the exact
beta3 training code.

    python aamas2027/scripts/train_baselines.py --algo iql_flat --seed 1
    python aamas2027/scripts/train_baselines.py --algo bc_flat  --seed 1
    python aamas2027/scripts/train_baselines.py --algo bc_sdar  --seed 1
    python aamas2027/scripts/train_baselines.py --algo sdar_iql --seed 1   # beta3, new training seed

The training code is the single cell of new_new_removed_batch_normalizatrion.ipynb,
frozen byte for byte in faithful/cells/train_beta3_cell00.py (sha256 checked).
It is executed unchanged apart from three documented text substitutions
(run name / checkpoint directory, number of epochs) and, for the baselines,
replacement of the loss functions listed below. Data, reward, encoder, actor,
critic architecture, optimiser, learning rates, batch size, expectile, beta,
gamma, w_max and lambda values are those of beta3.

Algorithms (everything not listed is identical to beta3):

  sdar_iql  beta3 exactly (IQL critic with augmented value, exp(3A) weighting,
            per-dimension update selector + masked hybrid proposal). Only the
            random seed differs, to measure training-run variability.
  iql_flat  IQL without action repetition. Same critic and weighting, but no
            update selector: the actor is trained on the full 19-dim logged
            executed action (log-likelihood of every dimension) and acts on all
            dimensions at every step. At export the selector is replaced by a
            constant "update everything" selector (logit +30, p = 1), so the
            unchanged evaluation code executes the full action every step.
  bc_flat   Behaviour cloning. Same encoder and actor as iql_flat, trained only
            by the unweighted log-likelihood of the logged executed action
            (critic losses switched off, so they do not shape the encoder).
  bc_sdar   Behaviour cloning with the SDAR structure: selector imitates the
            logged update masks, proposal imitates the logged values on updated
            dimensions, both unweighted; critic off. Isolates the contribution
            of the IQL critic from that of the act/repeat architecture.
  iql_flat_prev  as iql_flat, but the actor receives the previously executed action
            a_{t-1} as its second input instead of the all-marker vector (one text
            substitution in the training loop: the proposal loss is called with
            prev_action instead of action_mix). It can therefore hold a command by
            re-issuing it. Evaluated with run_eval.py --mode always_prev (batch 7).
  sdar_iql_k1  beta3 exactly, except SEQ_LEN = 36 -> 1 (one extra text
            substitution): the GRU sees only the current observation, i.e. no
            observation history. Ablation of the recurrent history (batch 6).
            The evaluation code reads seq_len from the checkpoint config, so the
            unchanged run_eval.py evaluates it with a 1-step window.

Checkpoints go to checkpoints/aamas2027/<run-name>/ (never into existing runs);
a train_manifest.json records the seed, code hashes and timings.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
AAMAS = HERE.parent
ROOT = AAMAS.parent
CELL = HERE / "faithful" / "cells" / "train_beta3_cell00.py"
CELLS_JSON = HERE / "faithful" / "CELLS.json"
ALGOS = ("sdar_iql", "iql_flat", "bc_flat", "bc_sdar", "sdar_iql_k1", "iql_flat_prev")
BASE = "sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3"
CONST_SELECTOR_LOGIT = 30.0          # sigmoid(30) == 1.0 in float32


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


def run_name(algo: str, seed: int) -> str:
    return f"aamas_{algo}_seed{seed}"


def load_cell(run: str, ckpt_root: str, epochs: int, steps: int | None, seq_len: int | None = None,
              proposal_sees_prev: bool = False) -> str:
    data = CELL.read_bytes()
    meta = json.loads(CELLS_JSON.read_text())["cells"]["train_beta3_cell00.py"]
    if sha256_bytes(data) != meta["sha256"]:
        raise RuntimeError("train_beta3_cell00.py differs from the notebook cell recorded in CELLS.json")
    src = data.decode("utf-8")

    def sub(pattern, repl, flags=0):
        nonlocal src
        new, n = re.subn(pattern, repl, src, count=1, flags=flags)
        if n != 1:
            raise RuntimeError(f"text substitution failed: {pattern!r}")
        src = new

    # 1. run name, 2. checkpoint directory (keeps all new runs under checkpoints/aamas2027/)
    sub(r'RUN_NAME = \(\n\s*"sdar_iql_v10_thermal3_tclip50_"\n\s*"14episodes_gru2_seq36_augmented_value_standard_iql_beta3"\n\)',
        f'RUN_NAME = {run!r}')
    sub(r'CHECKPOINT_DIR = Path\("checkpoints"\) / RUN_NAME',
        f'CHECKPOINT_DIR = Path({ckpt_root!r}) / RUN_NAME')
    # 3. training length
    sub(r'^NUM_EPOCHS = 500\b', f'NUM_EPOCHS = {int(epochs)}', flags=re.M)
    if steps is not None:   # tests only
        sub(r'^STEPS_PER_EPOCH = 1000\b', f'STEPS_PER_EPOCH = {int(steps)}', flags=re.M)
    # 4. (sdar_iql_k1 only) observation history length
    if seq_len is not None:
        sub(r'^SEQ_LEN = 36\b', f'SEQ_LEN = {int(seq_len)}', flags=re.M)
    # 5. (iql_flat_prev only) the proposal loss receives the previous executed action
    if proposal_sees_prev:
        sub(r'loss_pi = sdar_proposal_loss\(actor, rep, action_mix, action_actor, selection_mask, weights\)',
            'loss_pi = sdar_proposal_loss(actor, rep, prev_action, action_actor, selection_mask, weights)')
    return src


def seed_everything(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def install_algo(ns: dict, algo: str):
    """Replace loss functions in the executed training namespace (looked up by
    name inside train() at call time)."""
    import torch
    if algo in ("sdar_iql", "sdar_iql_k1"):
        return

    def zero_like_params(module_params):
        # zero-valued loss that still gives every parameter a (zero) gradient, so
        # the AMP GradScaler records inf-checks for every optimiser
        return 0.0 * sum(p.sum() for p in module_params)

    if algo == "iql_flat_prev":
        def selector_off_p(selector, rep, prev_action, selection_mask, weights):
            return zero_like_params(selector.parameters())

        def proposal_full_action_prev(actor, rep, prev_action, action_actor_format, selection_mask, weights):
            # second argument is a_{t-1} (substitution 5), not the masked action mix
            proposal_rep = torch.cat([rep, prev_action], dim=-1)
            log_prob = actor.log_prob(proposal_rep, action_actor_format)
            return -(weights * log_prob).mean()

        ns["sdar_selector_loss"] = selector_off_p
        ns["sdar_proposal_loss"] = proposal_full_action_prev

    if algo in ("iql_flat", "bc_flat"):
        mask_value = ns["ACTION_MASK_VALUE"]

        def selector_off(selector, rep, prev_action, selection_mask, weights):
            return zero_like_params(selector.parameters())

        def proposal_full_action(actor, rep, action_mix, action_actor_format, selection_mask, weights):
            full_mix = torch.full_like(action_mix, mask_value)      # every dimension is proposed
            proposal_rep = torch.cat([rep, full_mix], dim=-1)
            log_prob = actor.log_prob(proposal_rep, action_actor_format)
            return -(weights * log_prob).mean()

        ns["sdar_selector_loss"] = selector_off
        ns["sdar_proposal_loss"] = proposal_full_action

    if algo in ("bc_flat", "bc_sdar"):
        def unit_weights(qnet_target, vnet, rep, value_state, action_q_input, beta):
            ones = torch.ones(rep.shape[0], 1, device=rep.device, dtype=rep.dtype)
            return ones, torch.zeros_like(ones)

        def critic_off_v(qnet_target, vnet, rep, value_state, action_q_input, expectile):
            return zero_like_params(vnet.parameters())

        def critic_off_q(qnet, vnet, rep, action_q_input, reward, value_state_next, done, gamma):
            return zero_like_params(qnet.parameters())

        ns["compute_iql_weights"] = unit_weights
        ns["iql_value_loss"] = critic_off_v
        ns["iql_q_loss"] = critic_off_q


def export_constant_selector(ckpt_path: Path, algo: str, seed: int):
    """Flat baselines act on every dimension every step: make the stored selector
    output p = 1 so the unchanged evaluation code executes the full action."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["selector"]
    last_bias = sorted(k for k in sd if k.endswith(".bias"))
    final = max((k for k in sd if k.endswith("bias")), key=lambda k: int(k.split(".")[-2]))
    for k in sd:
        sd[k] = torch.zeros_like(sd[k])
    sd[final] = torch.full_like(sd[final], CONST_SELECTOR_LOGIT)
    ck["selector"] = sd
    ck["config"]["selector_class"] = "constant_always_update"
    ck["config"]["aamas_baseline"] = algo
    ck["config"]["aamas_train_seed"] = seed
    torch.save(ck, ckpt_path)
    return final, last_bias


def tag_checkpoint(ckpt_path: Path, algo: str, seed: int):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck["config"]["aamas_baseline"] = algo
    ck["config"]["aamas_train_seed"] = seed
    torch.save(ck, ckpt_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=100, help="beta3 is reported at 100 epochs")
    ap.add_argument("--ckpt-root", default="checkpoints/aamas2027")
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="TEST ONLY")
    a = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    run = run_name(a.algo, a.seed)
    out = Path(a.ckpt_root) / run
    if (out / "train_manifest.json").is_file():
        m = json.loads((out / "train_manifest.json").read_text())
        if m.get("status") == "ok":
            sys.exit(f"{out} already trained (status ok); refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "algo": a.algo, "seed": a.seed, "epochs": a.epochs,
                "run_name": run, "checkpoint_dir": str(out), "started": _dt.datetime.now().isoformat(timespec="seconds"),
                "command": " ".join([sys.executable] + sys.argv),
                "code": {"train_baselines.py": sha256_file(__file__), "train_beta3_cell00.py": sha256_file(CELL),
                         "notebook": json.loads(CELLS_JSON.read_text())["notebooks"].get(
                             "new_new_removed_batch_normalizatrion.ipynb")},
                "dataset_sha256": None}
    mpath = out / "train_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    t0 = time.time()
    try:
        seq_len = 1 if a.algo == "sdar_iql_k1" else None
        manifest["seq_len_override"] = seq_len
        manifest["proposal_sees_prev"] = (a.algo == "iql_flat_prev")
        src = load_cell(run, a.ckpt_root, a.epochs, a.steps_per_epoch, seq_len,
                        proposal_sees_prev=(a.algo == "iql_flat_prev"))
        ns = {"__name__": "beta3_training_cell", "__file__": str(CELL)}
        exec(compile(src, str(CELL), "exec"), ns)
        manifest["dataset_sha256"] = sha256_file(ROOT / ns["DATASET_PATH"])
        seed_everything(a.seed)
        install_algo(ns, a.algo)
        ns["train"]()
        ckpts = sorted(out.glob("*.pt"))
        for p in ckpts:
            if a.algo in ("iql_flat", "bc_flat", "iql_flat_prev"):
                export_constant_selector(p, a.algo, a.seed)
            else:
                tag_checkpoint(p, a.algo, a.seed)
        manifest["checkpoints"] = {p.name: sha256_file(p) for p in sorted(out.glob("*.pt"))}
        manifest["status"] = "ok" if a.steps_per_epoch is None else "ok_test_only"
    except BaseException as e:
        import traceback
        manifest["status"] = "failed"
        manifest["error"] = f"{type(e).__name__}: {e}"
        manifest["traceback"] = traceback.format_exc()
        print(manifest["traceback"], file=sys.stderr)
        raise
    finally:
        manifest["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        manifest["train_s"] = round(time.time() - t0, 1)
        mpath.write_text(json.dumps(manifest, indent=2))
    print(f"OK {run}: {len(manifest['checkpoints'])} checkpoints in {out}")


if __name__ == "__main__":
    main()
