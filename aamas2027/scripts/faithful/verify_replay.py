#!/usr/bin/env python3
"""Check that run_eval.py reproduces an existing rollout step by step.

    python aamas2027/scripts/faithful/verify_replay.py \
        --reference agent_eval_dataset_decomposed_epoch500_thermal3_tclip50_selector-sample_propdet_s20260728.npz \
        --checkpoint checkpoints/<beta3>/..._checkpoint_epoch100.pt --days 3

It runs the faithful pipeline (run_eval.py, mode sample) for the first N days,
but replaces every Bernoulli draw of the update selector by the mask that the
reference rollout logged at that step. Everything else - observation assembly,
GRU, proposal actor, safeguard, EnergyPlus and Radiance - runs for real. If the
evaluation code and simulation setup are the ones that produced the reference,
the observations and executed actions must match the reference step by step.

Why the replay: the selector draws come from a torch.Generator on the device
(cuda on the workstation). A CPU run draws a different random stream from the
same seed, so an unforced comparison would diverge for that reason alone.
On the same machine and device, run without --replay-masks to check the full
chain including the random draws.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent))
import run_eval  # noqa: E402

GROUPS = {"zone temperature": slice(0, 5), "lighting rate": slice(5, 10),
          "transmitted solar": slice(10, 14), "exterior irradiance": slice(14, 18),
          "WPI": slice(18, 22), "HVAC demand": slice(22, 23), "time/occupancy": slice(23, 29)}
ACT = {"glazing": slice(0, 4), "lighting": slice(4, 9), "heating": slice(9, 14), "cooling": slice(14, 19)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="rollout .npz produced by the notebook")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--days", type=int, default=3, help="0 = the full year")
    ap.add_argument("--selector-seed", type=int, default=20260728)
    ap.add_argument("--no-replay-masks", dest="replay", action="store_false")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()

    root = run_eval.PROJECT_ROOT
    refp = Path(a.reference)
    refp = refp if refp.is_absolute() else root / refp
    ref = np.load(refp, allow_pickle=True)
    ref_masks = ref["selection_masks"].astype(np.float32)

    import torch
    state = {"k": 0}
    real_bernoulli = torch.bernoulli

    def replay_bernoulli(probs, *args, **kwargs):
        state["k"] += 1                     # step 0 is forced and never draws
        if state["k"] >= len(ref_masks):    # the year's last step has no logged transition
            return real_bernoulli(probs, *args, **kwargs)
        m = torch.as_tensor(ref_masks[state["k"]], dtype=probs.dtype, device=probs.device)
        return m.reshape(probs.shape)

    def hook(ns):
        if a.replay:
            torch.bernoulli = replay_bernoulli
            print("[verify] selector draws replaced by the reference masks")

    name = a.run_name or f"verify_replay_{refp.stem[:40]}_{a.days or 365}d"
    argv = ["--checkpoint", a.checkpoint, "--mode", "sample", "--selector-seed", str(a.selector_seed),
            "--run-name", name, "--smoke-days", str(max(a.days, 0)), "--overwrite",
            "--runs-dir", str(run_eval.AAMAS_DIR / "runs_verify")]
    try:
        ns, manifest = run_eval.main(argv, before_run=hook)
    finally:
        torch.bernoulli = real_bernoulli

    ad = ns["agent_data"]
    obs = np.asarray(ad["observation"], dtype=np.float64)
    act = np.asarray(ad["executed_action_norm"], dtype=np.float64)
    n = len(obs)
    n = min(n, len(ref["observations"]))
    obs, act = obs[:n], act[:n]
    r_obs = ref["observations"][:n].astype(np.float64)
    r_act = ref["actions"][:n].astype(np.float64)
    report = {"reference": str(refp), "steps_compared": n, "replayed_masks": a.replay,
              "selector_draws_replaced": state["k"], "observations": {}, "actions": {}}
    print(f"\n=== step-by-step comparison, first {n} steps ===")
    print(f"{'observation group':22s} {'max |diff|':>12s} {'mean |diff|':>12s} {'ref scale':>10s}")
    for g, s in GROUPS.items():
        d = np.abs(obs[:, s] - r_obs[:, s])
        report["observations"][g] = {"max": float(d.max()), "mean": float(d.mean()),
                                     "ref_mean_abs": float(np.abs(r_obs[:, s]).mean())}
        print(f"{g:22s} {d.max():12.5g} {d.mean():12.5g} {np.abs(r_obs[:, s]).mean():10.4g}")
    print(f"{'executed action':22s} {'max |diff|':>12s} {'steps equal':>12s}")
    for g, s in ACT.items():
        d = np.abs(act[:, s] - r_act[:, s])
        eq = float((d.max(axis=1) < 1e-4).mean())
        report["actions"][g] = {"max": float(d.max()), "share_steps_equal": eq}
        print(f"{g:22s} {d.max():12.5g} {100 * eq:11.1f}%")
    worst_t = np.abs(obs[:, 0:5] - r_obs[:, 0:5]).max(axis=1)
    report["first_step_temp_diff_gt_0.01C"] = int(np.argmax(worst_t > 0.01)) if (worst_t > 0.01).any() else None
    out = run_eval.AAMAS_DIR / "runs_verify" / name / "replay_comparison.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
