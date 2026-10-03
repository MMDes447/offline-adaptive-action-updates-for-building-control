"""Weight-level provenance check for SDAR rollouts.

The check replays a rollout's logged observations and previous actions
through a checkpoint's GRU encoder and selector (teacher forcing), then
compares the resulting selector logits with the logits the rollout logged.
If the checkpoint produced the rollout, the mean absolute difference is
numerical noise (< 1e-2). A different checkpoint gives differences of
about 0.5 or more.

The check exists because a rollout labelled "epoch 500" turned out to have
come from epoch 100: its metadata named the epoch-500 file, but the weights
that actually ran were epoch 100's.
"""
from __future__ import annotations

import numpy as np
import torch

MATCH_TOL = 1e-2  # mean |delta logit| below this means the weights match


def _load_modules(checkpoint_path: str):
    import sdar_iql_train_4_updated as T  # model classes, identical to the beta3 notebook

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    gru = T.GRUEncoder(cfg["obs_dim"], cfg["latent_dim"], cfg.get("gru_layers", 1))
    gru.load_state_dict(ckpt["gru_encoder"])
    gru.eval()
    sel = T.SDARSelector(
        cfg.get("selector_input_dim", cfg["rep_dim"] + cfg.get("action_dim", 19)),
        cfg.get("mask_dim", 19),
        cfg["hidden_dim"],
    )
    sel.load_state_dict(ckpt["selector"])
    sel.eval()
    norm = T.Normalizer()
    norm.load_state_dict(ckpt["obs_normalizer"])
    norm.mean = norm.mean.detach().cpu().float()
    norm.std = norm.std.detach().cpu().float()
    actor = T.SDARFlatHybridActor(
        cfg.get("proposal_input_dim", cfg["rep_dim"] + cfg.get("action_dim", 19)), cfg["hidden_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    _load_modules.actor = actor
    return gru, sel, norm, int(cfg["seq_len"]), ckpt.get("epoch")


ACTOR_MATCH_MIN = 0.95  # share of updated heating setpoints reproduced to 1e-3


def teacher_forced_actor(checkpoint_path: str, observations, action_mixes, n_steps=2000):
    """Deterministic proposals (glazing argmax, clipped Gaussian mean) for the logged
    observation history and logged action mix. For flat baselines the selector is a
    constant, so this is the check that identifies the trained weights."""
    gru, _sel, norm, seq_len, _ = _load_modules(checkpoint_path)
    actor = _load_modules.actor
    n = min(int(n_steps), len(observations))
    obs = torch.as_tensor(np.asarray(observations[:n]), dtype=torch.float32)
    mix = torch.as_tensor(np.asarray(action_mixes[:n]), dtype=torch.float32)
    obs_n = norm.normalize(obs)
    padded = torch.cat([torch.zeros(seq_len - 1, obs.shape[1]), obs_n])
    out = []
    with torch.no_grad():
        for start in range(0, n, 256):
            idx = range(start, min(n, start + 256))
            win = torch.stack([padded[t : t + seq_len] for t in idx])
            z, _ = gru(win)
            rep = torch.cat([z, win[:, -1, :]], dim=-1)
            prop = actor.sample(torch.cat([rep, mix[start : start + len(idx)]], dim=-1), deterministic=True)
            out.append(prop[:, 4:].clamp(-1.0, 1.0))
    return torch.cat(out).numpy()


def check_actor(npz_path: str, checkpoint_path: str, n_steps=2000) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "action_mixes" not in data.files:
        return {"actor_checked": False}
    cont = teacher_forced_actor(checkpoint_path, data["observations"], data["action_mixes"], n_steps)
    n = len(cont)
    logged = np.asarray(data["actions"][:n, 4:], dtype=np.float64)
    mask = np.asarray(data["selection_masks"][:n, 4:]) > 0.5
    heat = slice(5, 10)  # heating setpoints within the 15 continuous dims; no safeguard acts on them
    sel = mask[:, heat]
    d = np.abs(cont[:, heat] - logged[:, heat])[sel]
    share = float((d < 1e-3).mean()) if d.size else float("nan")
    return {"actor_checked": True, "actor_heating_dims_compared": int(d.size),
            "actor_share_reproduced": share, "actor_match": bool(d.size and share >= ACTOR_MATCH_MIN)}


def teacher_forced_logits(checkpoint_path: str, observations, prev_actions, n_steps=2000):
    gru, sel, norm, seq_len, epoch = _load_modules(checkpoint_path)
    n = min(int(n_steps), len(observations))
    obs = torch.as_tensor(np.asarray(observations[:n]), dtype=torch.float32)
    prev = torch.as_tensor(np.asarray(prev_actions[:n]), dtype=torch.float32)
    obs_n = norm.normalize(obs)
    padded = torch.cat([torch.zeros(seq_len - 1, obs.shape[1]), obs_n])
    out = []
    with torch.no_grad():
        for start in range(0, n, 256):
            idx = range(start, min(n, start + 256))
            win = torch.stack([padded[t : t + seq_len] for t in idx])
            z, _ = gru(win)
            rep = torch.cat([z, win[:, -1, :]], dim=-1)
            out.append(sel(rep, prev[start : start + len(idx)]))
    return torch.cat(out).numpy(), epoch


def check_rollout(npz_path: str, checkpoint_path: str, n_steps=2000) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    key = (
        "learned_selection_logits"
        if "learned_selection_logits" in data.files
        else "selection_logits"
    )
    logits, epoch = teacher_forced_logits(
        checkpoint_path, data["observations"], data["prev_actions"], n_steps
    )
    ref = data[key][: len(logits)]
    diff = np.abs(logits - ref)
    actor_res = check_actor(npz_path, checkpoint_path, n_steps)
    sel_match = bool(diff.mean() < MATCH_TOL)
    return {
        **actor_res,
        "selector_match": sel_match,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "npz": str(npz_path),
        "logit_key": key,
        "n_steps": int(len(logits)),
        "mean_abs_logit_diff": float(diff.mean()),
        "max_abs_logit_diff": float(diff.max()),
        "match": sel_match and actor_res.get("actor_match", True),
        "tolerance_mean": MATCH_TOL,
    }


if __name__ == "__main__":
    import argparse, json, sys, os

    ap = argparse.ArgumentParser(description="Which checkpoint produced this rollout NPZ?")
    ap.add_argument("npz")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--n-steps", type=int, default=2000)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    results = [check_rollout(a.npz, c, a.n_steps) for c in a.checkpoints]
    for r in sorted(results, key=lambda r: r["mean_abs_logit_diff"]):
        print(f"{r['mean_abs_logit_diff']:9.4f}  actor={r.get('actor_share_reproduced', float('nan')):.3f}  {'MATCH' if r['match'] else '     '}  {r['checkpoint']}")
    print(json.dumps(results, indent=2))
