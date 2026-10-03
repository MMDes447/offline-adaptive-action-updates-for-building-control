#!/usr/bin/env python3
"""Hour-of-day update-probability table of the learned selector (batch 7, mode clock).

    python aamas2027/scripts/make_hourly_rates.py \
        --rollout aamas2027/runs/b3e100_sample_s20260728/rollout.npz \
        --out aamas2027/batch7_inputs/hourly_rates_b3e100.json

For every hour of day h and action dimension i, the table holds the mean of the
learned selector's update probability p_{t,i} over all logged steps whose time of
day falls in hour h (time of day decoded from the observation's hour sin/cos
features, obs[23] and obs[24]). The first, forced step is excluded, as in the
rate-matched batch-2/5 selectors. The clock selector built from this table has, by
construction, the learned selector's per-dimension budget and its average daily
profile, but no access to the state. Refuses to overwrite an existing --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    d = np.load(a.rollout, allow_pickle=True)
    key = "learned_selection_probabilities" if "learned_selection_probabilities" in d.files else "selection_probabilities"
    p = np.asarray(d[key], dtype=np.float64)[1:]
    obs = np.asarray(d["observations"], dtype=np.float64)[1:]
    tod = np.mod(np.arctan2(obs[:, 23], obs[:, 24]) / (2.0 * np.pi), 1.0)
    hours = np.minimum((tod * 24.0 + 1e-6).astype(int), 23)
    table = np.zeros((24, p.shape[1]))
    counts = np.bincount(hours, minlength=24)
    if np.any(counts == 0):
        raise SystemExit(f"hours without data: {np.flatnonzero(counts == 0)}")
    for h in range(24):
        table[h] = p[hours == h].mean(axis=0)
    # keep every probability strictly inside (0, 1) so that the logged logits round-trip
    # through the sigmoid within the builder's tolerance (no exact 0 or 1)
    table = np.clip(table, 1e-6, 1.0 - 1e-6)
    sub = {"glazing": slice(0, 4), "lighting": slice(4, 9), "heating": slice(9, 14), "cooling": slice(14, 19)}
    overall = {k: float(p[:, s].mean()) for k, s in sub.items()}
    from_table = {k: float((table[:, s] * counts[:, None]).sum() / counts.sum() / (s.stop - s.start)) for k, s in sub.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "hourly_update_rates": table.tolist(),
        "source_rollout": str(a.rollout),
        "source_sha256": hashlib.sha256(Path(a.rollout).read_bytes()).hexdigest(),
        "probability_key": key,
        "steps_per_hour": counts.tolist(),
        "mean_rate_learned": overall,
        "mean_rate_table": from_table,
    }, indent=1))
    print(f"wrote {out}")
    for k in sub:
        print(f"  {k:8s} learned mean {overall[k]:.4f}  table mean {from_table[k]:.4f}")
    print("  heating rate by hour:", " ".join(f"{x:.2f}" for x in table[:, 9:14].mean(axis=1)))


if __name__ == "__main__":
    main()
