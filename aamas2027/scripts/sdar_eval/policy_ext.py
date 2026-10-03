"""Two additional evaluation selector modes for the AAMAS 2027 batch-7 experiments,
built from the project's own ``sdar_iql_train_4_updated.load_and_act`` by exact
text substitutions (each must match exactly once; the module itself is not edited).

  clock        state-independent, time-of-day Bernoulli selector. The update
               probability of every action dimension depends only on the hour of
               day, taken from a 24 x 19 table (``hourly_update_rates``) that is the
               learned selector's mean update probability per hour and dimension in
               a reference rollout. Same per-dimension, per-hour budget as the
               learned selector, but blind to everything except the clock.

  always_prev  every dimension is updated at every step (like ``always``), but the
               proposal policy receives the previously executed action instead of
               the "-2" update markers. Used for the full-action IQL baseline whose
               actor was trained with the previous action as input
               (train_baselines.py --algo iql_flat_prev).

Every other mode is untouched: the patched function is behaviourally identical to
the original for threshold/sample/constant/periodic/always.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap

import sdar_iql_train_4_updated as _base

NEW_MODES = ("clock", "always_prev")

_SUBSTITUTIONS = [
    # 1. accept the two new modes
    ('    if selector_mode not in {"threshold", "sample", "constant", "periodic", "always"}:',
     '    if selector_mode not in {"threshold", "sample", "constant", "periodic", "always", "clock", "always_prev"}:'),
    # 2. new keyword argument (hour-of-day x dimension table)
    ('    periodic_phase_seed: int | None = None,\n',
     '    periodic_phase_seed: int | None = None,\n    hourly_update_rates=None,\n'),
    # 3. build the hourly table once
    ('    fixed_rate_vector = torch.from_numpy(fixed_rate_np).unsqueeze(0).to(DEVICE)\n',
     '    fixed_rate_vector = torch.from_numpy(fixed_rate_np).unsqueeze(0).to(DEVICE)\n'
     '    hourly_table = None\n'
     '    if selector_mode == "clock":\n'
     '        if hourly_update_rates is None:\n'
     '            raise ValueError("selector_mode=\'clock\' requires hourly_update_rates (24 x mask_dim).")\n'
     '        hourly_np = np.asarray(hourly_update_rates, dtype=np.float32)\n'
     '        if hourly_np.shape != (24, mask_dim) or not np.all(np.isfinite(hourly_np)) \\\n'
     '                or hourly_np.min() < 0.0 or hourly_np.max() > 1.0:\n'
     '            raise ValueError(f"hourly_update_rates must be a finite 24 x {mask_dim} table in [0,1].")\n'
     '        hourly_table = torch.from_numpy(hourly_np).to(DEVICE)\n'),
    # 4. active probabilities for the new modes
    ('            elif active_selector_mode == "always":\n'
     '                active_selection_probability = torch.ones(\n',
     '            elif active_selector_mode == "clock":\n'
     '                _tod = float(np.mod(np.arctan2(raw_obs[23], raw_obs[24]) / (2.0 * np.pi), 1.0))\n'
     '                _hour = min(int(_tod * 24.0 + 1e-6), 23)\n'
     '                active_selection_probability = hourly_table[_hour:_hour + 1]\n'
     '                active_selection_logits = torch.logit(\n'
     '                    active_selection_probability.clamp(1e-6, 1.0 - 1e-6)\n'
     '                )\n'
     '            elif active_selector_mode in {"always", "always_prev"}:\n'
     '                active_selection_probability = torch.ones(\n'),
    # 5. masks for the new modes
    ('            if force_first_update or active_selector_mode == "always":\n',
     '            if force_first_update or active_selector_mode in {"always", "always_prev"}:\n'),
    ('            elif active_selector_mode in {"sample", "constant"}:\n'
     '                mask = torch.bernoulli(\n',
     '            elif active_selector_mode in {"sample", "constant", "clock"}:\n'
     '                mask = torch.bernoulli(\n'),
    # 6. proposal input: previous action instead of markers for always_prev
    ('            action_mix = (1.0 - mask) * prev_action_batch + mask * ACTION_MASK_VALUE\n',
     '            if active_selector_mode == "always_prev":\n'
     '                action_mix = prev_action_batch.clone()\n'
     '            else:\n'
     '                action_mix = (1.0 - mask) * prev_action_batch + mask * ACTION_MASK_VALUE\n'),
    # 7. logging flag
    ('            "selector_stochastic": active_selector_mode in {"sample", "constant"},\n',
     '            "selector_stochastic": active_selector_mode in {"sample", "constant", "clock"},\n'),
]


def _build():
    src = textwrap.dedent(inspect.getsource(_base.load_and_act))
    original_sha = hashlib.sha256(src.encode()).hexdigest()
    for old, new in _SUBSTITUTIONS:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"policy_ext: expected exactly one match, found {n}: {old[:80]!r}")
        src = src.replace(old, new)
    ns = dict(vars(_base))
    exec(compile(src, "<policy_ext:load_and_act>", "exec"), ns)
    return ns["load_and_act"], original_sha, hashlib.sha256(src.encode()).hexdigest()


load_and_act, BASE_LOAD_AND_ACT_SHA256, PATCHED_LOAD_AND_ACT_SHA256 = _build()
BEST_CKPT_PATH = _base.BEST_CKPT_PATH
