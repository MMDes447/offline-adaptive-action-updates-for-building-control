#!/usr/bin/env python3
"""Offline-baseline comparison table (batch 4).

    python aamas2027/scripts/make_baselines.py

Reads every finished run (status ok, provenance match) from aamas2027/runs and
aamas2027/batch*_runs/*/runs, as rollout.csv or rollout.npz:

  Ours      training seed 0 = the reported beta3 checkpoint: b3e100_sample_s<eval-seed>
            training seeds 1.. : sdar_iql_t<seed>_e100_sample_s<eval-seed>
  IQL (full action)   iql_flat_t<seed>_e100_sample_s*
  BC (full action)    bc_flat_t<seed>_e100_sample_s*
  BC (act/repeat)     bc_sdar_t<seed>_e100_sample_s*

Per method: mean over evaluation (selector) seeds within each training run, then
mean +- SD over training runs. Paired differences to ours use the day-by-day
average over all runs of each method and a 30-day moving-block bootstrap.

Writes paper/tab_baselines.tex, paper/numbers_baselines.tex,
paper/baselines_summary.json and prints the table.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
AAMAS = HERE.parent
ROOT = AAMAS.parent
sys.path.insert(0, str(HERE / "sdar_eval"))
import kpis as K  # noqa: E402

OUT = AAMAS / "paper"
RBC = "offline_smooth_clean_episode_000.csv"
METHODS = [  # key, label, run-name regex (group 1 = training seed or None)
    ("bc_flat", r"BC (full action)", r"^bc_flat_t(\d+)_e100_sample_s\d+$"),
    ("iql_flat", r"IQL (full action)", r"^iql_flat_t(\d+)_e100_sample_s\d+$"),
    # batch 7: full-action IQL whose actor also receives a_{t-1}; row omitted until its runs exist
    ("iql_flat_prev", r"IQL (full action) $+\,a_{t-1}$", r"^iql_flat_prev_t(\d+)_e100_always_prev$"),
    ("bc_sdar", r"BC (act/repeat)", r"^bc_sdar_t(\d+)_e100_sample_s\d+$"),
    ("sdar_iql", r"\textbf{Ours}", r"^(?:sdar_iql_t(\d+)_e100|b3e100)_sample_s\d+$"),
]
COLS = [("combined_kwh_day", 1), ("tv_pct", 1), ("dzh_day", 2), ("vis_in_pct", 1), ("upd_pct", 1), ("reward_mean", 3)]


def roots():
    return [r for r in [AAMAS / "runs"] + sorted(AAMAS.glob("batch*_runs/*/runs")) if r.is_dir()]


def finished_runs():
    seen = {}
    for root in roots():
        for d in sorted(root.iterdir()):
            m = d / "manifest.json"
            if not m.is_file():
                continue
            man = json.loads(m.read_text())
            if man.get("status") != "ok" or not (man.get("provenance") or {}).get("match", False):
                continue
            f = d / "rollout.csv" if (d / "rollout.csv").is_file() else d / "rollout.npz"
            if f.is_file() and d.name not in seen:
                seen[d.name] = f
    return seen


def load(path: Path):
    if path.suffix == ".npz":
        import kpis_npz
        df = kpis_npz.frame(path)
        rew = float(np.mean(np.load(path, allow_pickle=True)["rewards"]))
    else:
        df = K.load(path)
        npz = path.with_suffix(".npz")
        rew = float(np.mean(np.load(npz, allow_pickle=True)["rewards"])) if npz.is_file() else None
    ann = K.annual(df)
    ann["reward_mean"] = rew
    return ann, K.daily(df)


def f(v, nd=1):
    return r"\textcolor{red}{TBD}" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def main():
    runs = finished_runs()
    res, dailies = {}, {}
    for key, label, pat in METHODS:
        per_train = {}
        for name, path in runs.items():
            mt = re.match(pat, name)
            if not mt:
                continue
            ts = int(mt.group(1)) if mt.group(1) else 0
            ann, day = load(path)
            per_train.setdefault(ts, []).append(ann)
            dailies.setdefault(key, []).append(day)
        agg = {"label": label, "n_train": len(per_train), "n_eval": sum(len(v) for v in per_train.values()),
               "train_seeds": sorted(per_train)}
        for c, _ in COLS:
            means = [np.mean([a[c] for a in v if a.get(c) is not None]) for v in per_train.values()
                     if any(a.get(c) is not None for a in v)]
            agg[c] = float(np.mean(means)) if means else None
            agg[c + "_sd_train"] = float(np.std(means, ddof=1)) if len(means) > 1 else None
        res[key] = agg

    rbc_ann, rbc_day = load(ROOT / RBC)
    res["rbc"] = {"label": "Behaviour controller", **rbc_ann}

    # paired daily differences vs ours
    if dailies.get("sdar_iql"):
        ours = np.nanmean([d[["combined_kwh", "tv_pct", "dzh"]].to_numpy() for d in dailies["sdar_iql"]], axis=0)
        for key in ("bc_flat", "iql_flat", "iql_flat_prev", "bc_sdar"):
            if dailies.get(key):
                theirs = np.nanmean([d[["combined_kwh", "tv_pct", "dzh"]].to_numpy() for d in dailies[key]], axis=0)
                res[key]["delta_vs_ours"] = {
                    k: K.block_bootstrap_mean(theirs[:, i] - ours[:, i], block=30)
                    for i, k in enumerate(("combined_kwh", "tv_pct", "dzh"))}

    def cell(a, c, nd):
        v = a.get(c)
        s = f(v, nd)
        sd = a.get(c + "_sd_train")
        if v is not None and sd is not None:
            s += r"{\scriptsize$\pm$" + f"{sd:.{nd}f}" + "}"
        return s

    T = [r"\begin{table}[t]",
         r"\caption{Comparison with offline baselines trained on the same data with the same encoder, actor, "
         r"optimiser and number of updates (100 epochs; for our reported run, the 100-epoch checkpoint of a 500-epoch run), and evaluated with the same protocol. "
         r"Values: mean over evaluation seeds within each training run, then mean $\pm$ SD over training runs "
         r"($n_{\mathrm{tr}}$). Full-action methods update all 19 dimensions at every step.}",
         r"\label{tab:baselines}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Method & $n_{\mathrm{tr}}$ & Energy & Viol. & Sev. & Visual & Upd. & Reward \\",
         r" & & kWh/d & \% & Kzh/d & \% & \% & /step \\", r"\midrule"]
    for key, label, _ in METHODS:
        a = res[key]
        if key == "iql_flat_prev" and not a["n_train"]:
            continue
        T.append(f"{label} & {a['n_train'] or '--'} & " + " & ".join(cell(a, c, nd) for c, nd in COLS) + r" \\")
    r = res["rbc"]
    T += [r"\addlinespace", f"Behaviour controller & -- & {f(r['combined_kwh_day'])} & {f(r['tv_pct'])} & {f(r['dzh_day'], 2)} & "
          f"{f(r['vis_in_pct'])} & {f(r.get('upd_pct'))} & -- \\\\", r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_baselines.tex").write_text("\n".join(T) + "\n")

    L = ["% generated by make_baselines.py; do not edit"]
    names = {"bc_flat": "BcFlat", "iql_flat": "IqlFlat", "iql_flat_prev": "IqlPrev", "bc_sdar": "BcSdar", "sdar_iql": "Ours"}
    for key, nm in names.items():
        a = res[key]
        if key == "iql_flat_prev" and not a["n_train"]:
            continue
        for c, tag, nd in (("combined_kwh_day", "E", 1), ("tv_pct", "Tv", 1), ("dzh_day", "Dzh", 2),
                           ("vis_in_pct", "Vis", 1), ("upd_pct", "Upd", 1), ("reward_mean", "Rew", 3)):
            L.append(f"\\newcommand{{\\Base{nm}{tag}}}{{{f(a.get(c), nd)}}}")
            L.append(f"\\newcommand{{\\Base{nm}{tag}Sd}}{{{f(a.get(c + '_sd_train'), nd)}}}")
        L.append(f"\\newcommand{{\\Base{nm}Ntr}}{{{a['n_train']}}}")
        for k, tag, nd in (("combined_kwh", "E", 1), ("tv_pct", "Tv", 1), ("dzh", "Dzh", 2)):
            d = (a.get("delta_vs_ours") or {}).get(k)
            L.append(f"\\newcommand{{\\Base{nm}Delta{tag}}}{{{f(d[0] if d else None, nd)}}}")
            L.append(f"\\newcommand{{\\Base{nm}Delta{tag}Lo}}{{{f(d[1] if d else None, nd)}}}")
            L.append(f"\\newcommand{{\\Base{nm}Delta{tag}Hi}}{{{f(d[2] if d else None, nd)}}}")
    (OUT / "numbers_baselines.tex").write_text("\n".join(L) + "\n")
    (OUT / "baselines_summary.json").write_text(json.dumps(res, indent=2, default=float))

    print(f"{'method':22s} {'n_tr':>4s} {'n_ev':>4s} {'kWh/d':>8s} {'viol%':>7s} {'Kzh/d':>6s} {'vis%':>6s} {'upd%':>6s} {'reward':>8s}")
    for key, label, _ in METHODS + [("rbc", "RBC", None)]:
        a = res[key]
        print(f"{key:22s} {str(a.get('n_train', '-')):>4s} {str(a.get('n_eval', '-')):>4s} "
              + " ".join(f"{(a.get(c) if a.get(c) is not None else float('nan')):{w}.{nd}f}"
                         for (c, nd), w in zip(COLS, (8, 7, 6, 6, 6, 8))))


if __name__ == "__main__":
    main()
