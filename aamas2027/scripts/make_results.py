#!/usr/bin/env python3
"""Build every number, table and figure of the AAMAS results section.

    python aamas2027/scripts/make_results.py [--main-epoch 100]

Inputs are read from the project root:

* the clean RBC episode and the 13 exploratory data episodes;
* agent rollouts in aamas2027/runs/<name>/rollout.csv, created by run_eval.py;
* for runs not redone yet, the verified legacy rollout files (see LEGACY below).

Outputs go to aamas2027/paper/:

* numbers.tex: one \\newcommand per number. Missing inputs appear as a red TBD.
* tab_main.tex and tab_ablation.tex: booktabs tables.
* fig_tradeoff.pdf: energy vs thermal comfort and energy vs visual comfort.
* results_summary.json: all values, machine-readable.

Rerun the script after new runs finish. The paper then picks up the new numbers
without any hand editing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(os.path.abspath(__file__)).parent
AAMAS = HERE.parent
ROOT = AAMAS.parent
sys.path.insert(0, str(HERE / "sdar_eval"))
import kpis as K  # noqa: E402

RUNS = AAMAS / "runs"
OUT = AAMAS / "paper"

RBC = "offline_smooth_clean_episode_000.csv"
EPISODES = [RBC] + sorted(
    str(p.name) for p in ROOT.glob("offline_smooth_*_episode_0*.csv") if p.name != RBC
)
# Legacy rollouts whose checkpoint was verified at the weight level
# (see aamas2027/CSV_CATALOG.md).
LEGACY = {
    "b3e100_sample_s20260728": "agent_eval_dataset_decomposed_epoch100_thermal3_tclip50_selector-sample_propdet_s20260728_beta3.csv",
    "b3e150_sample_s20260728": "agent_eval_dataset_decomposed_epoch150_thermal3_tclip50_selector-sample_propdet_s20260728_beta3_150.csv",
}
SEEDS = [20260728, 20260729, 20260730, 20260731, 20260732]


def run_roots():
    """aamas2027/runs plus every isolated batch root (batch2_runs/*/runs, batch3_runs/*/runs)."""
    roots = [RUNS] + sorted(AAMAS.glob("batch*_runs/*/runs"))
    return [r for r in roots if r.is_dir()]


def run_csv(name):
    """Return (rollout path, kind). Prefers rollout.csv; falls back to rollout.npz
    (identical KPIs, see sdar_eval/kpis_npz.py). Only runs with status ok and a
    passing provenance check are used."""
    for root in run_roots():
        m = root / name / "manifest.json"
        if not m.is_file():
            continue
        man = json.loads(m.read_text())
        if man.get("status") != "ok" or not (man.get("provenance") or {}).get("match", False):
            continue
        for fn in ("rollout.csv", "rollout.npz"):
            if (root / name / fn).is_file():
                return root / name / fn, "new"
    if name in LEGACY and (ROOT / LEGACY[name]).is_file():
        return ROOT / LEGACY[name], "legacy-verified"
    return None, None


def load_any(path):
    path = Path(path)
    if path.suffix == ".npz":
        import kpis_npz
        return kpis_npz.frame(path)
    return K.load(path)


def mean_reward(path):
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        return float(np.mean(d["rewards"])) if "rewards" in d.files else None
    return None


def fmt(v, nd=1):
    return r"\textcolor{red}{TBD}" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def macro_name(s):
    return "".join(ch for ch in s.title() if ch.isalpha())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-epoch", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    E = a.main_epoch
    main_name = f"b3e{E}_sample_s20260728"

    res = {"sources": {}}
    dfs = {}
    ann = {}

    def get(name, path, src):
        dfs[name] = load_any(path)
        res["sources"][name] = {"path": str(path), "kind": src}
        out = K.annual(dfs[name])
        r = mean_reward(path) if src != "dataset" else None
        if r is None and src != "dataset" and Path(path).with_suffix(".npz").is_file():
            r = mean_reward(Path(path).with_suffix(".npz"))
        if r is not None:
            out["reward_mean"] = r
        ann[name] = out
        return out

    # data episodes
    ep = {}
    for f in EPISODES:
        ep[f] = get(f, ROOT / f, "dataset")
    rbc = ep[RBC]
    res["rbc"] = rbc
    res["episodes"] = ep

    # learned policies
    agents = {}
    for e in (100, 150, 500):
        n = f"b3e{e}_sample_s20260728"
        p, src = run_csv(n)
        if p is not None:
            agents[e] = get(n, p, src)
    res["agents"] = agents
    main = agents.get(E)

    # selector ablation, all on the main checkpoint
    abl = {}
    for mode in ("threshold", "always"):
        p, src = run_csv(f"b3e{E}_{mode}")
        if p is not None:
            abl[mode] = [get(f"b3e{E}_{mode}", p, src)]
    for mode in ("constant", "periodic", "clock"):     # clock: batch 7, hour-of-day selector
        vals = []
        for s in SEEDS:
            p, src = run_csv(f"b3e{E}_{mode}_s{s}")
            if p is not None:
                vals.append(get(f"b3e{E}_{mode}_s{s}", p, src))
        if vals:
            abl[mode] = vals
    samp = []
    for s in SEEDS:
        n = main_name if s == 20260728 else f"b3e{E}_sample_s{s}"
        p, src = run_csv(n)
        if p is not None:
            samp.append(ann[n] if n in ann else get(n, p, src))
    if samp:
        abl["sample"] = samp
        # headline = mean over all available selector seeds of the main checkpoint
        keys = set.intersection(*[set(v) for v in samp])
        main = {k: float(np.mean([v[k] for v in samp])) for k in keys}
        main.update({f"{k}_sd": float(np.std([v[k] for v in samp], ddof=1)) if len(samp) > 1 else 0.0 for k in keys})
        main["n_seeds"] = len(samp)
        agents[E] = main
        res["agents"] = agents
    p, src = run_csv(f"b3e{E}_sample_s20260728_sgoff")
    if p is not None:
        res["safeguard_off"] = get(f"b3e{E}_sgoff", p, src)
    res["ablation"] = abl

    # paired daily differences, agent minus RBC (same weather calendar)
    if main is not None:
        seed_names = [n for n in ([main_name] + [f"b3e{E}_sample_s{s}" for s in SEEDS[1:]]) if n in dfs]
        dr = K.daily(dfs[RBC])
        dms = [K.daily(dfs[n]) for n in seed_names]
        res["paired"] = {k: K.block_bootstrap_mean(np.nanmean([(dm[k] - dr[k]).to_numpy() for dm in dms], axis=0), block=30)
                         for k in ("combined_kwh", "tv_pct", "dzh")}
        res["paired_seeds"] = seed_names
        dominated_by = [f for f, v in ep.items()
                        if v["combined_kwh_day"] <= main["combined_kwh_day"] and v["tv_pct"] <= main["tv_pct"]]
        res["dominated_by"] = dominated_by
        res["dominates"] = [f for f, v in ep.items()
                            if v["combined_kwh_day"] >= main["combined_kwh_day"] and v["tv_pct"] >= main["tv_pct"]]

    # ---------------- numbers.tex ----------------
    L = [f"% generated by make_results.py (main epoch {E}); do not edit by hand"]
    def nc(name, val, nd=1):
        L.append(f"\\newcommand{{\\{name}}}{{{fmt(val, nd)}}}")
    get_ = lambda d, k: None if d is None else d.get(k)
    for tag, d in (("Rbc", rbc), ("Ag", main)):
        for k, nm in (("combined_kwh_day", "E"), ("hvac_kwh_day", "Hvac"), ("lighting_kwh_day", "Light"),
                      ("tv_pct", "Tv"), ("cold_pct", "Cold"), ("hot_pct", "Hot"), ("dzh_day", "Dzh"),
                      ("vis_in_pct", "Vis"), ("glare_pct", "Glare"), ("core_in_pct", "Core"),
                      ("upd_pct", "Upd"), ("persistence_steps", "Persist"),
                      ("glz_clear_pct", "GlzClear"), ("glz_dark_pct", "GlzDark"), ("glz_clear_sunny_pct", "GlzClearSun")):
            nc(f"{tag}{nm}", get_(d, k), 2 if nm in ("Dzh", "Persist") else 1)
    if main is not None:
        for k, nm in (("combined_kwh_day", "E"), ("tv_pct", "Tv"), ("dzh_day", "Dzh"), ("vis_in_pct", "Vis"), ("upd_pct", "Upd")):
            nc(f"Ag{nm}Sd", main.get(f"{k}_sd"), 2)
        nc("AgNSeeds", float(main.get("n_seeds", 1)), 0)
        rel = lambda k: 100 * (rbc[k] - main[k]) / rbc[k]
        nc("RedE", rel("combined_kwh_day")); nc("RedHvac", rel("hvac_kwh_day")); nc("RedLight", rel("lighting_kwh_day"))
        nc("RedTv", rel("tv_pct")); nc("RedDzh", rel("dzh_day")); nc("RedUpd", rel("upd_pct"))
        nc("DropTvPp", rbc["tv_pct"] - main["tv_pct"]); nc("DropVisPp", rbc["vis_in_pct"] - main["vis_in_pct"])
        for k, nm in (("combined_kwh", "E"), ("tv_pct", "Tv"), ("dzh", "Dzh")):
            m, lo, hi = res["paired"][k]
            nc(f"Pair{nm}", m, 2 if nm == "Dzh" else 1); nc(f"Pair{nm}Lo", lo, 2 if nm == "Dzh" else 1); nc(f"Pair{nm}Hi", hi, 2 if nm == "Dzh" else 1)
        nc("NDominated", float(len(res["dominates"])), 0)
        nc("NDominatedBy", float(len(res["dominated_by"])), 0)
    words = {100: "Hundred", 150: "OneFifty", 500: "FiveHundred"}
    for e in (100, 150, 500):
        d = agents.get(e)
        for k, nm in (("combined_kwh_day", "E"), ("tv_pct", "Tv"), ("dzh_day", "Dzh"), ("vis_in_pct", "Vis"), ("upd_pct", "Upd")):
            nc(f"Ep{words[e]}{nm}", None if d is None else d.get(k), 2 if nm == "Dzh" else 1)
    sg = res.get("safeguard_off")
    cl = abl.get("clock") or []
    if cl:   # batch 7: hour-of-day (clock) selector, mean over selector seeds
        for k, nm, nd in (("combined_kwh_day", "E", 1), ("tv_pct", "Tv", 1), ("dzh_day", "Dzh", 2),
                          ("vis_in_pct", "Vis", 1), ("upd_pct", "Upd", 1)):
            nc(f"Clock{nm}", float(np.mean([v[k] for v in cl])), nd)
        nc("ClockN", float(len(cl)), 0)
    nc("SgOffUnocc", None if sg is None else sg["unocc_light_kwh_day"], 1)
    nc("SgOffE", None if sg is None else sg["combined_kwh_day"], 1)
    epE = [v["combined_kwh_day"] for v in ep.values()]
    epT = [v["tv_pct"] for v in ep.values()]
    nc("DataEMin", min(epE)); nc("DataEMax", max(epE)); nc("DataTvMin", min(epT)); nc("DataTvMax", max(epT))
    (OUT / "numbers.tex").write_text("\n".join(L) + "\n")

    # ---------------- main table ----------------
    rows = [("Lighting (kWh/day)", "lighting_kwh_day", 1), ("HVAC (kWh/day)", "hvac_kwh_day", 1),
            ("Combined (kWh/day)", "combined_kwh_day", 1), None,
            ("Thermal violations (\\%)", "tv_pct", 1), ("\\quad cold / hot (\\%)", ("cold_pct", "hot_pct"), 1),
            ("Severity (K\\,zone\\,h/day)", "dzh_day", 2), None,
            ("Perimeter visual in band (\\%)", "vis_in_pct", 1), ("Over-illum. $>$1000\\,lx (\\%)", "glare_pct", 1),
            ("Core visual in band (\\%)", "core_in_pct", 1), None,
            ("Update rate (\\%)", "upd_pct", 1), ("\\quad glazing / lighting (\\%)", ("upd_glazing_pct", "upd_lighting_pct"), 1),
            ("\\quad heating / cooling (\\%)", ("upd_heating_pct", "upd_cooling_pct"), 1),
            ("Persistence (steps/update)", "persistence_steps", 2)]
    lo = {k: min(v[k] for v in ep.values() if k in v) for k in rbc}
    hi = {k: max(v[k] for v in ep.values() if k in v) for k in rbc}
    def cell(d, k, nd):
        if d is None:
            return r"\textcolor{red}{TBD}"
        if isinstance(k, tuple):
            return " / ".join(fmt(d.get(x), nd) for x in k)
        return fmt(d.get(k), nd)
    T = [r"\begin{table}[t]", r"\caption{Annual performance in the evaluation year. Data range: min--max over all 14 data episodes (clean and exploratory) of the rule-based controller that generated the dataset. Ours: reported training run, mean over its selector seeds (mean over five training runs in Table~\ref{tab:baselines}).}",
         r"\label{tab:main}", r"\small", r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Metric & Clean RBC & Data range & Ours \\", r"\midrule"]
    for r in rows:
        if r is None:
            T.append(r"\addlinespace[2pt]"); continue
        lab, k, nd = r
        rng = (" / ".join(f"{lo[x]:.{nd}f}--{hi[x]:.{nd}f}" for x in k) if isinstance(k, tuple)
               else f"{lo[k]:.{nd}f}--{hi[k]:.{nd}f}")
        T.append(f"{lab} & {cell(rbc, k, nd)} & {rng} & {cell(main, k, nd)} \\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_main.tex").write_text("\n".join(T) + "\n")

    # ---------------- ablation table ----------------
    def agg(vals, k, nd):
        if not vals:
            return r"\textcolor{red}{TBD}"
        x = np.array([v[k] for v in vals])
        return fmt(x.mean(), nd) + (f"$\\pm${x.std(ddof=1):.{nd}f}" if len(x) > 1 else "")
    A = [r"\begin{table}[t]", r"\caption{Update-selector ablation at inference time. The checkpoint, proposal policy, safeguard and weather are identical in every row. Stochastic selectors: mean$\pm$SD over five selector seeds. Random and periodic selectors use the learned policy's mean per-subsystem update probabilities; the time-of-day selector its mean probability per hour of day and dimension. Reward: mean training reward per step (higher is better). Bottom row: the behaviour controller, for reference.}",
         r"\label{tab:ablation}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Selector & Energy & Viol. & Sev. & Visual & Upd. & Reward \\", r" & kWh/d & \% & Kzh/d & \% & \% & /step \\", r"\midrule"]
    for key, lab in (("sample", "Learned (sampled)"), ("threshold", "Learned ($p\\geq0.5$)"),
                     ("constant", "Random, rate-matched"), ("periodic", "Periodic, rate-matched"),
                     ("clock", "Time of day, rate-matched"), ("always", "Always update")):
        v = abl.get(key, [])
        if key == "clock" and not v:
            continue
        A.append(f"{lab} & {agg(v,'combined_kwh_day',1)} & {agg(v,'tv_pct',1)} & {agg(v,'dzh_day',2)} & {agg(v,'vis_in_pct',1)} & {agg(v,'upd_pct',1)} & {agg(v,'reward_mean',3) if all('reward_mean' in x for x in v) else '--'} \\\\")
    A += [r"\addlinespace", f"Behaviour controller & {fmt(rbc['combined_kwh_day'])} & {fmt(rbc['tv_pct'])} & {fmt(rbc['dzh_day'], 2)} & {fmt(rbc['vis_in_pct'])} & {fmt(rbc['upd_pct'])} & -- \\\\",
          r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(A) + "\n")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    GRAY, ORANGE, BLUE, AQUA, INK2 = "#9a9893", "#eb6834", "#2a78d6", "#1baf7a", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.75), constrained_layout=True)
    for ax, (yk, ylab) in zip(axes, (("tv_pct", "Thermal violations (%)"), ("vis_in_pct", "Visual comfort in band (%)"))):
        xs = [v["combined_kwh_day"] for f, v in ep.items() if f != RBC]
        ys = [v[yk] for f, v in ep.items() if f != RBC]
        ax.scatter(xs, ys, s=12, marker="o", facecolor=GRAY, edgecolor="white", linewidth=0.6, label="Data episodes", zorder=2)
        ax.scatter([rbc["combined_kwh_day"]], [rbc[yk]], s=26, marker="D", facecolor=ORANGE, edgecolor="white", linewidth=0.8, label="Clean RBC", zorder=3)
        ax.annotate("RBC", (rbc["combined_kwh_day"], rbc[yk]), xytext=(4, 3), textcoords="offset points", color=INK2, fontsize=6)
        for e, mk in ((100, "s"), (150, "^"), (500, "v")):
            if e in agents:
                d = agents[e]
                ax.scatter([d["combined_kwh_day"]], [d[yk]], s=30 if e == E else 20, marker=mk,
                           facecolor=BLUE, edgecolor="white", linewidth=0.8, zorder=4,
                           label="Ours" if e == E else None)
                off = {100: (5, -8), 150: (-19, 1), 500: (5, -3)}[e] if yk == "tv_pct" else {100: (5, -8), 150: (-19, 1), 500: (5, 0)}[e]
                ax.annotate(f"e{e}", (d["combined_kwh_day"], d[yk]), xytext=off,
                            textcoords="offset points", color=INK2, fontsize=6)
        for key, mk, lab in (("constant", "P", "rand."), ("periodic", "X", "per."), ("always", "*", "always")):
            if key in abl:
                vv = abl[key]
                x = np.mean([v["combined_kwh_day"] for v in vv]); y = np.mean([v[yk] for v in vv])
                ax.scatter([x], [y], s=40 if mk == "*" else 24, marker=mk, facecolor=AQUA, edgecolor="white", linewidth=0.5, zorder=4)
                ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points", color=INK2, fontsize=6)
        ax.set_xlabel("Electricity (kWh/day)")
        ax.margins(x=0.08, y=0.12)
        ax.set_ylabel(ylab)
        ax.grid(True, color="#e6e5e1", linewidth=0.4, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=5.5, frameon=False, handletextpad=0.2, borderaxespad=0.1)
    fig.savefig(OUT / "fig_tradeoff.pdf")
    fig.savefig(OUT / "fig_tradeoff.png", dpi=220)

    (OUT / "results_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print("wrote", sorted(p.name for p in OUT.iterdir()))
    for n, s in res["sources"].items():
        if "episode" not in n:
            print(f"  {n:40s} <- {s['kind']:16s} {Path(s['path']).name}")


if __name__ == "__main__":
    main()
