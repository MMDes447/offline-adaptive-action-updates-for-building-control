#!/usr/bin/env python3
"""Component ablation and training-length results for our agent (beta3).

    python aamas2027/scripts/make_components.py [--main-epoch 100] [--ref-seed 20260728]

Reads aamas2027/runs/<name>/rollout.csv (status "ok" in manifest.json) produced
by run_batch3_components.sh. For selector seed 20260728 it falls back to legacy
rollouts whose checkpoint was verified at the weight level (LEGACY below), so a
first version of the table exists before batch 3 finishes. Legacy rows are
marked with a dagger in the table.

Writes to aamas2027/paper/:
  tab_components.tex        one row per variant, mean over selector seeds
  numbers_components.tex    macros for the text (\\CmpPlainE, \\CmpDeltaAugvE, ...)
  fig_epochs.pdf/.png       beta3 training length: energy, thermal, visual vs epoch
  components_summary.json   everything, machine-readable (per seed and aggregated)
  components_per_seed.csv   one row per (variant, seed)

Variance note: seeds differ only in the Bernoulli draws of the update selector
at evaluation time. They measure evaluation stochasticity of ONE trained
policy, not training variability; say so in the caption.
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


def run_roots():
    """Primary run tree plus isolated batch run trees, in precedence order."""
    return [RUNS, *sorted(AAMAS.glob("batch*_runs/*/runs"))]

# key, run-name prefix, table label, V input, policy weighting, critic
VARIANTS = [
    ("plain", "plain_e100", r"Plain (no aug.\ $V$, std.\ adv.)", r"$\rho_t$", r"$e^{\tilde A/T}$", "single"),
    ("augv", "augv_e100", r"+ augmented value", r"$\rho_t,a_{t-1}$", r"$e^{\tilde A/T}$", "single"),
    ("b3", "b3e100", r"+ standard IQL weight $e^{\beta A}$ (\textbf{ours})", r"$\rho_t,a_{t-1}$", r"$e^{\beta A}$", "single"),
    ("v11e100", "v11_e100", r"Decomposed critic, ep.\ 100", r"$\rho_t,a_{t-1}$", r"$e^{\tilde A/T}$", "task+SDAR"),
    ("v11e250", "v11_e250", r"Decomposed critic, ep.\ 250", r"$\rho_t,a_{t-1}$", r"$e^{\tilde A/T}$", "task+SDAR"),
    # batch 6: ours without observation history (GRU window K = 1); omitted until its runs exist
    ("k1", "k1_e100", r"Ours without history ($K{=}1$)", r"$\rho_t,a_{t-1}$", r"$e^{\beta A}$", "single"),
]
FULL = "b3"
# compact single-column table labels
SHORT = {"plain": r"Plain ($V(\rho_t)$, standardised adv.)", "augv": r"+ augmented value $V(\rho_t,a_{t-1})$",
         "b3": r"+ standard IQL weight (\textbf{ours})", "v11e100": r"Decomposed critic, 100 ep.",
         "v11e250": r"Decomposed critic, 250 ep.", "k1": r"Ours without history ($K{=}1$)"}
OPTIONAL = {"k1"}   # variants left out of the table/macros until at least one run exists

# Verified at the weight level (teacher-forced selector logits, mean |diff| < 1e-2):
#   augv e100 NPZ -> augmented_value epoch 100 (0.0016), beta3 e100 differs by 1.42
#   v11 e250 NPZ  -> v11 epoch 250 (0.0017), v11 epoch 100 differs by 1.40
#   beta3 e100    -> see CSV_CATALOG.md (0.004)
LEGACY = {
    "b3e100_sample_s20260728": "agent_eval_dataset_decomposed_epoch100_thermal3_tclip50_selector-sample_propdet_s20260728_beta3.csv",
    "b3e150_sample_s20260728": "agent_eval_dataset_decomposed_epoch150_thermal3_tclip50_selector-sample_propdet_s20260728_beta3_150.csv",
    "augv_e100_sample_s20260728": "agent_eval_dataset_decomposed_epoch100_thermal3_tclip50_selector-sample_propdet_s20260728_augmented_value.csv",
    "v11_e250_sample_s20260728": "agent_eval_dataset_decomposed_epoch250_thermal3_tclip50_selector-sample_propdet_s20260728.csv",
}
SWEEP = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
COLS = ["combined_kwh_day", "hvac_kwh_day", "lighting_kwh_day", "tv_pct", "dzh_day",
        "vis_in_pct", "glare_pct", "upd_pct", "persistence_steps"]


def run_csv(name):
    for root in run_roots():
        m = root / name / "manifest.json"
        p = root / name / "rollout.csv"
        if not p.is_file():
            p = root / name / "rollout.npz"  # NPZ fallback (same KPIs, see kpis_npz.py)
        if p.is_file() and m.is_file():
            man = json.loads(m.read_text())
            prov = man.get("provenance", {}) or {}
            if man.get("status") == "ok" and prov.get("match", True):
                return p, "new" if root == RUNS else "new-isolated"
    if name in LEGACY and (ROOT / LEGACY[name]).is_file():
        return ROOT / LEGACY[name], "legacy-verified"
    return None, None


_CACHE: dict = {}


def load(path):
    if path not in _CACHE:
        if str(path).endswith(".npz"):
            import kpis_npz
            df = kpis_npz.frame(path)
        else:
            df = K.load(path)
        _CACHE[path] = (K.annual(df), K.daily(df))
    return _CACHE[path]


def discover_seeds(prefix):
    seeds = set()
    for root in run_roots():
        if not root.is_dir():
            continue
        for d in root.glob(f"{prefix}_sample_s*"):
            tail = d.name.rsplit("_s", 1)[-1]
            if tail.isdigit():
                seeds.add(int(tail))
    for k in LEGACY:
        if k.startswith(f"{prefix}_sample_s"):
            seeds.add(int(k.rsplit("_s", 1)[-1]))
    return sorted(seeds)


def paired_delta(dailies_a, dailies_b, col, seed=0):
    """Variant minus full model, paired by day; 30-day moving-block bootstrap.

    Each side is first averaged over all of its selector seeds (seeds only change the
    evaluation-time Bernoulli draws, so seed numbers carry no pairing across models).
    The mean difference therefore equals the difference of the table rows.
    """
    if not dailies_a or not dailies_b:
        return None
    a = np.nanmean(np.stack([d[col].to_numpy(float) for d in dailies_a.values()]), axis=0)
    b = np.nanmean(np.stack([d[col].to_numpy(float) for d in dailies_b.values()]), axis=0)
    m, lo, hi = K.block_bootstrap_mean(a - b, block=30, seed=seed)
    return {"mean": m, "lo": lo, "hi": hi, "n_seeds_a": len(dailies_a), "n_seeds_b": len(dailies_b)}


def f(v, nd=1):
    return r"\textcolor{red}{TBD}" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def macro(name, val, nd=1):
    return f"\\newcommand{{\\{name}}}{{{f(val, nd)}}}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-epoch", type=int, default=100)
    ap.add_argument("--ref-seed", type=int, default=20260728)
    ap.add_argument("--seeds", nargs="+", type=int,
                    help="selector seeds to include in the matched component table")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, agg, dailies, missing = [], {}, {}, []
    for key, prefix, *_ in VARIANTS:
        seeds = a.seeds if a.seeds is not None else (discover_seeds(prefix) or [a.ref_seed])
        dailies[key] = {}
        for s in seeds:
            name = f"{prefix}_sample_s{s}"
            p, src = run_csv(name)
            if p is None:
                missing.append(name)
                continue
            ann, day = load(p)
            dailies[key][s] = day
            rows.append({"variant": key, "seed": s, "source": src, "file": str(p), **ann})
    per_seed = pd.DataFrame(rows)
    if not per_seed.empty:
        per_seed.to_csv(OUT / "components_per_seed.csv", index=False)

    for key, *_ in VARIANTS:
        sub = per_seed[per_seed.variant == key] if not per_seed.empty else pd.DataFrame()
        d = {"n": int(len(sub)), "legacy": bool((sub.source == "legacy-verified").any()) if len(sub) else False}
        for c in COLS:
            v = sub[c].to_numpy(float) if len(sub) and c in sub else np.array([])
            d[c] = float(v.mean()) if len(v) else None
            d[c + "_min"] = float(v.min()) if len(v) else None
            d[c + "_max"] = float(v.max()) if len(v) else None
        for col in ("combined_kwh", "tv_pct", "dzh"):
            d["delta_" + col] = (paired_delta(dailies[key], dailies[FULL], col)
                                 if key != FULL and dailies.get(FULL) else None)
        agg[key] = d

    rbc = None
    if (ROOT / RBC).is_file():
        rbc = load(ROOT / RBC)[0]

    # ---------------- table ----------------
    def cell(d, c, nd=1):
        v = d.get(c)
        if v is None:
            return f(None)
        s = f(v, nd)
        if d["n"] > 1 and d.get(c + "_max") is not None:
            spread = (d[c + "_max"] - d[c + "_min"]) / 2
            s += r"{\scriptsize$\pm$" + f(spread, nd) + "}"
        return s

    def dcell(d, col, nd=1):
        x = d.get("delta_" + col)
        if not x:
            return "--"
        sig = x["lo"] > 0 or x["hi"] < 0
        body = f"{x['mean']:+.{nd}f}" + r" {\scriptsize[" + f"{x['lo']:+.{nd}f}, {x['hi']:+.{nd}f}" + "]}"
        return body if sig else r"\textit{" + body + "}"

    lines = [
        r"\begin{table}[t]",
        r"\caption{Design-component ablation (100 training epochs unless stated); each row changes one component. "
        r"Same data, reward, encoder and evaluation protocol. Means over selector seeds ($\pm$ half-range; $n$ seeds), "
        r"which differ only in the evaluation-time draws of the update selector. Paired differences to our agent are given in the text.}",
        r"\label{tab:components}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Variant & kWh/d & Viol.\,\% & Kzh/d & Vis.\,\% & Upd.\,\% \\",
        r"\midrule",
    ]
    for key, _p, label, vin, w, critic in VARIANTS:
        d = agg[key]
        if key in OPTIONAL:          # reported in the supplementary history table, not here
            continue
        lab = SHORT[key] + (r"$^\dagger$" if d["legacy"] else "") + (f" {{\\scriptsize($n$={d['n']})}}" if d["n"] else "")
        if key in ("v11e100", "k1"):
            lines.append(r"\addlinespace")
        lines.append(" & ".join([
            lab, cell(d, "combined_kwh_day"), cell(d, "tv_pct", 2), cell(d, "dzh_day", 2),
            cell(d, "vis_in_pct"), cell(d, "upd_pct"),
        ]) + r" \\")
    if rbc:
        lines.append(r"\addlinespace")
        lines.append(r"Behaviour controller & " + " & ".join([f(rbc["combined_kwh_day"]), f(rbc["tv_pct"], 2), f(rbc["dzh_day"], 2),
                                                             f(rbc["vis_in_pct"]), f(rbc.get("upd_pct"))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "tab_components.tex").write_text("\n".join(lines) + "\n")

    # ---------------- supplementary: observation-history ablation ----------------
    if agg.get("k1", {}).get("n"):
        def dfmt(x, nd):
            return "--" if not x else f"{x['mean']:+.{nd}f} [{x['lo']:+.{nd}f}, {x['hi']:+.{nd}f}]"
        H = [r"\begin{table}[t]",
             r"\caption{Observation-history ablation: our agent with the 36-step history ($K{=}36$, reported checkpoint) "
             r"and the same method trained with $K{=}1$ (the GRU sees only the current observation; one training run). "
             r"Means over selector seeds ($\pm$ half-range). $\Delta$: $K{=}1$ minus $K{=}36$, paired by day, "
             r"95\% moving-block bootstrap interval (30-day blocks).}",
             r"\label{tab:history}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"History & kWh/d & Viol.\,\% & Kzh/d & Vis.\,\% & Upd.\,\% \\", r"\midrule"]
        for key, lab in (("b3", r"$K{=}36$ (ours)"), ("k1", r"$K{=}1$")):
            d = agg[key]
            H.append(" & ".join([lab + f" {{\\scriptsize($n$={d['n']})}}", cell(d, "combined_kwh_day"), cell(d, "tv_pct", 2),
                                 cell(d, "dzh_day", 2), cell(d, "vis_in_pct"), cell(d, "upd_pct")]) + r" \\")
        k = agg["k1"]
        H.append(r"\addlinespace")
        H.append(r"$\Delta$ & " + dfmt(k.get("delta_combined_kwh"), 1) + " & " + dfmt(k.get("delta_tv_pct"), 2)
                 + " & " + dfmt(k.get("delta_dzh"), 2) + r" & & \\")
        H += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        (OUT / "tab_history.tex").write_text("\n".join(H) + "\n")

    # ---------------- macros ----------------
    mac = "% generated by make_components.py; do not edit\n"
    names = {"plain": "Plain", "augv": "Augv", "b3": "Full", "v11e100": "DecA", "v11e250": "DecB", "k1": "KOne"}
    for key, nm in names.items():
        d = agg[key]
        if key in OPTIONAL and d["n"] == 0:
            continue
        mac += macro(f"Cmp{nm}E", d["combined_kwh_day"])
        mac += macro(f"Cmp{nm}Tv", d["tv_pct"], 2)
        mac += macro(f"Cmp{nm}Dzh", d["dzh_day"], 2)
        mac += macro(f"Cmp{nm}Vis", d["vis_in_pct"])
        mac += macro(f"Cmp{nm}Upd", d["upd_pct"])
        mac += f"\\newcommand{{\\Cmp{nm}N}}{{{d['n']}}}\n"
        for col, tag, nd in (("combined_kwh", "E", 1), ("tv_pct", "Tv", 2), ("dzh", "Dzh", 2)):
            x = d.get("delta_" + col)
            mac += macro(f"CmpDelta{nm}{tag}", x["mean"] if x else None, nd)
            mac += macro(f"CmpDelta{nm}{tag}Lo", x["lo"] if x else None, nd)
            mac += macro(f"CmpDelta{nm}{tag}Hi", x["hi"] if x else None, nd)

    # ---------------- training-length sweep ----------------
    sweep = []
    for e in SWEEP:
        p, src = run_csv(f"b3e{e}_sample_s{a.ref_seed}")
        if p is None:
            missing.append(f"b3e{e}_sample_s{a.ref_seed}")
            continue
        sweep.append({"epoch": e, "source": src, **load(p)[0]})
    sw = pd.DataFrame(sweep)
    if len(sw):
        best_e = int(sw.loc[sw.combined_kwh_day.idxmin(), "epoch"])
        mac += f"\\newcommand{{\\SweepN}}{{{len(sw)}}}\n"
        mac += macro("SweepEMin", sw.combined_kwh_day.min()) + macro("SweepEMax", sw.combined_kwh_day.max())
        mac += macro("SweepTvMin", sw.tv_pct.min(), 2) + macro("SweepTvMax", sw.tv_pct.max(), 2)
        mac += macro("SweepVisMin", sw.vis_in_pct.min()) + macro("SweepVisMax", sw.vis_in_pct.max())
        mac += f"\\newcommand{{\\SweepBestEpochE}}{{{best_e}}}\n"
        _plot_sweep(sw, a.main_epoch, rbc)
    (OUT / "numbers_components.tex").write_text(mac)

    summary = {"variants": agg, "sweep": sweep, "rbc": rbc, "missing_runs": sorted(set(missing)),
               "note": "seed spread = evaluation-time selector sampling only; one training run per variant"}
    (OUT / "components_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    def pf(v, nd=1):
        return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"

    print(f"{'variant':10s} {'n':>2s} {'kWh/d':>7s} {'TV%':>6s} {'Kzh/d':>6s} {'vis%':>6s} {'upd%':>6s}  dE vs full [95% CI]")
    for key, *_ in VARIANTS:
        d = agg[key]
        x = d.get("delta_combined_kwh")
        dd = f"{x['mean']:+6.1f} [{x['lo']:+.1f},{x['hi']:+.1f}]" if x else ""
        print(f"{key:10s} {d['n']:2d} {pf(d['combined_kwh_day']):>7s} {pf(d['tv_pct'],2):>6s} "
              f"{pf(d['dzh_day'],2):>6s} {pf(d['vis_in_pct']):>6s} {pf(d['upd_pct']):>6s}  {dd}")
    if len(sw):
        print("\nsweep:\n" + sw[["epoch", "source", "combined_kwh_day", "tv_pct", "dzh_day", "vis_in_pct", "upd_pct"]]
              .round(2).to_string(index=False))
    if missing:
        print(f"\n{len(set(missing))} runs not available yet (shown as TBD): " + ", ".join(sorted(set(missing))))


def _plot_sweep(sw, main_epoch, rbc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})

    # sized for one column (3.33 in) so that text prints at 6-7 pt without scaling
    panels = [("combined_kwh_day", "Electricity (kWh/d)"),
              ("tv_pct", "Thermal viol. (%)"),
              ("vis_in_pct", "Visual in-band (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(3.33, 1.45), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.04)
    for ax, (c, lab) in zip(axes, panels):
        ax.plot(sw.epoch, sw[c], marker="o", ms=2.5, lw=1.0, color="#2f6db3")
        if rbc and c in rbc:
            ax.axhline(rbc[c], color="#d9822b", lw=0.9, ls="--", label="RBC")
        if main_epoch in set(sw.epoch):
            v = float(sw.loc[sw.epoch == main_epoch, c].iloc[0])
            ax.plot([main_epoch], [v], marker="o", ms=5.5, mfc="none", mec="#1a1a1a", mew=0.9)
        ax.set_xlabel("Epoch", fontsize=6.5, labelpad=1)
        ax.set_title(lab, fontsize=6.5, pad=2)
        ax.set_xticks([100, 300, 500])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.grid(alpha=0.25, lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False, handlelength=1.5, borderaxespad=0.1)
    fig.savefig(OUT / "fig_epochs.pdf")
    fig.savefig(OUT / "fig_epochs.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
