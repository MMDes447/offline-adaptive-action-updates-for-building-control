#!/usr/bin/env python3
"""One-day figure: WHEN the learned selector updates, against the rate-matched periodic
selector (same checkpoint, same proposals, same per-subsystem update budget).

    python aamas2027/scripts/make_timing_figure.py [--doy 9]

Inputs (existing rollouts, no simulation):
  learned : runs/b3e100_sample_s20260728/rollout.npz
  periodic: batch2_runs/*/runs/b3e100_periodic_s20260728/rollout.npz
Writes paper/fig_timing.pdf/.png and prints the day's statistics.
The day is chosen for illustration (default: day of year 9, a clear January day);
annual results are in the selector-ablation table.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
AAMAS = HERE.parent
OUT = AAMAS / "paper"
SPD = 144                      # 10-min steps per day
T_LO, T_HI = 21.0, 24.0
SUBSYS = [("glazing", range(0, 4), "#2a78d6"), ("lighting", range(4, 9), "#eb6834"),
          ("setpoints", range(9, 14), "#1baf7a")]   # setpoints: heating OR cooling of a zone


def find(pattern):
    hits = sorted(glob.glob(str(AAMAS / pattern)))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def day_slice(doy):
    return slice((doy - 1) * SPD, doy * SPD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doy", type=int, default=9)
    ap.add_argument("--learned", default="runs/b3e100_sample_s20260728/rollout.npz")
    ap.add_argument("--periodic", default="batch2_runs/*/runs/b3e100_periodic_s20260728/rollout.npz")
    a = ap.parse_args()
    runs = {"Learned": np.load(find(a.learned), allow_pickle=True),
            "Periodic (rate-matched)": np.load(find(a.periodic), allow_pickle=True)}
    s = day_slice(a.doy)
    hours = np.arange(SPD) / 6.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "axes.edgecolor": "#55554f",
                         "xtick.color": "#55554f", "ytick.color": "#55554f"})
    fig = plt.figure(figsize=(3.4, 2.8))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.7, 1.15, 1.0, 1.0], hspace=0.38)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    any_run = next(iter(runs.values()))
    o0 = any_run["observations"][s]
    occ = o0[:, 27] > 0.5
    starts = np.flatnonzero(np.diff(np.r_[0, occ.astype(int)]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[occ.astype(int), 0]) == -1)
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        for i, j in zip(starts, ends):
            ax.axvspan(hours[i], hours[j] + 1 / 6, facecolor="#e6e5df", edgecolor="none", zorder=-1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # (a) exterior irradiance on the south facade (identical weather in both runs)
    axes[0].fill_between(hours, o0[:, 14], color="#b9b8b0", lw=0)
    axes[0].set_ylabel("South\nirr. (W/m²)", fontsize=6.5)
    axes[0].set_ylim(0, max(100, o0[:, 14].max() * 1.1))

    # (b) mean zone temperature for both selectors, comfort band
    axes[1].axhspan(T_LO, T_HI, facecolor="#cfe8dc", edgecolor="none", alpha=0.6, zorder=0)
    stats = {}
    for (name, d), style in zip(runs.items(), ({"color": "#1f1f1c", "ls": "-"}, {"color": "#8a897f", "ls": "--"})):
        T = d["observations"][s][:, :5]
        axes[1].plot(hours, T.mean(axis=1), lw=1.2, **style)
        t = T[occ]
        m = d["selection_masks"][s]
        stats[name] = {"viol_pct": 100 * ((t < T_LO) | (t > T_HI)).mean(),
                       "updates": {k: int(m[:, list(idx)].sum() + (m[:, [i + 5 for i in idx]].sum() if k == "setpoints" else 0))
                                   for k, idx, _ in SUBSYS}}
        y4 = T.mean(axis=1)[int(2.5 * 6)]
        axes[1].text(2.5, y4 + (0.35 if name == "Learned" else -0.45), name.split(" ")[0],
                     ha="center", va="bottom" if name == "Learned" else "top", fontsize=6, color=style["color"])
    axes[1].set_ylabel("Mean zone\ntemp. (°C)", fontsize=6.5)
    axes[1].text(23.7, T_HI - 0.15, "21–24 °C band", fontsize=5.5, color="#3d6b55", va="top", ha="right")

    # (c, d) update events per actuator
    for ax, (name, d) in zip(axes[2:], runs.items()):
        m = d["selection_masks"][s]
        row = 0
        yt, yl = [], []
        for k, idx, col in SUBSYS:
            for i in idx:
                ev = m[:, i] > 0.5
                if k == "setpoints":
                    ev |= m[:, i + 5] > 0.5
                ax.vlines(hours[ev], row + 0.1, row + 0.9, color=col, lw=0.7)
                row += 1
            yt.append(row - len(idx) / 2)
            yl.append(k)
            row += 0.6
        ax.set_ylim(-0.2, row - 0.4)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=6)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_title(f"{name}: updates", fontsize=6.5, loc="left", pad=1.5, color="#1f1f1c")
    axes[-1].set_xlabel("Hour of day (shaded: occupied)")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_timing.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_timing.png", dpi=220, bbox_inches="tight")
    print(f"day of year {a.doy}:")
    for name, st in stats.items():
        print(f"  {name:26s} occupied violations {st['viol_pct']:.1f}%  updates {st['updates']}")


if __name__ == "__main__":
    main()
