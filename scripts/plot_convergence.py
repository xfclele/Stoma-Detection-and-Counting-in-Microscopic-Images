#!/usr/bin/env python
"""把若干依次衔接的训练段 (results.csv) 拼成连续的收敛曲线。

    python scripts/plot_convergence.py runs/stomata/human_ft1/results.csv runs/stomata/human_ft2/results.csv \
        -o docs/images/convergence_human.png
"""
from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"          # 参考调色板 categorical slot 1/2
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load(paths):
    rows, ends = [], []
    for p in paths:
        r = list(csv.DictReader(open(p)))
        rows += r
        ends.append(len(rows))
    return rows, ends[:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    rows, seams = load(a.csvs)
    ep = list(range(1, len(rows) + 1))
    g = lambda k: [float(r[k]) for r in rows]  # noqa: E731

    panels = [
        ("box loss", [("train", g("train/box_loss"), BLUE), ("val", g("val/box_loss"), ORANGE)]),
        ("cls loss", [("train", g("train/cls_loss"), BLUE), ("val", g("val/cls_loss"), ORANGE)]),
        ("val mAP50", [("", g("metrics/mAP50(B)"), BLUE)]),
        ("val mAP50-95", [("", g("metrics/mAP50-95(B)"), BLUE)]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), facecolor=SURF)
    for ax, (title, series) in zip(axes, panels):
        ax.set_facecolor(SURF)
        for name, ys, col in series:
            ax.plot(ep, ys, color=col, lw=2, marker="o", ms=4, label=name or None)
            ax.annotate(f"{ys[-1]:.3f}", (ep[-1], ys[-1]), textcoords="offset points", xytext=(6, 0),
                        color=INK2, fontsize=9, va="center")
        for s in seams:
            ax.axvline(s + 0.5, color=INK2, lw=1, ls=":")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("epoch", color=INK2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=INK2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if len(series) > 1:
            ax.legend(frameon=False, labelcolor=INK2)
    fig.suptitle("Fine-tuning on human labels (dotted line = training run boundary / restart)", color=INK2,
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, facecolor=SURF)
    print(a.out)


if __name__ == "__main__":
    main()
