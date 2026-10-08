#!/usr/bin/env python
"""误差分析：读取 eval_human.py 保存的 preds.npz，按目标属性分桶统计漏检/误检。

    python scripts/error_analysis.py outputs/eval_human/<run> --conf 0.4 [--gallery 40]

分桶维度（对真值 → 召回；对预测 → 精确率）：
    · 目标尺寸（框长边，px）
    · 局部密度（半径 100 px 内的真值数）
    · 到图像边缘的距离（px）
    · 预测置信度（仅预测）
另输出：每图计数误差排序、漏检/误检样例拼图 (fn_gallery.jpg / fp_gallery.jpg)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata.dl import load_yolo_polygon_labels  # noqa: E402


def match(P, G, tol):
    if not len(P) or not len(G):
        return np.zeros(len(P), bool), np.zeros(len(G), bool)
    D = np.hypot(P[:, None, 0] - G[None, :, 0], P[:, None, 1] - G[None, :, 1])
    ri, ci = linear_sum_assignment(np.where(D <= tol, D, 1e6))
    ok = D[ri, ci] <= tol
    mp, mg = np.zeros(len(P), bool), np.zeros(len(G), bool)
    mp[ri[ok]] = True
    mg[ci[ok]] = True
    return mp, mg


def bucket_table(values, hit, edges, name):
    lines = [f"| {name} | n | 命中率 |", "|---|---|---|"]
    idx = np.digitize(values, edges)
    labels = [f"<{edges[0]}"] + [f"{edges[i]}–{edges[i + 1]}" for i in range(len(edges) - 1)] + [f"≥{edges[-1]}"]
    for b, lab in enumerate(labels):
        sel = idx == b
        if sel.sum():
            lines.append(f"| {lab} | {int(sel.sum())} | {hit[sel].mean():.3f} |")
    return "\n".join(lines)


def crop(img, x, y, r=40):
    H, W = img.shape[:2]
    x0, y0 = int(max(0, x - r)), int(max(0, y - r))
    c = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
    c = cv2.copyMakeBorder(c, 0, 2 * r - c.shape[0], 0, 2 * r - c.shape[1], cv2.BORDER_CONSTANT)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--root", default="data/samples/pre-trainning/Stomata_Enhanced/val")
    ap.add_argument("--conf", type=float, default=None, help="默认取 summary.json 中 F1 最优阈值")
    ap.add_argument("--tol-factor", type=float, default=0.5)
    ap.add_argument("--gallery", type=int, default=40)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.run, "summary.json")))
    conf = a.conf if a.conf is not None else summ["best_f1"]["conf"]
    z = np.load(os.path.join(a.run, "preds.npz"))
    rng = np.random.default_rng(0)

    G_all = dict(size=[], dens=[], edge=[], hit=[])
    P_all = dict(size=[], conf=[], edge=[], hit=[])
    fn_samples, fp_samples, per = [], [], []
    for s in z.files:
        ip = os.path.join(a.root, "images", s + ".jpg")
        ann = load_yolo_polygon_labels(os.path.join(a.root, "labels", s + ".txt"), ip, 2560, 1920)
        P = z[s]
        P = P[P[:, 4] >= conf]
        PC = np.stack([(P[:, 0] + P[:, 2]) / 2, (P[:, 1] + P[:, 3]) / 2], 1) if len(P) else np.zeros((0, 2))
        G = ann.points
        tol = a.tol_factor * float(np.median(ann.majors))
        mp, mg = match(PC, G, tol)
        dG = np.hypot(G[:, None, 0] - G[None, :, 0], G[:, None, 1] - G[None, :, 1])
        G_all["size"] += list(ann.majors)
        G_all["dens"] += list((dG < 100).sum(1) - 1)
        G_all["edge"] += list(np.minimum.reduce([G[:, 0], G[:, 1], 2559 - G[:, 0], 1919 - G[:, 1]]))
        G_all["hit"] += list(mg)
        if len(P):
            P_all["size"] += list(np.maximum(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]))
            P_all["conf"] += list(P[:, 4])
            P_all["edge"] += list(np.minimum.reduce([PC[:, 0], PC[:, 1], 2559 - PC[:, 0], 1919 - PC[:, 1]]))
            P_all["hit"] += list(mp)
        per.append((s, len(G), len(PC), int((~mg).sum()), int((~mp).sum())))
        fn_samples += [(ip, *G[i]) for i in np.nonzero(~mg)[0]]
        fp_samples += [(ip, *PC[i], P[i, 4]) for i in np.nonzero(~mp)[0]]

    G_all = {k: np.array(v) for k, v in G_all.items()}
    P_all = {k: np.array(v) for k, v in P_all.items()}
    rep = [f"# 误差分析：{a.run}", "", f"置信度阈值 {conf}，匹配容差 {a.tol_factor} × 框长边。", "",
           f"总体召回 {G_all['hit'].mean():.3f}（{int((~G_all['hit'].astype(bool)).sum())} 个漏检 / {len(G_all['hit'])} 个真值），"
           f"精确率 {P_all['hit'].mean():.3f}（{int((~P_all['hit'].astype(bool)).sum())} 个误检 / {len(P_all['hit'])} 个预测）", "",
           "## 召回 vs 目标尺寸（真值框长边 px）", bucket_table(G_all["size"], G_all["hit"], [40, 50, 60, 70], "尺寸"), "",
           "## 召回 vs 局部密度（100 px 内邻居数）", bucket_table(G_all["dens"], G_all["hit"], [1, 3, 5, 7], "邻居数"), "",
           "## 召回 vs 到图像边缘距离（px）", bucket_table(G_all["edge"], G_all["hit"], [25, 50, 100], "边距"), "",
           "## 精确率 vs 预测置信度", bucket_table(P_all["conf"], P_all["hit"], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "置信度"), "",
           "## 精确率 vs 预测框尺寸（px）", bucket_table(P_all["size"], P_all["hit"], [40, 50, 60, 70], "尺寸"), "",
           "## 计数误差最大的 10 张图（预测 − 真值）", "| 图像 | 真值 | 预测 | 误差 | 漏检 | 误检 |", "|---|---|---|---|---|---|"]
    for s, g, p, fn, fp in sorted(per, key=lambda t: -abs(t[2] - t[1]))[:10]:
        rep.append(f"| {s} | {g} | {p} | {p - g:+d} | {fn} | {fp} |")
    open(os.path.join(a.run, f"error_analysis_conf{conf}.md"), "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print("\n".join(rep))

    for name, samples in (("fn", fn_samples), ("fp", fp_samples)):
        if not samples:
            continue
        pick = rng.choice(len(samples), min(a.gallery, len(samples)), replace=False)
        cache, tiles = {}, []
        for i in pick:
            ip, x, y = samples[i][:3]
            img = cache.setdefault(ip, cv2.imread(ip))
            c = cv2.resize(crop(img, x, y), (120, 120))
            cv2.circle(c, (60, 60), 4, (0, 0, 255) if name == "fp" else (0, 200, 0), -1)
            tiles.append(c)
        while len(tiles) % 10:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r * 10:(r + 1) * 10]) for r in range(len(tiles) // 10)])
        cv2.imwrite(os.path.join(a.run, f"{name}_gallery.jpg"), grid)


if __name__ == "__main__":
    main()
