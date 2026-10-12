#!/usr/bin/env python
"""漏检原因分类：判断"两个气孔被合成一个框"等后处理问题到底有多常见。

    python scripts/fn_causes.py outputs/eval_human/v2_b_op --conf 0.40

对每个漏检（FN）的人工真值中心：
    low       容差内有预测框，但分数低于 conf（模型找到了但不够确信）
    adjacent  容差内没有，但 1 个长轴距离内有一个已匹配的预测框（粘连 / 合并的上限）
    merged    adjacent 的子集：中心落在一个已匹配框内，且该框包含 ≥2 个真值中心（"二合一"）
    none      附近没有任何候选
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.error_analysis import match  # noqa: E402
from scripts.eval_human import load_gt  # noqa: E402


def centers(P):
    return np.stack([(P[:, 0] + P[:, 2]) / 2, (P[:, 1] + P[:, 3]) / 2], 1) if len(P) else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="eval_human.py 输出目录（含 preds.npz，建议推理阈值 ≤ 0.05）")
    ap.add_argument("--root", default="data/samples/pre-trainning/Stomata_Enhanced/val")
    ap.add_argument("--exclude", nargs="*", default=["118.2", "221.2"])
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--tol-factor", type=float, default=0.5)
    a = ap.parse_args()
    gt = {k: v for k, v in load_gt(a.root).items() if k not in a.exclude}
    z = np.load(os.path.join(a.run, "preds.npz"))

    cnt = dict(low=0, adjacent=0, none=0, merged=0)
    dense = dict(low=0, adjacent=0, none=0)
    merged_rel = []
    for s, (ip, ann) in gt.items():
        A = z[s]
        P = A[A[:, 4] >= a.conf]
        C, CA, G = centers(P), centers(A), ann.points
        med = float(np.median(ann.majors))
        tol = a.tol_factor * med
        mp, mg = match(C, G, tol)
        nb = (np.hypot(G[:, None, 0] - G[None, :, 0], G[:, None, 1] - G[None, :, 1]) < 100).sum(1) - 1
        for i in np.nonzero(~mg)[0]:
            d_any = np.hypot(*(CA - G[i]).T).min() if len(CA) else np.inf
            d_m = np.hypot(*(C[mp] - G[i]).T).min() if mp.any() else np.inf
            k = "low" if d_any <= tol else ("adjacent" if d_m <= med else "none")
            cnt[k] += 1
            if nb[i] >= 5:
                dense[k] += 1
            x, y = G[i]
            inside = (P[:, 0] <= x) & (x <= P[:, 2]) & (P[:, 1] <= y) & (y <= P[:, 3]) & mp
            for j in np.nonzero(inside)[0]:
                n_in = ((P[j, 0] <= G[:, 0]) & (G[:, 0] <= P[j, 2]) & (P[j, 1] <= G[:, 1]) & (G[:, 1] <= P[j, 3])).sum()
                if n_in >= 2:
                    cnt["merged"] += 1
                    merged_rel.append(max(P[j, 2] - P[j, 0], P[j, 3] - P[j, 1]) / med)
                    break

    n = cnt["low"] + cnt["adjacent"] + cnt["none"]
    print(f"漏检 {n} 个（conf {a.conf}，{len(gt)} 张图）")
    for k in ("low", "adjacent", "merged", "none"):
        print(f"  {k:9s} {cnt[k]:5d}  {cnt[k] / n:6.1%}")
    print(f"密集区（100 px 内 ≥5 邻居）漏检：{dense}")
    if merged_rel:
        r = np.array(merged_rel)
        print(f"二合一框长边 / 本图中位长轴：中位 {np.median(r):.2f}，>1.3 倍占 {(r > 1.3).mean():.0%}")


if __name__ == "__main__":
    main()
