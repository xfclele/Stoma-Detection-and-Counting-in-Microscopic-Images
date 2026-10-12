#!/usr/bin/env python
"""对比两个模型的整图评估结果（eval_human.py 的输出目录），重点看失焦目标的召回。
Compare whole-image evaluations of two models (eval_human.py output directories), focusing on recall of
defocused objects.

    python scripts/compare_runs.py outputs/eval_human/v2_b_op 0.40 outputs/eval_human/v3a_last 0.40

输出：
    1. 总体 P/R/F1 和计数误差，以及逐图配对 Wilcoxon 检验（F1、|相对计数误差|）；
    2. 按清晰度四分位（与 error_stratify_logit.py 相同的频带能量比）和尺寸档统计召回差，
       给出按图像整体重抽样的 bootstrap 95% CI，以照顾同一张图内目标的相关性。
Outputs:
    1. overall P/R/F1 and count error, plus per-image paired Wilcoxon tests (F1, |relative count error|);
    2. recall differences by sharpness quartile (same band-energy ratio as error_stratify_logit.py) and
       size bin, with bootstrap 95% CIs that resample whole images to respect within-image correlation.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.error_analysis import match  # noqa: E402
from scripts.error_stratify_logit import SIZE_EDGES, band_maps, object_features  # noqa: E402
from scripts.eval_human import evaluate, load_gt  # noqa: E402


def per_object_hits(gt, preds, conf, tol_factor):
    out = {}
    for s, (ip, ann) in gt.items():
        P = preds[s]
        P = P[P[:, 4] >= conf] if len(P) else P
        C = np.stack([(P[:, 0] + P[:, 2]) / 2, (P[:, 1] + P[:, 3]) / 2], 1) if len(P) else np.zeros((0, 2))
        out[s] = match(C, ann.points, tol_factor * float(np.median(ann.majors)))[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("conf_a", type=float)
    ap.add_argument("run_b")
    ap.add_argument("conf_b", type=float)
    ap.add_argument("--root", default="data/samples/pre-trainning/Stomata_Enhanced/val")
    ap.add_argument("--exclude", nargs="*", default=["118.2", "221.2"])
    ap.add_argument("--tol-factor", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("-o", "--out", default=None, help="把报告写到该 .md 文件 / write the report to this .md file")
    a = ap.parse_args()

    gt = {k: v for k, v in load_gt(a.root).items() if k not in a.exclude}
    za, zb = np.load(os.path.join(a.run_a, "preds.npz")), np.load(os.path.join(a.run_b, "preds.npz"))
    pa, pb = {k: za[k] for k in gt}, {k: zb[k] for k in gt}
    ra, sa = evaluate(gt, pa, a.conf_a, a.tol_factor)
    rb, sb = evaluate(gt, pb, a.conf_b, a.tol_factor)
    na, nb = os.path.basename(a.run_a.rstrip("/")), os.path.basename(a.run_b.rstrip("/"))

    L = [f"# 模型对比 / Model comparison: {na}（conf {a.conf_a}） vs {nb}（conf {a.conf_b}）", "",
         f"{len(gt)} 张验证图 / validation images.", "",
         "| 模型 Model | P | R | F1 | 计数 Count MAE | 计数 Count MAPE | 计数偏差 Count bias |", "|---|---|---|---|---|---|---|"]
    for n, s in ((na, sa), (nb, sb)):
        L.append(f"| {n} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} | {s['count_mae']} | "
                 f"{s['count_mape']}% | {s['count_bias']:+} |")
    fa, fb = np.array([r["f1"] for r in ra]), np.array([r["f1"] for r in rb])
    ea, eb = np.abs([r["rel_count_err"] for r in ra]), np.abs([r["rel_count_err"] for r in rb])
    L += ["", f"- 逐图 F1 / per-image F1：{nb} 更好 better {int((fb > fa).sum())} / 更差 worse {int((fb < fa).sum())} / "
              f"持平 tied {int((fb == fa).sum())}，"
              f"Wilcoxon p = {stats.wilcoxon(fb, fa).pvalue:.3g}",
          f"- 逐图 |相对计数误差| / per-image |relative count error|：{nb} 更小 smaller {int((eb < ea).sum())} / "
          f"更大 larger {int((eb > ea).sum())}，"
          f"Wilcoxon p = {stats.wilcoxon(eb, ea).pvalue:.3g}"]

    # 逐目标：清晰度四分位 / 尺寸档 / per object: sharpness quartile / size bin
    ha, hb = per_object_hits(gt, pa, a.conf_a, a.tol_factor), per_object_hits(gt, pb, a.conf_b, a.tol_factor)
    img_of, lsharp, size, hA, hB = [], [], [], [], []
    for i, (s, (ip, ann)) in enumerate(gt.items()):
        f2, c2, c = band_maps(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2GRAY))
        sh, _ = object_features(f2, c2, c, ann.points, ann.majors)
        img_of += [i] * len(sh)
        lsharp += list(np.log(sh))
        size += list(ann.majors)
        hA += list(ha[s])
        hB += list(hb[s])
    img_of, lsharp, size = np.array(img_of), np.array(lsharp), np.array(size)
    hA, hB = np.array(hA, float), np.array(hB, float)
    q = np.digitize(lsharp, np.quantile(lsharp, [0.25, 0.5, 0.75]))
    sb_ = np.digitize(size, SIZE_EDGES)
    rng = np.random.default_rng(0)
    n_img = len(gt)
    boots = [rng.integers(0, n_img, n_img) for _ in range(a.boot)]
    idx_by_img = [np.nonzero(img_of == i)[0] for i in range(n_img)]

    def row(name, sel):
        d = hB[sel].mean() - hA[sel].mean()
        ds = []
        for bi in boots:
            ii = np.concatenate([idx_by_img[i] for i in bi])
            ii = ii[sel[ii]]
            if len(ii):
                ds.append(hB[ii].mean() - hA[ii].mean())
        lo, hi = np.percentile(ds, [2.5, 97.5])
        return f"| {name} | {int(sel.sum())} | {hA[sel].mean():.3f} | {hB[sel].mean():.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] |"

    head = ["| 分组 Group | n | " + na + " 召回 recall | " + nb + " 召回 recall | 差值 Diff | 95% CI（按图 bootstrap / image bootstrap） |", "|---|---|---|---|---|---|"]
    L += ["", "## 按清晰度四分位的召回 / Recall by sharpness quartile", ""] + head
    for k, lab in enumerate(["Q1 最模糊 blurriest", "Q2", "Q3", "Q4 最清晰 sharpest"]):
        L.append(row(lab, q == k))
    size_lab = [f"<{SIZE_EDGES[0]}"] + [f"{SIZE_EDGES[i]}–{SIZE_EDGES[i + 1]}" for i in range(len(SIZE_EDGES) - 1)] + [f"≥{SIZE_EDGES[-1]}"]
    L += ["", "## 按尺寸档的召回 / Recall by size bin", ""] + head
    for k, lab in enumerate(size_lab):
        L.append(row(lab, sb_ == k))
    L += ["", "## Q1 最模糊 × 尺寸 / Q1 blurriest × size", ""] + head
    for k, lab in enumerate(size_lab):
        L.append(row(f"Q1, {lab}", (q == 0) & (sb_ == k)))

    text = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
