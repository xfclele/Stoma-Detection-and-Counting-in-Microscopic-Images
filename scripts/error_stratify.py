#!/usr/bin/env python
"""分层误差诊断：失焦（清晰度）是否在控制尺寸等因素后仍独立影响召回。

    python scripts/error_stratify.py outputs/eval_human/v2_b_op --conf 0.40

对每个人工真值气孔，在以其中心为圆心、边长 2 × 框长边的窗口内计算：
    · 清晰度 sharp = E[(G1−G2)²] / E[(G2−G8)²]
      细频带与粗频带能量之比。失焦会先衰减细频带，比值与对比度和亮度无关。
    · 对比度 contrast = std(G2−G8)，即粗频带幅度，用来区分"失焦"和"低对比度"。
输出：
    1. 尺寸 × 清晰度四分位的召回交叉表，按尺寸分层的清晰度效应，以及 Mantel–Haenszel 合并 OR；
    2. 多因素逻辑回归（尺寸、清晰度、对比度、密度、边缘），使用按图像聚类的稳健标准误；
    3. 图像固定效应模型：只用同一张图内部的清晰度差异，排除"整张图难"的混杂。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.error_analysis import match  # noqa: E402
from stomata.dl import load_yolo_polygon_labels  # noqa: E402

SIZE_EDGES = [40, 50, 60, 70]


def band_maps(gray):
    g = gray.astype(np.float32)
    g1, g2, g8 = (cv2.GaussianBlur(g, (0, 0), s) for s in (1, 2, 8))
    return (g1 - g2) ** 2, (g2 - g8) ** 2, g2 - g8


def object_features(fine2, coarse2, coarse, pts, majors):
    """用积分图快速计算每个气孔窗口内的频带能量。"""
    H, W = fine2.shape
    Ifine, Icoarse = cv2.integral(fine2.astype(np.float64)), cv2.integral(coarse2.astype(np.float64))
    Ic1 = cv2.integral(coarse.astype(np.float64))

    def box_sum(I, x0, y0, x1, y1):
        return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]

    sharp, contrast = [], []
    for (x, y), m in zip(pts, majors):
        r = max(int(round(m)), 8)
        x0, y0 = int(max(0, x - r)), int(max(0, y - r))
        x1, y1 = int(min(W, x + r)), int(min(H, y + r))
        n = max((x1 - x0) * (y1 - y0), 1)
        ef, ec = box_sum(Ifine, x0, y0, x1, y1) / n, box_sum(Icoarse, x0, y0, x1, y1) / n
        mc = box_sum(Ic1, x0, y0, x1, y1) / n
        sharp.append(ef / (ec + 1e-6))
        contrast.append(np.sqrt(max(ec - mc * mc, 0)))
    return np.array(sharp), np.array(contrast)


def fit_logit(X, y, groups=None, l2=1e-4):
    """带极小 L2 的逻辑回归；groups 给定时返回按组聚类的稳健（sandwich）标准误。"""
    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * b @ b

    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        return X.T @ (p - y) + l2 * b

    b = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x
    p = 1 / (1 + np.exp(-(X @ b)))
    Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
    Hinv = np.linalg.pinv(Hm)
    if groups is None:
        return b, np.sqrt(np.diag(Hinv))
    S = np.zeros_like(Hm)
    for gid in np.unique(groups):
        sel = groups == gid
        sg = X[sel].T @ (y[sel] - p[sel])
        S += np.outer(sg, sg)
    G = len(np.unique(groups))
    V = Hinv @ S @ Hinv * G / (G - 1)
    return b, np.sqrt(np.diag(V))


def mantel_haenszel(strata):
    """strata: [(a, b, c, d)]，其中 a、b 为暴露组的漏检/命中数，c、d 为对照组的漏检/命中数。
    返回合并 OR 和 CMH χ² 检验的 p 值。"""
    num = den = 0.0
    sa = ea = va = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
        r1, r0, c1 = a + b, c + d, a + c
        sa += a
        ea += r1 * c1 / n
        va += r1 * r0 * c1 * (n - c1) / (n * n * (n - 1))
    chi2 = (abs(sa - ea) - 0.5) ** 2 / va
    return num / den, stats.chi2.sf(chi2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="eval_human.py 的输出目录（含 preds.npz）")
    ap.add_argument("--root", default="data/samples/pre-trainning/Stomata_Enhanced/val")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--tol-factor", type=float, default=0.5)
    a = ap.parse_args()
    z = np.load(os.path.join(a.run, "preds.npz"))

    rows = []  # image, size, sharp, contrast, dens, edge, hit
    for s in z.files:
        ip = os.path.join(a.root, "images", s + ".jpg")
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        ann = load_yolo_polygon_labels(os.path.join(a.root, "labels", s + ".txt"), ip, W, H)
        P = z[s]
        P = P[P[:, 4] >= a.conf]
        PC = np.stack([(P[:, 0] + P[:, 2]) / 2, (P[:, 1] + P[:, 3]) / 2], 1) if len(P) else np.zeros((0, 2))
        G = ann.points
        _, mg = match(PC, G, a.tol_factor * float(np.median(ann.majors)))
        fine2, coarse2, coarse = band_maps(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        sharp, contrast = object_features(fine2, coarse2, coarse, G, ann.majors)
        dG = np.hypot(G[:, None, 0] - G[None, :, 0], G[:, None, 1] - G[None, :, 1])
        dens = (dG < 100).sum(1) - 1
        edge = np.minimum.reduce([G[:, 0], G[:, 1], W - 1 - G[:, 0], H - 1 - G[:, 1]])
        for i in range(len(G)):
            rows.append((s, ann.majors[i], sharp[i], contrast[i], dens[i], edge[i], mg[i]))

    img_id = np.array([r[0] for r in rows])
    size, sharp, contrast, dens, edge, hit = (np.array([r[k] for r in rows], float) for k in range(1, 7))
    miss = 1 - hit
    lsharp, lcon = np.log(sharp), np.log(contrast + 1e-6)
    sq = np.quantile(lsharp, [0.25, 0.5, 0.75])
    sharp_bin = np.digitize(lsharp, sq)  # 0 = 最模糊 … 3 = 最清晰
    size_bin = np.digitize(size, SIZE_EDGES)
    size_lab = [f"<{SIZE_EDGES[0]}"] + [f"{SIZE_EDGES[i]}–{SIZE_EDGES[i + 1]}" for i in range(len(SIZE_EDGES) - 1)] + [f"≥{SIZE_EDGES[-1]}"]
    q_lab = ["Q1 最模糊", "Q2", "Q3", "Q4 最清晰"]

    out = [f"# 尺寸 × 清晰度分层诊断：{a.run}", "",
           f"置信度阈值 {a.conf}，{len(np.unique(img_id))} 张图，{len(hit)} 个真值，总体召回 {hit.mean():.3f}。",
           f"清晰度 = 细频带/粗频带能量比（与对比度无关）；四分位界（log）：{', '.join(f'{v:.2f}' for v in sq)}。", "",
           f"清晰度与尺寸的 Spearman ρ = {stats.spearmanr(lsharp, size)[0]:.2f}；与对比度 ρ = {stats.spearmanr(lsharp, lcon)[0]:.2f}。", "",
           "## 1. 召回交叉表（单元格：召回 / n）", "",
           "| 尺寸 \\ 清晰度 | " + " | ".join(q_lab) + " | 合计 |", "|---" * 6 + "|"]
    for b, lab in enumerate(size_lab):
        cells = []
        for q in range(4):
            sel = (size_bin == b) & (sharp_bin == q)
            cells.append(f"{hit[sel].mean():.3f} / {sel.sum()}" if sel.sum() else "—")
        sel = size_bin == b
        out.append(f"| {lab} | " + " | ".join(cells) + f" | {hit[sel].mean():.3f} / {sel.sum()} |")
    cells = [f"{hit[sharp_bin == q].mean():.3f} / {(sharp_bin == q).sum()}" for q in range(4)]
    out.append("| 合计 | " + " | ".join(cells) + f" | {hit.mean():.3f} / {len(hit)} |")

    # 按尺寸分层：Q1（最模糊）对 Q2–Q4 的漏检 OR
    strata = []
    out += ["", "## 2. 按尺寸分层的清晰度效应（Q1 最模糊 vs Q2–Q4）", "",
            "| 尺寸 | Q1 召回 | Q2–Q4 召回 | 差值 | 漏检 OR |", "|---|---|---|---|---|"]
    for b, lab in enumerate(size_lab):
        e, c = (size_bin == b) & (sharp_bin == 0), (size_bin == b) & (sharp_bin > 0)
        t = (miss[e].sum(), hit[e].sum(), miss[c].sum(), hit[c].sum())
        strata.append(t)
        orr = (t[0] + .5) * (t[3] + .5) / ((t[1] + .5) * (t[2] + .5))
        out.append(f"| {lab} | {hit[e].mean():.3f} | {hit[c].mean():.3f} | {hit[e].mean() - hit[c].mean():+.3f} | {orr:.2f} |")
    or_mh, p_mh = mantel_haenszel(strata)
    out += ["", f"Mantel–Haenszel 合并漏检 OR（控制尺寸）= **{or_mh:.2f}**，CMH p = {p_mh:.1e}。",
            "注意：CMH 检验假定各目标相互独立，而同一张图的目标是相关的，所以这个 p 偏乐观，以下面的聚类稳健模型为准。"]

    # 多因素逻辑回归：因变量为漏检
    def z_(v):
        return (v - v.mean()) / v.std()
    size_d = np.stack([(size_bin == b).astype(float) for b in range(len(size_lab)) if b != 2], 1)  # 基准 50–60
    names = ["截距"] + [f"尺寸 {size_lab[b]}（vs 50–60）" for b in range(len(size_lab)) if b != 2] + \
            ["清晰度（每 1 SD，log）", "对比度（每 1 SD，log）", "密度（每 1 SD）", "距边缘 < 25 px"]
    X = np.column_stack([np.ones(len(hit)), size_d, z_(lsharp), z_(lcon), z_(dens), (edge < 25).astype(float)])
    b, se = fit_logit(X, miss, img_id)
    out += ["", "## 3. 多因素逻辑回归（因变量 = 漏检，聚类稳健 SE，按图像聚类）", "",
            "| 变量 | 漏检 OR | 95% CI | p |", "|---|---|---|---|"]
    for n, bi, si in zip(names, b, se):
        if n == "截距":
            continue
        out.append(f"| {n} | {np.exp(bi):.2f} | {np.exp(bi - 1.96 * si):.2f}–{np.exp(bi + 1.96 * si):.2f} | {2 * stats.norm.sf(abs(bi / si)):.1e} |")

    # 图像固定效应：只用图内的变化
    imgs = np.unique(img_id)
    FE = np.stack([(img_id == u).astype(float) for u in imgs], 1)
    Xfe = np.column_stack([FE, size_d, z_(lsharp), z_(lcon), z_(dens), (edge < 25).astype(float)])
    bf, sef = fit_logit(Xfe, miss, img_id, l2=1e-2)
    k = len(imgs)
    out += ["", "## 4. 图像固定效应模型（只用同一张图内部的差异）", "",
            "| 变量 | 漏检 OR | 95% CI | p |", "|---|---|---|---|"]
    for n, bi, si in zip(names[1:], bf[k:], sef[k:]):
        out.append(f"| {n} | {np.exp(bi):.2f} | {np.exp(bi - 1.96 * si):.2f}–{np.exp(bi + 1.96 * si):.2f} | {2 * stats.norm.sf(abs(bi / si)):.1e} |")

    # 图像层面：每图清晰度中位数 与 每图召回
    per = [(u, np.median(lsharp[img_id == u]), hit[img_id == u].mean(), (img_id == u).sum()) for u in imgs]
    rho, p_rho = stats.spearmanr([t[1] for t in per], [t[2] for t in per])
    out += ["", "## 5. 图像层面", "",
            f"每图清晰度中位数 与 每图召回：Spearman ρ = {rho:.2f}（p = {p_rho:.1e}，n = {len(per)}）。", "",
            "最模糊的 8 张图：", "", "| 图像 | 清晰度中位数（log） | 召回 | 真值数 |", "|---|---|---|---|"]
    for u, s_, r_, n_ in sorted(per, key=lambda t: t[1])[:8]:
        out.append(f"| {u} | {s_:.2f} | {r_:.3f} | {n_} |")

    text = "\n".join(out) + "\n"
    open(os.path.join(a.run, f"stratify_conf{a.conf}.md"), "w", encoding="utf-8").write(text)
    json.dump(dict(or_mh=or_mh, p_mh=p_mh, logit=dict(zip(names, map(float, b))),
                   fe=dict(zip(names[1:], map(float, bf[k:])))),
              open(os.path.join(a.run, f"stratify_conf{a.conf}.json"), "w"), ensure_ascii=False, indent=1)
    print(text)


if __name__ == "__main__":
    main()
