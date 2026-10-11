#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/error_stratify.py —— 气孔漏检的「三轴解耦」诊断

目的
----
训练报告里 "<40 px 召回 0.73" 是唯一被量化的事实，但「失焦」「低对比度」「开口过大」「粘连」
这四类漏检目前只是目视归纳，而且它们与尺寸高度混杂：失焦会让气孔变大变糊，低对比度灰视野
里的气孔同时偏小。在花掉宝贵的 CPU 训练预算（每段约 1 小时）之前，必须先确认这些因子
在**同一尺寸档内**是否仍然独立影响召回。

判据（方案 §0）：
    只有当「同一尺寸档内，低清晰度组的召回显著低于高清晰度组」时，失焦才是独立因子，
    此时做离焦增强才有效；否则预算应该投到小目标分辨率上。

输出
----
    instances.csv        每个 GT 一行：坐标、长轴、开口度、清晰度、对比度、邻近度、是否命中
    predictions.csv      每个预测一行：是否被匹配上（TP / FP）
    stratify_report.md   交叉表 + Fisher 检验 + Mann-Whitney 检验 + Wilson 置信区间

用法
----
    # 首次：跑推理并缓存预测（CPU 慢）
    python scripts/error_stratify.py \
        --images data/samples/pre-trainning/Stomata_Enhanced/val/images \
        --labels data/samples/pre-trainning/Stomata_Enhanced/val/labels \
        --weights models/stomata_yolov8n_p2_human_v2/best.pt \
        --conf 0.40 --out outputs_dl/stratify --save-preds preds_val.json

    # 之后：复用缓存，秒级重跑全部交叉表
    python scripts/error_stratify.py ... --preds-json preds_val.json

    # 验证脚本本身是否测得准（不需要数据和 ultralytics）
    python scripts/error_stratify.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

# 允许直接从仓库任意位置运行（本文件放在 scripts/ 下）
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent, _HERE):
    if str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# 与训练报告一致的分档：<40 / 40-70 / >=70 px
DEFAULT_SIZE_BINS: List[Tuple[float, float, str]] = [
    (0.0, 40.0, "<40px"),
    (40.0, 70.0, "40-70px"),
    (70.0, float("inf"), ">=70px"),
]

METRICS = [
    # (列名, 中文名, "低值组"的含义)
    ("sharp", "局部清晰度(Laplacian方差)", "低=失焦"),
    ("contrast", "局部对比度", "低=灰视野"),
    ("aspect", "开口度(短轴/长轴)", "低=细缝"),
    ("prox", "邻近度(最近邻距/长轴)", "低=粘连"),
]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def imread_any(path: str):
    """支持中文/空格路径的读取。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """二项比例的 Wilson 置信区间，小样本下比 normal 近似可靠。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_p(p: float) -> str:
    if p != p:
        return "  n/a"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


# --------------------------------------------------------------------------- #
# 标签解析
# --------------------------------------------------------------------------- #
def parse_label_file(path: str, W: int, H: int):
    """
    读取 YOLO 标签，支持两种格式：
      cls cx cy w h              (5 列)
      cls x1 y1 x2 y2 x3 y3 x4 y4 (9 列, 归一化 4 角点)

    返回 (centers(N,2), major(N,), minor(N,))，单位均为像素。
    对 9 列格式，长/短轴由相邻角点距离的最大/最小值得到 —— 这是**旋转椭圆真实的**长短轴，
    而 load_yolo_polygon_labels() 取的是外接矩形，对旋转目标会系统性偏大，故此处自行解析。
    """
    centers, majors, minors = [], [], []
    if not path or not os.path.exists(path):
        z = np.zeros((0, 2), float)
        return z, np.zeros(0), np.zeros(0)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if not t:
                continue
            try:
                v = [float(x) for x in t]
            except ValueError:
                continue
            if len(v) >= 9:
                c = np.array(v[1:9], float).reshape(4, 2)
                c[:, 0] *= W
                c[:, 1] *= H
                d = [float(np.linalg.norm(c[i] - c[(i + 1) % 4])) for i in range(4)]
                centers.append(c.mean(axis=0))
                majors.append(max(d))
                minors.append(min(d))
            elif len(v) >= 5:
                _, cx, cy, w, h = v[:5]
                centers.append([cx * W, cy * H])
                majors.append(max(w * W, h * H))
                minors.append(min(w * W, h * H))
    if not centers:
        z = np.zeros((0, 2), float)
        return z, np.zeros(0), np.zeros(0)
    return (np.array(centers, float), np.array(majors, float), np.array(minors, float))


# --------------------------------------------------------------------------- #
# 图像度量
# --------------------------------------------------------------------------- #
def local_metrics(gray: np.ndarray, cx: float, cy: float, radius: float,
                  g_p95: float, g_p5: float) -> Tuple[float, float]:
    """
    窗口内清晰度（Laplacian 方差）与局部对比度（归一化到全图动态范围）。
    窗口边长 = 1.5 × 长轴，即 radius = 0.75 × 长轴。
    """
    H, W = gray.shape
    x0 = max(0, int(round(cx - radius)))
    x1 = min(W, int(round(cx + radius)) + 1)
    y0 = max(0, int(round(cy - radius)))
    y1 = min(H, int(round(cy + radius)) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return (float("nan"), float("nan"))
    win = gray[y0:y1, x0:x1]
    if win.size < 9:
        return (float("nan"), float("nan"))
    sharp = float(cv2.Laplacian(win, cv2.CV_32F).var())
    loc = float(np.percentile(win, 95) - np.percentile(win, 5))
    g_range = g_p95 - g_p5
    contrast = loc / g_range if g_range > 1e-6 else float("nan")
    return (sharp, contrast)


# --------------------------------------------------------------------------- #
# 匹配
# --------------------------------------------------------------------------- #
def match_image_mode(pred_xy: np.ndarray, gt_xy: np.ndarray, majors: np.ndarray):
    """
    模式 A（默认）：整图单一容差 tol = 0.35 × 该图中位长轴，复用已测过的
    stomata.evaluation.match_points（匈牙利一对一，重复检测计 FP）。
    与 eval_human.py 的口径一致，因此本脚本的总召回应当能复现报告的整图数字。
    """
    tol = float(np.clip(0.35 * np.median(majors), 6.0, None)) if len(majors) else 12.0
    try:
        from stomata.evaluation import match_points
    except Exception:
        # 未安装 stomata 包时的等价回退（同样是一对一匈牙利 + 固定容差）
        return _hungarian_fixed_tol(pred_xy, gt_xy, tol)

    res = match_points(pred_xy.tolist(), gt_xy.tolist(), tol=tol)
    hit = np.zeros(len(gt_xy), bool)
    pred_matched = np.zeros(len(pred_xy), bool)
    pred_dist = np.full(len(pred_xy), np.nan)
    for pi, gi, dist in res.matches:
        hit[gi] = True
        pred_matched[pi] = True
        pred_dist[pi] = dist
    return hit, pred_matched, pred_dist, tol


def _hungarian_fixed_tol(pred_xy: np.ndarray, gt_xy: np.ndarray, tol: float):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    hit = np.zeros(len(gt_xy), bool)
    pred_matched = np.zeros(len(pred_xy), bool)
    pred_dist = np.full(len(pred_xy), np.nan)
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return hit, pred_matched, pred_dist, tol
    D = cdist(pred_xy, gt_xy)
    BIG = 1e6
    rows, cols = linear_sum_assignment(np.where(D <= tol, D, BIG))
    for i, j in zip(rows, cols):
        if D[i, j] <= tol:
            hit[j] = True
            pred_matched[i] = True
            pred_dist[i] = D[i, j]
    return hit, pred_matched, pred_dist, tol


def match_gt_mode(pred_xy: np.ndarray, gt_xy: np.ndarray, majors: np.ndarray):
    """
    模式 B：逐目标尺度相关容差 tol_i = 0.35 × 长轴_i（下限 6 px）。
    自实现匈牙利指派；用于检查结论对容差口径是否稳健。
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    hit = np.zeros(len(gt_xy), bool)
    pred_matched = np.zeros(len(pred_xy), bool)
    pred_dist = np.full(len(pred_xy), np.nan)
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return hit, pred_matched, pred_dist, float("nan")
    D = cdist(pred_xy, gt_xy)
    tol = np.clip(0.35 * majors, 6.0, None)
    BIG = 1e6
    cost = np.where(D <= tol[None, :], D, BIG)
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols):
        if D[i, j] <= tol[j]:
            hit[j] = True
            pred_matched[i] = True
            pred_dist[i] = D[i, j]
    return hit, pred_matched, pred_dist, float(np.median(tol))


# --------------------------------------------------------------------------- #
# 分层与检验
# --------------------------------------------------------------------------- #
def median_split_table(vals: np.ndarray, hit: np.ndarray, bin_mask: np.ndarray,
                       min_bin_n: int):
    """在一个尺寸档内按中位数二分，返回 (低值组, 高值组) 的召回与 Fisher 检验。"""
    from scipy.stats import fisher_exact

    m = bin_mask & np.isfinite(vals)
    v, h = vals[m], hit[m]
    if len(v) < min_bin_n:
        return None
    thr = float(np.median(v))
    lo = v <= thr
    hi = ~lo
    if lo.sum() == 0 or hi.sum() == 0:
        return None
    k_lo, n_lo = int(h[lo].sum()), int(lo.sum())
    k_hi, n_hi = int(h[hi].sum()), int(hi.sum())
    # 2x2: 行=低/高值组, 列=命中/漏检
    table = [[k_lo, n_lo - k_lo], [k_hi, n_hi - k_hi]]
    _, p = fisher_exact(table)
    return {
        "thr": thr, "n_lo": n_lo, "rec_lo": k_lo / n_lo, "k_lo": k_lo,
        "n_hi": n_hi, "rec_hi": k_hi / n_hi, "k_hi": k_hi,
        "diff": k_lo / n_lo - k_hi / n_hi, "p": float(p),
    }


def mannwhitney_table(vals: np.ndarray, hit: np.ndarray):
    """不做分档，直接比较命中组与漏检组的分布（Mann-Whitney U，双侧）。"""
    from scipy.stats import mannwhitneyu

    a = vals[hit & np.isfinite(vals)]
    b = vals[(~hit) & np.isfinite(vals)]
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return None
    return {"n_hit": len(a), "med_hit": float(np.median(a)),
            "n_miss": len(b), "med_miss": float(np.median(b)), "p": float(p)}


# --------------------------------------------------------------------------- #
# 主分析
# --------------------------------------------------------------------------- #
def analyze(items: List[Tuple[str, str, str]], preds: Dict[str, list],
            out_dir: Path, size_bins, min_bin_n: int, tol_mode: str,
            top_images: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    prows: List[dict] = []
    per_image: List[dict] = []

    for stem, img_path, lbl_path in items:
        img = imread_any(img_path)
        if img is None:
            print(f"[warn] 无法读取图像: {img_path}")
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g_p95 = float(np.percentile(gray, 95))
        g_p5 = float(np.percentile(gray, 5))

        centers, majors, minors = parse_label_file(lbl_path, W, H)
        n_gt = len(centers)
        if n_gt == 0:
            continue

        # --- GT 侧度量 ---
        aspect = np.where(majors > 1e-6, minors / np.maximum(majors, 1e-6), np.nan)
        sharp = np.full(n_gt, np.nan)
        contrast = np.full(n_gt, np.nan)
        for i, (cx, cy) in enumerate(centers):
            s, c = local_metrics(gray, cx, cy, 0.75 * majors[i], g_p95, g_p5)
            sharp[i] = s
            contrast[i] = c

        # 邻近度：最近邻 GT 中心距 / 长轴
        if n_gt > 1:
            from scipy.spatial.distance import cdist
            D = cdist(centers, centers)
            np.fill_diagonal(D, np.inf)
            prox = D.min(axis=1) / np.maximum(majors, 1e-6)
        else:
            prox = np.full(n_gt, np.inf)

        med_sharp = float(np.nanmedian(sharp)) if np.isfinite(sharp).any() else float("nan")
        sharp_rel = sharp / med_sharp if med_sharp and med_sharp > 1e-9 else np.full(n_gt, np.nan)

        # --- 预测 ---
        p = np.array(preds.get(stem, []), float).reshape(-1, 5)
        if len(p):
            pxy = np.stack([(p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2], axis=1)
        else:
            pxy = np.zeros((0, 2), float)

        if tol_mode == "gt":
            hit, pm, pdist, tol_used = match_gt_mode(pxy, centers, majors)
        else:
            hit, pm, pdist, tol_used = match_image_mode(pxy, centers, majors)

        for i in range(n_gt):
            rows.append({
                "image": stem, "gt_id": i,
                "x": round(float(centers[i, 0]), 2), "y": round(float(centers[i, 1]), 2),
                "major": round(float(majors[i]), 2), "minor": round(float(minors[i]), 2),
                "aspect": round(float(aspect[i]), 4) if np.isfinite(aspect[i]) else "",
                "sharp": round(float(sharp[i]), 3) if np.isfinite(sharp[i]) else "",
                "sharp_rel": round(float(sharp_rel[i]), 4) if np.isfinite(sharp_rel[i]) else "",
                "contrast": round(float(contrast[i]), 4) if np.isfinite(contrast[i]) else "",
                "prox": round(float(prox[i]), 4) if np.isfinite(prox[i]) else "",
                "hit": int(hit[i]),
            })
        for j in range(len(p)):
            prows.append({
                "image": stem, "pred_id": j,
                "x": round(float(pxy[j, 0]), 2), "y": round(float(pxy[j, 1]), 2),
                "w": round(float(p[j, 2] - p[j, 0]), 2), "h": round(float(p[j, 3] - p[j, 1]), 2),
                "score": round(float(p[j, 4]), 4),
                "matched": int(pm[j]),
                "dist": round(float(pdist[j]), 2) if np.isfinite(pdist[j]) else "",
            })

        per_image.append({
            "image": stem, "n_gt": n_gt, "n_pred": len(p),
            "tp": int(hit.sum()), "recall": float(hit.mean()),
            "med_major": round(float(np.median(majors)), 1),
            "med_sharp": round(med_sharp, 2) if np.isfinite(med_sharp) else "",
        })

    if not rows:
        raise RuntimeError("没有解析到任何 GT，请检查 --images / --labels 路径与 --exclude。")

    # 汇总为数组
    def col(name, dtype=float):
        return np.array([(float(r[name]) if r[name] != "" else np.nan) for r in rows], dtype)

    major = col("major")
    hit = np.array([r["hit"] for r in rows], bool)
    vals = {
        "sharp": col("sharp"),
        "contrast": col("contrast"),
        "aspect": col("aspect"),
        "prox": col("prox"),
        "sharp_rel": col("sharp_rel"),
    }

    # ---------------- 写 CSV ----------------
    with open(out_dir / "instances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        if prows:
            w = csv.DictWriter(f, fieldnames=list(prows[0].keys()))
            w.writeheader()
            w.writerows(prows)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    # ---------------- 报表 ----------------
    L: List[str] = []
    A = L.append
    A("# 气孔漏检三轴解耦诊断\n")
    A(f"- GT 总数：**{len(rows)}**，命中 {int(hit.sum())}，整体召回 **{hit.mean():.3f}**")
    A(f"- 匹配容差模式：`{tol_mode}`")
    A("")
    A("> **怎么读这张表**：在每个尺寸档内按中位数二分。只有当同一尺寸档内，"
      "「差」的那一半召回**显著**低于「好」的那一半（Fisher p 小），"
      "该因子才算被证实为独立漏检原因，值得为它单独做增强。")
    A("> 多重比较：4 个指标 × 3 个尺寸档 = 12 次检验，Bonferroni 校正后阈值约 p < 0.0042。")
    A("")

    # 总览：按尺寸档
    A("## 1. 按尺寸档的召回（与训练报告口径对齐）\n")
    A("| 尺寸档 | n | 命中 | 召回 | 95% Wilson CI |")
    A("|---|---|---|---|---|")
    for lo, hi, name in size_bins:
        m = (major >= lo) & (major < hi)
        n, k = int(m.sum()), int(hit[m].sum())
        if n == 0:
            A(f"| {name} | 0 | 0 | - | - |")
            continue
        a, b = wilson_ci(k, n)
        A(f"| {name} | {n} | {k} | {k/n:.3f} | [{a:.3f}, {b:.3f}] |")
    A("")

    # 逐指标：分档内中位数二分 + Fisher
    A("## 2. 尺寸档内二分：因子是否独立影响召回\n")
    for key, cn, lo_note in METRICS:
        A(f"### {cn}　（{lo_note}）\n")
        A("| 尺寸档 | 低值组 召回 (n) | 高值组 召回 (n) | 差值 | Fisher p |")
        A("|---|---|---|---|---|")
        for blo, bhi, bname in size_bins:
            m = (major >= blo) & (major < bhi)
            r = median_split_table(vals[key], hit, m, min_bin_n)
            if r is None:
                A(f"| {bname} | — | — | — | 样本不足（<{min_bin_n}） |")
                continue
            A(f"| {bname} | {r['rec_lo']:.3f} ({r['n_lo']}) | {r['rec_hi']:.3f} ({r['n_hi']}) "
              f"| {r['diff']:+.3f} | {fmt_p(r['p'])} |")
        A("")

    # 不分档的 Mann-Whitney
    A("## 3. 不分档：命中组 vs 漏检组的分布差异（Mann-Whitney U，双侧）\n")
    A("| 指标 | 命中组中位数 | 漏检组中位数 | p |")
    A("|---|---|---|---|")
    for key, cn, _ in METRICS + [("sharp_rel", "相对清晰度(窗口/本图中位)", "低=局部失焦")]:
        r = mannwhitney_table(vals[key], hit)
        if r is None:
            A(f"| {cn} | — | — | 样本不足 |")
            continue
        A(f"| {cn} | {r['med_hit']:.3g} (n={r['n_hit']}) | {r['med_miss']:.3g} (n={r['n_miss']}) "
          f"| {fmt_p(r['p'])} |")
    A("")

    # 每图召回
    A(f"## 4. 召回最低的 {min(top_images, len(per_image))} 张图\n")
    A("| 图像 | GT 数 | 命中 | 召回 | 中位长轴 | 中位清晰度 |")
    A("|---|---|---|---|---|---|")
    for r in sorted(per_image, key=lambda r: r["recall"])[:top_images]:
        A(f"| {r['image']} | {r['n_gt']} | {r['tp']} | {r['recall']:.3f} "
          f"| {r['med_major']} | {r['med_sharp']} |")
    A("")

    (out_dir / "stratify_report.md").write_text("\n".join(L), encoding="utf-8")

    return {
        "n_gt": len(rows), "n_hit": int(hit.sum()), "recall": float(hit.mean()),
        "per_image": per_image, "report": str(out_dir / "stratify_report.md"),
    }


# --------------------------------------------------------------------------- #
# 预测：缓存 / 推理
# --------------------------------------------------------------------------- #
def load_preds_json(path: str) -> Dict[str, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_preds(items, weights: str, conf: float, tile: int, overlap: float,
                device: str, save: Optional[str]) -> Dict[str, list]:
    from stomata.dl.sliced_inference import SlicedPredictor, ultralytics_tile_predictor

    sp = SlicedPredictor(
        ultralytics_tile_predictor(weights, conf=conf, imgsz=tile, device=device),
        tile, overlap, "center",
    )
    out: Dict[str, list] = {}
    for i, (stem, img_path, _) in enumerate(items, 1):
        img = imread_any(img_path)
        if img is None:
            out[stem] = []
            continue
        dets = sp(img)
        out[stem] = [[float(d.x0), float(d.y0), float(d.x1), float(d.y1), float(d.score)]
                     for d in dets]
        print(f"  [{i}/{len(items)}] {stem}: {len(out[stem])} 个预测", flush=True)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        with open(save, "w", encoding="utf-8") as f:
            json.dump(out, f)
        print(f"预测已缓存 → {save}")
    return out


def collect_items(images_dir: str, labels_dir: str, exclude: Sequence[str],
                  limit: int) -> List[Tuple[str, str, str]]:
    items = []
    for p in sorted(Path(images_dir).iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem
        if stem in exclude:
            continue
        items.append((stem, str(p), str(Path(labels_dir) / (stem + ".txt"))))
    if limit:
        items = items[:limit]
    return items


# --------------------------------------------------------------------------- #
# 自检：合成数据上验证脚本确实测得准
# --------------------------------------------------------------------------- #
def run_self_test(out_root: Path) -> int:
    """构造合成数据集，植入「失焦→更容易漏检」的效应，验证脚本能把它检出来。"""
    rng = np.random.default_rng(0)
    root = out_root / "selftest"
    idir, ldir = root / "images", root / "labels"
    idir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)

    N_IMG, N_PER = 6, 40
    W, H = 900, 700
    truth: Dict[str, list] = {}

    for i in range(N_IMG):
        img = np.full((H, W), 140, np.uint8)
        # 背景：平滑噪声纹理 + 叶脉暗线
        tex = cv2.GaussianBlur(rng.normal(140, 18, (H, W)).astype(np.float32), (0, 0), 9)
        img = np.clip(tex, 0, 255).astype(np.uint8)
        for _ in range(3):
            x0 = int(rng.integers(0, W))
            cv2.line(img, (x0, 0), (x0 + int(rng.integers(-120, 120)), H), 95, 14)

        recs = []
        for _ in range(N_PER):
            major = float(rng.uniform(26, 78))
            aspect = float(rng.uniform(0.35, 0.85))
            minor = major * aspect
            cx, cy = float(rng.uniform(60, W - 60)), float(rng.uniform(60, H - 60))
            ang = float(rng.uniform(0, 180))
            a, b = major / 2, minor / 2
            # 保卫细胞亮环 + 中央暗孔
            cv2.ellipse(img, (int(cx), int(cy)), (int(a), int(b)), ang, 0, 360, 215, 2)
            cv2.ellipse(img, (int(cx), int(cy)), (int(a * 0.30), int(b * 0.30)), ang, 0, 360, 55, -1)
            blurred = bool(rng.random() < 0.5)
            recs.append((cx, cy, major, minor, ang, blurred))

        # 植入失焦：对局部邻域做高斯模糊 + 轻微降对比
        for (cx, cy, major, minor, ang, blurred) in recs:
            if not blurred:
                continue
            r = int(major * 0.9)
            x0, x1 = max(0, int(cx - r)), min(W, int(cx + r))
            y0, y1 = max(0, int(cy - r)), min(H, int(cy + r))
            patch = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = cv2.GaussianBlur(patch, (0, 0), max(1.5, major * 0.12))

        stem = f"syn_{i:02d}"
        cv2.imwrite(str(idir / f"{stem}.png"), img)

        # 标签：归一化 4 角点
        lines = []
        for (cx, cy, major, minor, ang, blurred) in recs:
            a, b, t = major / 2, minor / 2, math.radians(ang)
            R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
            corners = np.array([[-a, -b], [a, -b], [a, b], [-a, b]], float) @ R.T + np.array([cx, cy])
            flat = corners / np.array([W, H])
            lines.append("0 " + " ".join(f"{v:.6f}" for v in flat.ravel()))
        (ldir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        # 植入预测：失焦者命中率显著降低，另加少量 FP
        preds = []
        for (cx, cy, major, minor, ang, blurred) in recs:
            base = 0.95 if major >= 40 else 0.80
            p = base * (0.45 if blurred else 0.95)
            if rng.random() < p:
                jx, jy = rng.normal(0, 2.5, 2)
                s = float(np.clip(rng.normal(0.75, 0.12), 0.05, 0.99))
                half = major * 0.55
                preds.append([cx + jx - half, cy + jy - half, cx + jx + half, cy + jy + half, s])
        for _ in range(int(0.08 * N_PER)):
            cx, cy = float(rng.uniform(0, W)), float(rng.uniform(0, H))
            half = float(rng.uniform(15, 35))
            preds.append([cx - half, cy - half, cx + half, cy + half, float(rng.uniform(0.3, 0.6))])
        truth[stem] = preds

    pjson = root / "preds.json"
    pjson.write_text(json.dumps(truth), encoding="utf-8")

    items = collect_items(str(idir), str(ldir), [], 0)
    res = analyze(items, truth, root, DEFAULT_SIZE_BINS, min_bin_n=10,
                  tol_mode="image", top_images=6)

    # 判定：脚本应当在「清晰度」这一行检出显著差距
    import re
    txt = (root / "stratify_report.md").read_text(encoding="utf-8")
    sec = txt.split("### 局部清晰度")[1].split("###")[0]
    ps = [float(m) for m in re.findall(r"(?<![\d.])(0\.\d+|<1e-4)", sec) if m != "<1e-4"]
    ok_sig = any(p < 0.05 for p in ps) or "<1e-4" in sec

    print("\n=== 自检结果 ===")
    print(f"合成 GT: {res['n_gt']}，整体召回: {res['recall']:.3f}")
    print(sec.strip())
    print("\n判定：植入的「失焦 → 漏检」效应是否被检出 →", "PASS" if ok_sig else "FAIL")
    print(f"完整报表: {res['report']}")
    return 0 if ok_sig else 1


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="气孔漏检三轴解耦诊断（失焦 / 低对比 / 开口度 / 粘连 vs 尺寸）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--images", help="图像目录")
    ap.add_argument("--labels", help="标签目录（与图像同 stem 的 .txt）")
    ap.add_argument("--weights", help="模型权重；提供 --preds-json 时可省略")
    ap.add_argument("--preds-json", help="复用缓存的预测 JSON")
    ap.add_argument("--save-preds", help="把推理结果缓存到该 JSON")
    ap.add_argument("--conf", type=float, default=0.40, help="推理置信度（默认 0.40）")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol-mode", choices=["image", "gt"], default="image",
                    help="image=整图统一容差(与 eval_human 口径一致, 默认); "
                         "gt=逐目标 0.35×长轴")
    ap.add_argument("--size-bins", default="0,40,70,inf",
                    help="尺寸分档边界，默认 0,40,70,inf")
    ap.add_argument("--min-bin-n", type=int, default=10,
                    help="尺寸档内少于该样本数则跳过二分检验")
    ap.add_argument("--exclude", nargs="*", default=["118.2", "221.2"],
                    help="排除的图像 stem（默认排除 118.2/221.2：伪标签污染 / 标注不全）")
    ap.add_argument("--include-all", action="store_true", help="不排除任何图像")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    ap.add_argument("--top-images", type=int, default=10)
    ap.add_argument("--out", default="outputs_dl/stratify")
    ap.add_argument("--self-test", action="store_true",
                    help="在合成数据上验证脚本本身（不需要真实数据和 ultralytics）")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)

    if args.self_test:
        return run_self_test(out_dir)

    if not args.images or not args.labels:
        ap.error("需要 --images 与 --labels（或用 --self-test）")

    bins = []
    edges = [float(x) for x in args.size_bins.split(",")]
    names = ["<40px", "40-70px", ">=70px"]
    for i in range(len(edges) - 1):
        nm = names[i] if i < len(names) else f"[{edges[i]:g},{edges[i+1]:g})"
        bins.append((edges[i], edges[i + 1], nm))

    exclude = [] if args.include_all else list(args.exclude)
    items = collect_items(args.images, args.labels, exclude, args.limit)
    print(f"共 {len(items)} 张图" + (f"，排除 {exclude}" if exclude else ""))

    if args.preds_json:
        preds = load_preds_json(args.preds_json)
        print(f"复用缓存预测: {args.preds_json}")
    else:
        if not args.weights:
            ap.error("需要 --weights 或 --preds-json")
        print(f"开始推理（conf={args.conf}, tile={args.tile}, overlap={args.overlap}）…")
        preds = infer_preds(items, args.weights, args.conf, args.tile,
                            args.overlap, args.device, args.save_preds)

    res = analyze(items, preds, out_dir, bins, args.min_bin_n,
                  args.tol_mode, args.top_images)
    print(f"\nGT {res['n_gt']}，命中 {res['n_hit']}，整体召回 {res['recall']:.3f}")
    print(f"报表: {res['report']}")
    print(f"明细: {out_dir/'instances.csv'}  {out_dir/'predictions.csv'}  {out_dir/'per_image.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
