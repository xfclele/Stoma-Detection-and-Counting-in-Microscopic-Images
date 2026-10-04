#!/usr/bin/env python
"""把 data/samples/pre-trainning 的"图像 + 检测叠加图"还原为点标注与有效区域掩膜。

输入目录结构（来自外部自动计数流程）：
    images/<name>.jpg      原图
    overlays/<name>.jpg    同一图像叠加了检测结果：
                           · 每个保留的气孔中心画一个绿色小圆点（约 30 px²）
                           · 被剔除的非表皮区域以暗红色半透明遮罩覆盖：overlay ≈ 0.35·img + (0, 0, 52)（BGR）
    auto_count_results.csv 每图计数（用于核对提取结果）

输出（--out 目录）：
    labels/<name>.json     {"image", "points": [[x, y], ...], "major_px": [...], "mask": "masks/<name>.png"}
    masks/<name>.png       有效表皮区域掩膜（255 = 有效，0 = 被剔除）
    summary.csv            每图：提取点数 / CSV 计数 / 有效面积比 / 估计长轴

注意：这些点是外部自动流程（YOLO）的输出，即**伪标签**而非人工真值。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata.detector import ClassicalStomataDetector, DetectorConfig  # noqa: E402
from stomata.preprocessing import clahe, flat_field_correct, to_gray_float  # noqa: E402


def extract_points(img: np.ndarray, ov: np.ndarray, min_area: int = 8, max_area: int = 200) -> np.ndarray:
    """绿色圆点：G 明显高于 R、B 的小连通域，取质心。"""
    b, g, r = [ov[..., i].astype(np.int32) for i in range(3)]
    green = (g > 150) & (g - r > 80) & (g - b > 80)
    n, _, st, cen = cv2.connectedComponentsWithStats(green.astype(np.uint8))
    keep = (st[1:, cv2.CC_STAT_AREA] >= min_area) & (st[1:, cv2.CC_STAT_AREA] <= max_area)
    return cen[1:][keep].astype(np.float32)


def extract_valid_mask(img: np.ndarray, ov: np.ndarray, points: np.ndarray) -> np.ndarray:
    """红色遮罩区：overlay 的 R−B 明显偏大（原图为灰度，R≈B）。返回有效区域 (uint8, 255=有效)。"""
    b, r = ov[..., 0].astype(np.int32), ov[..., 2].astype(np.int32)
    red = ((r - b) > 25).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k)   # 填补遮罩内的近黑/近白像素（色差被压缩）
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k)
    valid = (1 - red).astype(np.uint8) * 255
    for x, y in points:                               # 绿点处一定有效
        cv2.circle(valid, (int(x), int(y)), 6, 255, -1)
    return valid


def estimate_major(img: np.ndarray, points: np.ndarray, guess: float) -> np.ndarray:
    """在每个点上用射线-椭圆拟合测量长轴；失败的点用成功点的中位数代替。"""
    det = ClassicalStomataDetector(DetectorConfig(expected_major_px=guess))
    g = clahe(flat_field_correct(to_gray_float(img)))
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    majors = []
    for x, y in points:
        cov, emaj, emin, res = det._ray_ellipse(g, float(x), float(y), r_max=0.9 * guess)
        ok = emaj > 0 and res <= 0.3 and 0.5 * guess <= emaj <= 1.8 * guess
        majors.append(emaj if ok else np.nan)
    majors = np.array(majors, np.float32)
    med = float(np.nanmedian(majors)) if np.any(np.isfinite(majors)) else guess
    return np.where(np.isfinite(majors), majors, med), med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default="data/samples/pre-trainning")
    ap.add_argument("-o", "--out", default="data/pretraining_labels")
    ap.add_argument("--major-guess", type=float, default=50.0, help="长轴初值（像素），用于射线测量的搜索半径")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    csv_rows = {}
    csv_path = os.path.join(a.src, "auto_count_results.csv")
    if os.path.exists(csv_path):
        csv_rows = {r["image"]: r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}

    summary = []
    names = sorted(f for f in os.listdir(os.path.join(a.src, "images")) if f.lower().endswith((".jpg", ".png", ".tif")))
    for name in names:
        ov_path = os.path.join(a.src, "overlays", name)
        if not os.path.exists(ov_path):
            print(f"跳过 {name}：缺少 overlay")
            continue
        img = cv2.imread(os.path.join(a.src, "images", name))
        ov = cv2.imread(ov_path)
        pts = extract_points(img, ov)
        valid = extract_valid_mask(img, ov, pts)
        majors, med = estimate_major(img, pts, a.major_guess)
        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(a.out, "masks", stem + ".png"), valid)
        with open(os.path.join(a.out, "labels", stem + ".json"), "w", encoding="utf-8") as f:
            # 框尺寸统一用整图中位数（逐点射线测量噪声较大，会让框尺寸抖动）；逐点测量值另存备查
            json.dump({"image": name, "points": np.round(pts, 1).tolist(), "major_px": [round(med, 1)] * len(pts),
                       "major_px_measured": np.round(majors, 1).tolist(),
                       "mask": f"masks/{stem}.png", "source": "pseudo-label (external auto-count overlay)"}, f)
        ref = csv_rows.get(name, {})
        row = {"image": name, "points": len(pts), "csv_valid": ref.get("stomata_valid", ""),
               "valid_area_pct": round(100 * float((valid > 0).mean()), 1), "csv_valid_area_pct": ref.get("valid_area_pct", ""),
               "major_median_px": round(med, 1)}
        summary.append(row)
        print(row)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
