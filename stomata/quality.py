"""拍摄质量指标：清晰度、对比度、曝光截断、对焦均匀性。
Capture-quality metrics: sharpness, contrast, exposure clipping and focus uniformity.

清晰度和对比度的定义与 `scripts/error_stratify_logit.py` 一致：
The sharpness and contrast definitions match `scripts/error_stratify_logit.py`:
    · 清晰度 sharpness = ΣE[(G1−G2)²] / ΣE[(G2−G8)²]
      细频带与粗频带能量之比；失焦先衰减细频带，比值与亮度、对比度无关。
      fine-band to coarse-band energy ratio; defocus attenuates the fine band first, and the ratio
      is independent of brightness and contrast.
    · 对比度 contrast = std(G2−G8)，粗频带（气孔轮廓尺度）的幅度。
      contrast = std(G2−G8), the amplitude of the coarse band (the scale of stoma outlines).
整图按网格分块计算，用分块的中位数代表整图，用较差分块（低分位）衡量对焦是否均匀。
The image is split into a grid; the median over cells represents the whole image, and a low
percentile over cells measures whether focus is uniform.
"""
from __future__ import annotations

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def image_quality(image: np.ndarray, grid: tuple[int, int] = (4, 4), min_energy: float = 4.0) -> dict:
    """计算一张图的拍摄质量指标。
    Compute capture-quality metrics for one image.

    参数 / Parameters
        image       BGR 或灰度图，8 位 / BGR or greyscale, 8-bit
        grid        (列, 行) 分块数 / (columns, rows) of cells
        min_energy  粗频带均方能量低于此值的分块视为空白背景，不参与清晰度统计
                    cells whose mean coarse-band energy is below this are treated as blank background
                    and left out of the sharpness statistics
    返回 / Returns
        sharpness          有纹理分块清晰度的中位数 / median sharpness over textured cells
        sharpness_low      有纹理分块清晰度的 25% 分位（对焦不均时明显低于中位数）
                           25th percentile over textured cells (well below the median when focus is uneven)
        focus_uniformity   sharpness_low / sharpness，1 = 均匀 / 1 = uniform
        contrast           分块对比度的中位数 / median cell contrast
        clipped            灰度 ≤ 2 或 ≥ 253 的像素比例 / fraction of pixels with grey level ≤ 2 or ≥ 253
        dynamic_range      灰度 99% 分位 − 1% 分位 / grey-level 99th minus 1st percentile
        textured_cells     参与清晰度统计的分块数 / number of cells used for sharpness
    """
    gray = _to_gray(image)
    g = gray.astype(np.float32)
    g1, g2, g8 = (cv2.GaussianBlur(g, (0, 0), s) for s in (1, 2, 8))
    fine2, band = (g1 - g2) ** 2, g2 - g8
    coarse2 = band ** 2

    H, W = gray.shape
    nx, ny = grid
    xs = np.linspace(0, W, nx + 1).astype(int)
    ys = np.linspace(0, H, ny + 1).astype(int)
    sharp, contrast = [], []
    for j in range(ny):
        for i in range(nx):
            sl = (slice(ys[j], ys[j + 1]), slice(xs[i], xs[i + 1]))
            c2 = float(coarse2[sl].mean())
            contrast.append(float(band[sl].std()))
            if c2 >= min_energy:
                sharp.append(float(fine2[sl].mean()) / c2)

    p1, p99 = np.percentile(gray, [1, 99])
    if sharp:
        s_med, s_low = float(np.median(sharp)), float(np.percentile(sharp, 25))
    else:
        s_med = s_low = 0.0
    return {
        "sharpness": s_med,
        "sharpness_low": s_low,
        "focus_uniformity": s_low / s_med if s_med > 0 else 0.0,
        "contrast": float(np.median(contrast)),
        "clipped": float(((gray <= 2) | (gray >= 253)).mean()),
        "dynamic_range": float(p99 - p1),
        "textured_cells": len(sharp),
    }


# 阈值取自 234 张训练图的分布（scripts/capture_qc.py --calibrate 可重算）：
# "重拍"阈值 = 训练图 5% 分位，"注意"阈值 = 10% 分位。在 57 张验证图上，
# sharpness_low < 0.150 的 6 张召回 0.831（其余 0.955），contrast < 26.6 的 4 张召回 0.768（其余 0.955）。
# Thresholds come from the distribution over the 234 training images (recompute with
# scripts/capture_qc.py --calibrate): "retake" = 5th percentile of training images, "check" = 10th percentile.
# On the 57 validation images, the 6 with sharpness_low < 0.150 have recall 0.831 (others 0.955) and the
# 4 with contrast < 26.6 have recall 0.768 (others 0.955).
QC_THRESHOLDS = {
    "sharpness_low_retake": 0.150,
    "sharpness_check": 0.191,
    "contrast_retake": 26.6,
    "contrast_check": 34.6,
    "focus_uniformity_check": 0.76,
    "clipped_check": 0.20,      # 高于全部训练图 / above every training image
    # 预测框长边中位数（px）。训练标注长边 5–95% 为 42–72 px，预测框平均比人工框大约 8%。
    # Median predicted box long side (px). Training labels span 42–72 px (5th–95th percentile) and
    # predicted boxes are about 8% larger than human boxes on average.
    "size_min": 45.0,
    "size_max": 78.0,
}

RETAKE, CHECK, OK = "重拍 RETAKE", "注意 CHECK", "合格 OK"


def assess(q: dict, thresholds: dict | None = None) -> tuple[str, list[str]]:
    """按阈值给出结论与原因。q 为 image_quality() 的结果，可另含 n_stomata、size_px
    （检出数 ≥ 10 时才检查尺寸）。
    Turn metrics into a verdict and reasons. q is the output of image_quality(), optionally with
    n_stomata and size_px added (size is checked only when at least 10 stomata are detected)."""
    t = {**QC_THRESHOLDS, **(thresholds or {})}
    retake, check = [], []
    if q["sharpness_low"] < t["sharpness_low_retake"]:
        retake.append("失焦（较差区域清晰度低于训练图 95%） / defocused (worse regions less sharp than 95% of training images)")
    elif q["sharpness"] < t["sharpness_check"]:
        check.append("清晰度偏低 / sharpness on the low side")
    if q["contrast"] < t["contrast_retake"]:
        retake.append("对比度过低（低于训练图 95%），检查照明或是否做了与训练图相同的增强 / "
                      "contrast too low (below 95% of training images); check illumination and whether the same "
                      "enhancement as the training images was applied")
    elif q["contrast"] < t["contrast_check"]:
        check.append("对比度偏低 / contrast on the low side")
    if q["focus_uniformity"] < t["focus_uniformity_check"]:
        check.append("对焦不均（部分区域明显更模糊），检查样品是否压平 / "
                     "uneven focus (some regions clearly blurrier); check that the sample is flat")
    if q["clipped"] > t["clipped_check"]:
        check.append("纯黑/纯白像素过多 / too many pure black/white pixels")
    if "n_stomata" in q and q["n_stomata"] == 0:
        retake.append("未检出气孔：倍率、焦面或视野可能不对 / no stomata detected: magnification, focal plane or field may be wrong")
    elif q.get("n_stomata", 0) >= 10 and q.get("size_px") is not None and not t["size_min"] <= q["size_px"] <= t["size_max"]:
        retake.append(f"气孔长边约 {q['size_px']:.0f} px，超出训练范围 {t['size_min']:.0f}–{t['size_max']:.0f} px，"
                      f"请调整放大倍率 / stoma long side ≈ {q['size_px']:.0f} px, outside the training range "
                      f"{t['size_min']:.0f}–{t['size_max']:.0f} px; adjust the magnification")
    if retake:
        return RETAKE, retake + check
    return (CHECK, check) if check else (OK, [])
