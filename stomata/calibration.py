"""物理尺度标定：由载物台测微尺 (stage micrometer) 图像计算 µm/px。

同一物镜 + 相机拍摄的测微尺图像，刻线为等间距暗线（常见 1 mm / 100 格 ⇒ 每格 10 µm）。
算法：
    1. 取竖直暗线能量最强的水平条带（细刻度所在行），按列求平均得到一维剖面；
    2. 去趋势后做自相关，在 [min_period, max_period] 内找第一主峰 ⇒ 刻度周期初值；
    3. 在剖面上检测每条刻线的位置（按周期约束找局部极小），
       对"刻线序号 → 位置"做最小二乘直线拟合，斜率即亚像素精度的周期 p；
    4. µm/px = division_um / p。

得到 µm/px 后，就可以用物理尺寸设置气孔先验：expected_major_px = 气孔长轴(µm) / (µm/px)，
不同放大倍率的图像共用一套 µm 参数。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks

from .preprocessing import to_gray_float


@dataclass
class MicrometerCalibration:
    um_per_px: float
    period_px: float
    n_ticks: int
    residual_px: float        # 刻线位置对拟合直线的 RMS 残差（越小越可靠）
    orientation: str          # "vertical"（竖刻线，沿 x 测量）或 "horizontal"


def _profile(gray: np.ndarray, band_frac: float = 0.15) -> np.ndarray:
    """竖刻线：沿 x 的暗线剖面。选取竖直边缘能量最强的水平条带求列平均。"""
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    row_energy = gx.mean(1)
    k = max(3, int(len(row_energy) * band_frac))
    sm = np.convolve(row_energy, np.ones(k) / k, mode="same")
    c = int(np.argmax(sm))
    y0, y1 = max(0, c - k // 2), min(gray.shape[0], c + k // 2 + 1)
    prof = gray[y0:y1].mean(0)
    trend = cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (0, 0), 25).ravel()
    return (trend - prof).astype(np.float64)       # 暗线 → 正峰


def calibrate_stage_micrometer(image: np.ndarray, division_um: float = 10.0, min_period: float = 5.0,
                               max_period: float = 300.0) -> MicrometerCalibration:
    gray = to_gray_float(image)
    best = None
    for orient, g in (("vertical", gray), ("horizontal", gray.T)):
        prof = _profile(g)
        x = prof - prof.mean()
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        ac /= ac[0] + 1e-12
        lo, hi = int(min_period), min(int(max_period), len(ac) - 1)
        pk, props = find_peaks(ac[lo:hi], height=0.1)
        if len(pk) == 0:
            continue
        p0 = float(lo + pk[np.argmax(props["peak_heights"])])
        # 以更小的周期倍数优先（主峰可能是 5 格长刻线的周期）
        for cand in sorted(lo + pk):
            if ac[cand] > 0.6 * ac[int(p0)] and abs(p0 / cand - round(p0 / cand)) < 0.1:
                p0 = float(cand)
                break
        ticks, _ = find_peaks(prof, distance=max(2, int(0.6 * p0)), prominence=0.3 * np.std(prof))
        if len(ticks) < 4:
            continue
        # 亚像素：抛物线插值
        sub = []
        for t in ticks:
            if 0 < t < len(prof) - 1:
                a, b, c = prof[t - 1], prof[t], prof[t + 1]
                den = a - 2 * b + c
                sub.append(t + (0.5 * (a - c) / den if den != 0 else 0.0))
        sub = np.array(sub)
        idx = np.round((sub - sub[0]) / p0)
        A = np.vstack([idx, np.ones_like(idx)]).T
        keep = np.ones(len(sub), bool)
        for _ in range(3):  # 迭代剔除离群刻线（污点、长短刻线错位）后重新拟合
            (slope, icpt), *_ = np.linalg.lstsq(A[keep], sub[keep], rcond=None)
            r = np.abs(A @ [slope, icpt] - sub)
            keep = r < max(1.0, 3 * np.median(r[keep]) + 1e-6)
        res = float(np.sqrt(np.mean((A[keep] @ [slope, icpt] - sub[keep]) ** 2)))
        cal = MicrometerCalibration(division_um / slope, float(slope), int(keep.sum()), res, orient)
        # 选方向：自相关周期峰越高、刻线位置残差（相对周期）越小越可信
        score = float(ac[int(round(p0))]) / (1.0 + res / slope)
        if best is None or score > best[0]:
            best = (score, cal)
    if best is None:
        raise RuntimeError("未能在图像中检测到等间距刻线，请确认这是测微尺图像。")
    return best[1]
