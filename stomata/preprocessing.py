"""预处理模块：光照校正、对比度增强、带通滤波与暗线结构强化。

透射显微镜叶表皮图像的亮度可以分解为

    I(x, y) = L(x, y) · R(x, y) + n(x, y)

其中 L 为低频照明/叶脉阴影（尺度 >> 气孔），R 为组织透过率（包含气孔、细胞壁），
n 为高频噪声。本模块依次：

1. **平场校正 (flat-field)**：用大尺度高斯 G_σbg * I 估计 L，做除法 I / L，
   把乘性阴影变为常数 —— 这一步直接解决"叶脉引起的强暗区"问题，
   也是禁止使用全局阈值的根本原因（未校正前同一气孔在亮区/暗区灰度差异巨大）。
2. **CLAHE**：分块限制对比度的直方图均衡化，进一步拉平局部对比度。
3. **DoG 带通**：G_σ1 - G_σ2，σ1 抑制像素级噪声，σ2 去除残余低频背景。
4. **形态学黑顶帽 (black-hat)**：closing(I) - I，提取宽度小于结构元的暗线
   （保卫细胞外轮廓、气孔裂隙、细胞壁），作为"暗结构强度图"。
5. **多尺度 Hessian 斑点图**：在暗结构密度图上计算 Hessian 行列式 (DoH)，
   两个特征值同号且接近 ⇒ 各向同性斑点（气孔）；一大一小 ⇒ 线状结构（细胞壁）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def to_gray_float(image: np.ndarray) -> np.ndarray:
    """任意输入 (uint8/uint16/float, 灰度/BGR) → float32 灰度 [0, 1]。"""
    img = image
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > 1.0:  # 整型动态范围
        img = img / (65535.0 if hi > 255 else 255.0)
    return np.clip(img, 0.0, 1.0)


def robust_normalize(x: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    """按百分位做鲁棒线性拉伸到 [0, 1]，避免少数极亮/极暗像素主导。"""
    lo, hi = np.percentile(x, [p_lo, p_hi])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def flat_field_correct(gray: np.ndarray, sigma_bg: float = 40.0) -> np.ndarray:
    """乘性光照校正：I / (G_σ * I)。

    σ_bg 需远大于气孔尺寸（~30 px），否则会把气孔本身当作背景除掉。
    """
    bg = cv2.GaussianBlur(gray, (0, 0), sigma_bg)
    flat = gray / (bg + 1e-3)
    return robust_normalize(flat)


def clahe(img01: np.ndarray, clip_limit: float = 2.0, tile: int = 16) -> np.ndarray:
    u8 = (img01 * 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(u8)
    return out.astype(np.float32) / 255.0


def difference_of_gaussians(img: np.ndarray, sigma_lo: float = 1.0, sigma_hi: float = 10.0) -> np.ndarray:
    """带通滤波：保留 [σ_lo, σ_hi] 频带内的结构（气孔轮廓宽 2–4 px、整体 ~30 px）。"""
    return (cv2.GaussianBlur(img, (0, 0), sigma_lo) - cv2.GaussianBlur(img, (0, 0), sigma_hi)).astype(np.float32)


def black_tophat(img: np.ndarray, line_width: int = 9, pre_sigma: float = 1.2) -> np.ndarray:
    """黑顶帽：提取宽度 < line_width 的暗线（保卫细胞壁 / 裂隙 / 表皮细胞壁）。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width, line_width))
    sm = cv2.GaussianBlur(img, (0, 0), pre_sigma)
    bh = cv2.morphologyEx(sm, cv2.MORPH_BLACKHAT, k)
    return robust_normalize(bh, 0.0, 99.5)


def hessian_blob_map(ridge: np.ndarray, sigmas: Sequence[float] = (6, 8, 10)) -> np.ndarray:
    """多尺度 Hessian 行列式斑点响应（尺度归一化 σ²·sqrt(λ1·λ2)）。

    输入为暗结构强度图（亮 = 暗线多）。气孔 = 闭合环 + 内部裂隙，
    在 σ≈气孔半径/√2 尺度下表现为亮斑，λ1, λ2 同为负；
    细胞壁为线状，|λ1| >> |λ2| ≈ 0，sqrt(λ1·λ2) 很小，从而被抑制。
    """
    best = np.zeros_like(ridge, dtype=np.float32)
    for s in sigmas:
        H = hessian_matrix(ridge, sigma=s, order="rc", use_gaussian_derivatives=False)
        l1, l2 = hessian_matrix_eigvals(H)
        both_neg = (l1 < 0) & (l2 < 0)
        resp = np.zeros_like(ridge, dtype=np.float32)
        resp[both_neg] = (s ** 2) * np.sqrt(l1[both_neg] * l2[both_neg])
        best = np.maximum(best, resp)
    return robust_normalize(best, 0.0, 99.9)


@dataclass
class PreprocessResult:
    gray: np.ndarray       # 原始灰度 [0,1]
    flat: np.ndarray       # 平场校正后
    enhanced: np.ndarray   # CLAHE 增强（模板匹配的输入）
    bandpass: np.ndarray   # DoG 带通
    ridge: np.ndarray      # 黑顶帽暗结构强度 [0,1]
    energy: np.ndarray     # 暗结构局部密度（用于抑制空白背景的伪响应）


def preprocess(
    image: np.ndarray,
    sigma_bg: float = 40.0,
    clahe_clip: float = 2.0,
    clahe_tile: int = 16,
    dog_sigmas: tuple[float, float] = (1.0, 10.0),
    line_width: int = 9,
    energy_sigma: float = 8.0,
) -> PreprocessResult:
    gray = to_gray_float(image)
    flat = flat_field_correct(gray, sigma_bg)
    enh = clahe(flat, clahe_clip, clahe_tile)
    bp = difference_of_gaussians(enh, *dog_sigmas)
    ridge = black_tophat(enh, line_width)
    energy = cv2.GaussianBlur(ridge, (0, 0), energy_sigma)
    return PreprocessResult(gray, flat, enh, bp, ridge, energy)
