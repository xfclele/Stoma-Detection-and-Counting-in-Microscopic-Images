"""气孔模板：合成先验模板 + 自举 (self-bootstrapping) 数据模板 + 旋转/尺度模板库。

为什么用模板匹配而不是单纯阈值/形态学？
------------------------------------------------
气孔的判别信息在于 *结构*：一个闭合椭圆暗环 + 内部一对亮的肾形保卫细胞 + 中央暗裂隙。
细胞壁同样是暗线、叶脉同样是暗区，仅凭灰度或形态学无法区分；而归一化互相关 (NCC)

    NCC(x) = Σ (I - Ī)(T - T̄) / (‖I - Ī‖ · ‖T - T̄‖)

对局部亮度/对比度（残余光照不均）天然不变，只对 "结构形状" 敏感。

自举流程（完全无监督）
----------------------
1. 在暗结构图 (black-hat) 上用 **合成椭圆环模板** 做多方向匹配，乘以局部暗结构能量，
   取得分最高且互不重叠的 K 个种子（高置信度，几乎全是气孔）。
2. 用二阶矩估计每个种子的主轴方向，把 patch 旋转到"长轴竖直"的规范姿态。
3. 求平均并做 4 重对称化（气孔关于长/短轴对称）⇒ 数据驱动模板，噪声 ∝ 1/√K。
4. 可再迭代一次：用新模板重新选种子、按匹配到的最佳角度对齐，模板更锐利。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from skimage.feature import peak_local_max


# ----------------------------------------------------------------------------
# 合成模板
# ----------------------------------------------------------------------------
def synthetic_ring_template(a: float, b: float, theta_deg: float = 0.0, pad: int = 6) -> np.ndarray:
    """暗结构域 (亮=暗线) 的理想气孔：外椭圆环 + 内部透镜形裂隙环。"""
    size = int(2 * max(a, b) + 2 * pad) | 1
    T = np.zeros((size, size), np.float32)
    c = (size // 2, size // 2)
    cv2.ellipse(T, c, (int(round(a)), int(round(b))), theta_deg, 0, 360, 1.0, 2, cv2.LINE_AA)
    cv2.ellipse(T, c, (int(round(a * 0.72)), int(round(b * 0.38))), theta_deg, 0, 360, 0.8, 2, cv2.LINE_AA)
    return cv2.GaussianBlur(T, (0, 0), 1.0)


# ----------------------------------------------------------------------------
# 旋转模板匹配
# ----------------------------------------------------------------------------
def rotate_template(T: np.ndarray, angle_deg: float) -> np.ndarray:
    n = T.shape[0]
    M = cv2.getRotationMatrix2D(((n - 1) / 2.0, (n - 1) / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(T, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def match_full(image: np.ndarray, T: np.ndarray) -> np.ndarray:
    """cv2.matchTemplate(TM_CCOEFF_NORMED) 并把结果回填到与原图同尺寸（模板中心对齐）。"""
    r = cv2.matchTemplate(image, T, cv2.TM_CCOEFF_NORMED)
    out = np.full(image.shape, -1.0, np.float32)
    h, w = T.shape[0] // 2, T.shape[1] // 2
    out[h:h + r.shape[0], w:w + r.shape[1]] = r
    out[~np.isfinite(out)] = -1.0
    return out


def make_variant(T: np.ndarray, scale: float = 1.0, aspect: float = 1.0, crop_radius: Optional[float] = None) -> np.ndarray:
    """模板形变：长轴(竖直)方向缩放 scale，短轴方向缩放 scale/aspect，再裁剪为奇数边长方形。

    aspect > 1 ⇒ 更细长的气孔（本数据中开放度不同的气孔长宽比约 1.1–1.7）。
    crop_radius: 以规范模板像素计的裁剪半径；紧裁剪让 NCC 聚焦气孔本体而非邻域。
    """
    Ts = cv2.resize(T, None, fx=scale / aspect, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = Ts.shape
    r = int(round((crop_radius if crop_radius is not None else T.shape[0] / 2.0) * scale))
    P = cv2.copyMakeBorder(Ts, r, r, r, r, cv2.BORDER_REFLECT)
    cy, cx = h // 2 + r, w // 2 + r
    return np.ascontiguousarray(P[cy - r:cy + r + 1, cx - r:cx + r + 1])


def build_bank(T, scales=(1.0,), aspects=(1.0,), crop_radius=None):
    return [(float(s), float(a), make_variant(T, s, a, crop_radius)) for s in scales for a in aspects]


def multi_orientation_match(image, template, angles, scales=(1.0,), aspects=(1.0,), crop_radius=None):
    """在 角度×尺度×长宽比 模板库上取逐像素最大 NCC。

    返回 (best, best_angle, best_scale, best_aspect)。
    """
    best = np.full(image.shape, -1.0, np.float32)
    best_a = np.zeros(image.shape, np.float32)
    best_s = np.ones(image.shape, np.float32)
    best_r = np.ones(image.shape, np.float32)
    for s, asp, Ts in build_bank(template, scales, aspects, crop_radius):
        if Ts.shape[0] >= min(image.shape):
            continue
        for ang in angles:
            r = match_full(image, rotate_template(Ts, ang))
            upd = r > best
            best[upd] = r[upd]
            best_a[upd] = ang
            best_s[upd] = s
            best_r[upd] = asp
    return best, best_a, best_s, best_r


# ----------------------------------------------------------------------------
# 自举数据模板
# ----------------------------------------------------------------------------
def _moment_orientation(patch: np.ndarray, radius: float) -> float:
    """二阶中心矩求主轴方向（度），返回使长轴旋转到竖直所需的角度。"""
    R = patch.shape[0] // 2
    yy, xx = np.mgrid[-R:patch.shape[0] - R, -R:patch.shape[1] - R].astype(np.float32)
    w = patch * ((xx ** 2 + yy ** 2) < radius ** 2)
    mxx, myy, mxy = (w * xx * xx).sum(), (w * yy * yy).sum(), (w * xx * yy).sum()
    # 惯性主轴相对 x 轴的角度；cv2 旋转角为逆时针（图像 y 向下时视觉上是逆时针）
    return 0.5 * float(np.degrees(np.arctan2(2 * mxy, mxx - myy)))


def _make_major_vertical(T: np.ndarray) -> np.ndarray:
    """规范姿态 = 长轴竖直。

    以"外轮廓填充区域"的二阶矩判断：分割暗环 → 填充孔洞 → 取含中心的连通域，
    若其水平方差 > 竖直方差则转置。比依赖中央裂隙明暗更稳健（有的成像条件下孔隙呈亮色）。
    """
    from scipy import ndimage as ndi

    mu, sd = float(T.mean()), float(T.std())
    dark = T < mu - 0.3 * sd
    filled = ndi.binary_fill_holes(dark)
    lab, _ = ndi.label(filled)
    c = T.shape[0] // 2
    region = lab == lab[c, c] if lab[c, c] > 0 else filled
    ys, xs = np.nonzero(region)
    if len(xs) < 10:
        return T
    return T.T.copy() if xs.var() > ys.var() * 1.02 else T


def symmetrize(T: np.ndarray) -> np.ndarray:
    return (T + T[:, ::-1] + T[::-1, :] + T[::-1, ::-1]) / 4.0


def extract_aligned_mean(image, centers, angles, half: int) -> np.ndarray:
    """按给定角度把各 patch 旋转到规范姿态后取平均。"""
    H, W = image.shape
    acc, n = np.zeros((2 * half, 2 * half), np.float64), 0
    for (r, c), ang in zip(centers, angles):
        r, c = int(r), int(c)
        if r - half < 0 or c - half < 0 or r + half > H or c + half > W:
            continue
        p = image[r - half:r + half, c - half:c + half]
        M = cv2.getRotationMatrix2D((half - 0.5, half - 0.5), float(ang), 1.0)
        acc += cv2.warpAffine(p, M, (2 * half, 2 * half), borderMode=cv2.BORDER_REFLECT)
        n += 1
    if n == 0:
        raise RuntimeError("没有可用于自举模板的种子，请检查 expected_size 或图像内容。")
    return symmetrize((acc / n).astype(np.float32))


@dataclass
class StomaTemplate:
    """规范姿态（长轴竖直）的数据模板及其几何量。"""
    image: np.ndarray                 # 模板灰度 (奇数边长)
    major_px: float                   # 模板中气孔外轮廓长轴（像素）
    minor_px: float                   # 短轴
    n_seeds: int = 0
    history: list = field(default_factory=list)

    def save(self, path: str) -> None:
        np.savez(path, image=self.image, major_px=self.major_px, minor_px=self.minor_px, n_seeds=self.n_seeds)

    @classmethod
    def load(cls, path: str) -> "StomaTemplate":
        d = np.load(path)
        return cls(d["image"].astype(np.float32), float(d["major_px"]), float(d["minor_px"]), int(d["n_seeds"]))


def measure_template_axes(T: np.ndarray) -> tuple[float, float]:
    """在模板上分割暗环并拟合椭圆，得到外轮廓长/短轴。

    做法：暗像素 (T < μ - 0.5σ) → 闭运算连接环 → 填充 → 取包含中心的连通域 → fitEllipse。
    """
    mu, sd = float(T.mean()), float(T.std())
    dark = (T < mu - 0.5 * sd).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c0 = (T.shape[1] / 2.0, T.shape[0] / 2.0)
    cnts = [c for c in cnts if len(c) >= 5 and cv2.pointPolygonTest(c, c0, False) >= 0]
    if not cnts:
        n = T.shape[0]
        return 0.7 * n, 0.55 * n
    (_, _), (d1, d2), _ = cv2.fitEllipse(max(cnts, key=cv2.contourArea))
    return float(max(d1, d2)), float(min(d1, d2))


def bootstrap_template(
    enhanced: np.ndarray,
    ridge: np.ndarray,
    energy: np.ndarray,
    expected_major: float = 34.0,
    n_seeds: int = 60,
    n_iter: int = 2,
    angle_step: float = 15.0,
) -> StomaTemplate:
    """从单张图像无监督地学习气孔模板（见模块文档）。"""
    a, b = expected_major / 2.0, expected_major / 2.0 * 0.78
    half = int(round(expected_major * 0.75))
    min_dist = int(expected_major * 0.6)
    border = half + 2

    # ---- 第 0 轮：合成椭圆环模板在暗结构域匹配，乘以局部能量抑制空白背景 ----
    syn = synthetic_ring_template(a, b)
    angles = np.arange(0, 180, angle_step)
    ncc = multi_orientation_match(ridge, syn, angles)[0]
    gate = np.clip(energy / (np.percentile(energy, 90) + 1e-6), 0, 1)
    score = ncc * gate
    pk = peak_local_max(score, min_distance=min_dist, num_peaks=n_seeds, exclude_border=border)
    rot = [_moment_orientation(ridge[r - half:r + half, c - half:c + half], a * 1.05) for r, c in pk]
    T = extract_aligned_mean(enhanced, pk, rot, half)
    T = _make_major_vertical(T)
    history = [T]

    # ---- 后续轮：用数据模板重新选种子，按最佳匹配角度对齐 ----
    for _ in range(max(0, n_iter - 1)):
        Tc = T[1:, 1:] if T.shape[0] % 2 == 0 else T
        ncc, best_a, _, _ = multi_orientation_match(enhanced, Tc, angles)
        score = ncc * gate
        pk = peak_local_max(score, min_distance=min_dist, num_peaks=n_seeds, exclude_border=border)
        # 模板被旋转 +ang 后匹配 ⇒ 将 patch 旋转 -ang 回到规范姿态
        T = _make_major_vertical(extract_aligned_mean(enhanced, pk, [-best_a[r, c] for r, c in pk], half))
        history.append(T)

    T = T[1:, 1:] if T.shape[0] % 2 == 0 else T
    major, minor = measure_template_axes(T)
    return StomaTemplate(T.astype(np.float32), major, minor, len(pk), history)
