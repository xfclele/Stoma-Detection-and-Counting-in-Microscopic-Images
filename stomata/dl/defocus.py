"""离焦模糊增强：模拟显微镜焦平面外的视野（整片或局部失焦）。

分层诊断（docs/error_stratify_v2*.md）表明：在同一尺寸档、甚至同一张图内，清晰度最低的 1/4
气孔召回低 5–9 个百分点，而且与对比度独立。所以增强需要同时覆盖两种情况：

* 局部失焦（默认 70%）：用平滑随机场混合清晰图和模糊图，模拟叶面起伏造成的部分离焦；
* 整片失焦（30%）：整块切片模糊。

点扩散函数默认用圆盘核（几何光学离焦 PSF），也会混用高斯核。模糊区域可同时降低对比度，
并加入少量传感器噪声，因为真实失焦图像并不是"干净的模糊"。
增强只改像素，不改框：失焦的气孔仍然是气孔，标签保持不变。

用法（在 ultralytics 训练进程 fork dataloader 之前调用）：

    from stomata.dl.defocus import install_defocus_augment
    install_defocus_augment(p=0.5)
    YOLO(...).train(...)
"""
from __future__ import annotations

import cv2
import numpy as np


def disk_kernel(radius: float) -> np.ndarray:
    """半径 radius（像素，可为小数）的归一化圆盘核，边缘做抗锯齿。"""
    r = max(float(radius), 0.5)
    n = int(np.ceil(r)) * 2 + 1
    yy, xx = np.mgrid[:n, :n] - n // 2
    k = np.clip(r + 0.5 - np.hypot(xx, yy), 0, 1).astype(np.float32)
    return k / k.sum()


def defocus_blur(img: np.ndarray, radius: float, kind: str = "disk") -> np.ndarray:
    if kind == "gauss":
        return cv2.GaussianBlur(img, (0, 0), radius / 1.5)
    return cv2.filter2D(img, -1, disk_kernel(radius), borderType=cv2.BORDER_REFLECT)


def smooth_mask(h: int, w: int, rng: np.random.Generator, coverage: float) -> np.ndarray:
    """平滑随机场：低分辨率噪声上采样，再按 coverage 分位数做软阈值，得到 0–1 掩膜。"""
    gh, gw = rng.integers(2, 6), rng.integers(2, 6)
    field = cv2.resize(rng.random((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), max(h, w) / 16)
    t = np.quantile(field, 1 - coverage)
    soft = max(float(field.std()) * 0.5, 1e-3)
    return np.clip((field - t) / soft + 0.5, 0, 1)


def apply_defocus(img: np.ndarray, rng: np.random.Generator, radius=(1.5, 5.0), local_p: float = 0.7,
                  contrast=(0.7, 1.0), contrast_p: float = 0.5, noise=(0.0, 3.0)) -> np.ndarray:
    """对一张 uint8 BGR 图像施加离焦退化。"""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    r = float(rng.uniform(*radius))
    b = defocus_blur(f, r, "disk" if rng.random() < 0.7 else "gauss")
    if rng.random() < contrast_p:
        c = float(rng.uniform(*contrast))
        mu = cv2.GaussianBlur(b, (0, 0), 25)
        b = mu + (b - mu) * c
    if noise[1] > 0:
        b = b + rng.normal(0, rng.uniform(*noise), b.shape).astype(np.float32)
    if rng.random() < local_p:
        m = smooth_mask(h, w, rng, coverage=float(rng.uniform(0.3, 0.8)))[..., None]
        b = m * b + (1 - m) * f
    return np.clip(b, 0, 255).astype(np.uint8)


class DefocusAugment:
    """ultralytics 训练变换：以概率 p 对 labels["img"] 施加离焦，标签不变。"""

    def __init__(self, p: float = 0.5, **kw):
        self.p, self.kw = p, kw
        self.rng = np.random.default_rng()

    def __call__(self, labels: dict) -> dict:
        if self.rng.random() < self.p:
            labels["img"] = apply_defocus(labels["img"], self.rng, **self.kw)
        return labels


def install_defocus_augment(p: float = 0.5, **kw) -> None:
    """给 YOLODataset.build_transforms 打补丁：训练集（augment=True）在 Format 之前插入 DefocusAugment。
    验证集不受影响。必须在构建 dataloader 之前调用。"""
    from ultralytics.data.dataset import YOLODataset

    if getattr(YOLODataset, "_defocus_installed", False):
        YOLODataset._defocus_kw = dict(p=p, **kw)
        return
    orig = YOLODataset.build_transforms

    def build_transforms(self, hyp=None):
        t = orig(self, hyp)
        if self.augment:
            t.insert(len(t.transforms) - 1, DefocusAugment(**YOLODataset._defocus_kw))
        return t

    YOLODataset.build_transforms = build_transforms
    YOLODataset._defocus_kw = dict(p=p, **kw)
    YOLODataset._defocus_installed = True
