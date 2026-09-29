"""合成叶表皮图像：气孔（暗外环 + 亮保卫细胞 + 暗裂隙）+ 细胞壁弧线 + 叶脉条纹 + 乘性光照不均 + 噪声。"""
from __future__ import annotations

import cv2
import numpy as np


def draw_stoma(canvas, x, y, major, minor, angle_deg, dark=0.35):
    """angle_deg 与检测器约定一致：长轴 = 竖直方向逆时针旋转 angle_deg。"""
    ang = 90.0 - angle_deg  # cv2.ellipse 角度：第一轴相对 x 轴
    c = (int(round(x)), int(round(y)))
    a, b = int(major / 2), int(minor / 2)
    cv2.ellipse(canvas, c, (a, b), ang, 0, 360, 0.93, -1, cv2.LINE_AA)            # 亮保卫细胞
    cv2.ellipse(canvas, c, (a, b), ang, 0, 360, dark, 3, cv2.LINE_AA)             # 暗外壁
    cv2.ellipse(canvas, c, (int(a * 0.72), max(2, int(b * 0.28))), ang, 0, 360, dark + 0.1, -1, cv2.LINE_AA)  # 裂隙


def make_synthetic(w=640, h=480, n=22, seed=0, major=(30, 40), veins=True, walls=True):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 0.82, np.float32)
    if walls:  # 表皮细胞壁：随机短弧
        for _ in range(160):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            ax = int(rng.uniform(8, 25))
            cv2.ellipse(img, (int(cx), int(cy)), (ax, int(ax * rng.uniform(0.2, 0.6))), rng.uniform(0, 180),
                        0, rng.uniform(40, 120), 0.55, 2, cv2.LINE_AA)
    if veins:  # 叶脉：右下角一组平行长条纹
        for k in range(7):
            off = 12 * k
            cv2.line(img, (w // 2 + off, h), (w + off, h // 2 + 40), 0.5, 3, cv2.LINE_AA)
    pts = []
    tries = 0
    while len(pts) < n and tries < 5000:
        tries += 1
        L = rng.uniform(*major)
        x, y = rng.uniform(L, w - L), rng.uniform(L, h - L)
        if veins and (x - w / 2) > (h - y) * (w / 2) / (h / 2 - 40) - 30:  # 避开叶脉区
            continue
        if all(np.hypot(x - px, y - py) > 1.2 * L for px, py, _ in pts):
            pts.append((x, y, L))
    for x, y, L in pts:
        draw_stoma(img, x, y, L, L * rng.uniform(0.62, 0.8), rng.uniform(0, 180))
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    yy, xx = np.mgrid[0:h, 0:w]
    illum = 0.55 + 0.45 * np.exp(-(((xx - w * 0.3) / (w * 0.6)) ** 2 + ((yy - h * 0.4) / (h * 0.6)) ** 2))
    img = img * illum + rng.normal(0, 0.02, img.shape)
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img8, np.array([(x, y) for x, y, _ in pts]), np.array([L for _, _, L in pts])
