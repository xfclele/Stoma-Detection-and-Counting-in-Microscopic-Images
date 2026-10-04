"""阶段二 · 数据集构建：点标注 → YOLO 框标注 → 重叠切片。

为什么先做"点标注"？
    气孔形状规整、尺度集中，一个中心点 + 本图的典型长轴就足以生成可靠的框：
        box = (x - L/2, y - L/2, L, L)   （L = 该气孔长轴，缺省时用全图中位数）
    点标注的速度约为画框的 3–5 倍，而且可直接由阶段一的检测结果"预标注 → 人工校正"。

为什么训练时必须切片？
    原图 2000×1500 中气孔仅 ~35 px。若整图缩放到 640，气孔变为 ~11 px、裂隙完全消失
    （这正是需求中禁止的"粗暴缩小"）。因此按网络输入尺寸在 *原始分辨率* 上切成
    640×640（重叠 20%）的小块训练，推理时用 SAHI 同样切片，训练/推理尺度严格一致。
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


@dataclass
class PointAnnotation:
    image_path: str
    points: np.ndarray              # (N, 2) 气孔中心 x, y
    majors: Optional[np.ndarray]    # (N,) 长轴像素，可为 None
    mask_path: Optional[str] = None  # 可选有效区域掩膜（255=已标注的有效区域；0=未标注/剔除区，训练时被遮蔽）

    @classmethod
    def from_json(cls, json_path: str, image_dir: str) -> "PointAnnotation":
        """读取点标注 JSON（StomataCounter.export_points_json 的输出或人工标注，整图完整标注）。"""
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        pts = np.asarray(d.get("points", []), float).reshape(-1, 2)
        majors = np.asarray(d["major_px"], float) if "major_px" in d else None
        mask = os.path.join(os.path.dirname(json_path), "..", d["mask"]) if d.get("mask") else None
        return cls(os.path.join(image_dir, d["image"]), pts, majors, mask)


def points_to_boxes(points: np.ndarray, majors: Optional[np.ndarray] = None, default_major: Optional[float] = None,
                    pad: float = 1.1) -> np.ndarray:
    """中心点 → 正方形框 (x0, y0, x1, y1)。气孔朝向任意，用长轴×pad 的正方形可覆盖任意朝向的椭圆。"""
    if majors is None or len(majors) != len(points):
        L = np.full(len(points), default_major if default_major else 34.0)
    else:
        L = np.where(majors > 0, majors, np.median(majors[majors > 0]) if np.any(majors > 0) else 34.0)
    h = 0.5 * L * pad
    return np.stack([points[:, 0] - h, points[:, 1] - h, points[:, 0] + h, points[:, 1] + h], 1)


def iter_tiles(width: int, height: int, tile: int = 640, overlap: float = 0.2):
    """生成覆盖整图的重叠切片窗口 (x0, y0, x1, y1)；最后一行/列贴边对齐，保证无遗漏。"""
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(width - tile, 0) + 1, step))
    ys = list(range(0, max(height - tile, 0) + 1, step))
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, min(x0 + tile, width), min(y0 + tile, height)


def clip_boxes_to_tile(boxes: np.ndarray, win, min_visible: float = 0.6):
    """把全图框裁剪到切片内；可见面积 < min_visible 的截断目标丢弃（避免把半个气孔当正样本）。"""
    x0, y0, x1, y1 = win
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    cb = boxes.copy()
    cb[:, [0, 2]] = np.clip(cb[:, [0, 2]], x0, x1)
    cb[:, [1, 3]] = np.clip(cb[:, [1, 3]], y0, y1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    vis = (cb[:, 2] - cb[:, 0]).clip(0) * (cb[:, 3] - cb[:, 1]).clip(0)
    keep = vis >= min_visible * area
    cb = cb[keep]
    cb[:, [0, 2]] -= x0
    cb[:, [1, 3]] -= y0
    return cb


def boxes_to_yolo_lines(boxes: np.ndarray, w: int, h: int, cls: int = 0) -> list[str]:
    out = []
    for bx0, by0, bx1, by1 in boxes:
        cx, cy = (bx0 + bx1) / 2 / w, (by0 + by1) / 2 / h
        bw, bh = (bx1 - bx0) / w, (by1 - by0) / h
        out.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


def build_yolo_dataset(annotations: Sequence[PointAnnotation], out_dir: str, tile: int = 640, overlap: float = 0.2,
                       val_fraction: float = 0.2, min_visible: float = 0.6, keep_empty: float = 0.3,
                       seed: int = 0, split_by: str = "image", val_images: Optional[Sequence[str]] = None,
                       min_valid_frac: float = 0.3) -> str:
    """把若干整图点标注转换为 Ultralytics YOLO 切片数据集，返回 data.yaml 路径。

    split_by="image"：按整图划分训练/验证（推荐；同一张图的重叠切片若分到两边会导致验证集信息泄漏）。
    当只有 1 张图时自动退化为按空间划分：最右一列切片作验证，与之重叠的训练切片丢弃。
    keep_empty：无目标切片的保留比例（适量负样本可压制叶脉/细胞壁误检）。
    val_images：显式指定验证集图像文件名（覆盖 val_fraction 的随机划分）。
    有掩膜时：掩膜外（未标注区域）的像素用切片有效区的中位灰度填充，避免"有气孔却没标签"被当作负样本；
    有效面积 < min_valid_frac 的切片丢弃；中心落在掩膜外的框丢弃。
    """
    rng = random.Random(seed)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    anns = list(annotations)
    n_val_img = int(round(len(anns) * val_fraction)) if len(anns) > 1 else 0
    idx = list(range(len(anns)))
    rng.shuffle(idx)
    val_imgs = set(idx[:n_val_img])
    if val_images is not None:
        vs = set(val_images)
        val_imgs = {i for i, a in enumerate(anns) if os.path.basename(a.image_path) in vs}

    for i, ann in enumerate(anns):
        img = cv2.imdecode(np.fromfile(ann.image_path, np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        boxes = points_to_boxes(ann.points, ann.majors)
        mask = None
        if ann.mask_path:
            mask = cv2.imread(ann.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (H, W):
                raise ValueError(f"掩膜缺失或尺寸不符: {ann.mask_path}")
            if len(boxes):
                cx = np.clip(ann.points[:, 0].astype(int), 0, W - 1)
                cy = np.clip(ann.points[:, 1].astype(int), 0, H - 1)
                boxes = boxes[mask[cy, cx] > 0]
        stem = os.path.splitext(os.path.basename(ann.image_path))[0]
        wins = list(iter_tiles(W, H, tile, overlap))
        # 单图：最右一列切片作验证，与其有重叠的训练切片丢弃（防止重叠像素泄漏）
        val_x0 = max(w[0] for w in wins) if len(wins) > 1 else None
        for win in wins:
            x0, y0, x1, y1 = win
            if len(anns) == 1 and val_x0 is not None and val_x0 > 0:
                if x0 >= val_x0:
                    split = "val"
                elif x1 <= val_x0:
                    split = "train"
                else:
                    continue
            elif len(anns) == 1:
                split = "train"
            else:
                split = "val" if i in val_imgs else "train"
            crop = img[y0:y1, x0:x1].copy()
            if mask is not None:
                m = mask[y0:y1, x0:x1] > 0
                if m.mean() < min_valid_frac:
                    continue
                if not m.all():
                    crop[~m] = np.median(crop[m], axis=0).astype(np.uint8)
            tb = clip_boxes_to_tile(boxes, win, min_visible)
            if len(tb) == 0 and rng.random() > keep_empty:
                continue
            name = f"{stem}_{x0}_{y0}"
            cv2.imwrite(os.path.join(out_dir, "images", split, name + ".png"), crop)
            with open(os.path.join(out_dir, "labels", split, name + ".txt"), "w") as f:
                f.write("\n".join(boxes_to_yolo_lines(tb, x1 - x0, y1 - y0)))

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\nnames:\n  0: stoma\n")
    return yaml_path


def load_point_annotations(json_paths: Iterable[str], image_dir: str) -> list[PointAnnotation]:
    return [PointAnnotation.from_json(p, image_dir) for p in json_paths]
