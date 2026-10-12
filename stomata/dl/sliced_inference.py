"""阶段二 · 切片辅助推理 (SAHI, Slicing Aided Hyper Inference)。
Stage 2 · sliced inference (SAHI, Slicing Aided Hyper Inference).

核心机制 / Mechanism
--------------------
1. 把大图切成与训练一致的 640×640 重叠切片（overlap 20%），每片在 *原始分辨率* 上推理，
   ~35 px 的气孔在网络眼中仍是 ~35 px（而不是整图缩放后的 ~11 px）；
   Cut the large image into 640×640 overlapping tiles (20% overlap) as in training and run each at
   *original resolution*, so a ~35 px stoma is still ~35 px to the network (not ~11 px after downscaling).
2. 可选再做一次整图低分辨率推理（standard prediction），补充跨越多片的大目标；
   Optionally add a low-resolution whole-image pass (standard prediction) for large objects spanning tiles.
3. 把各片预测平移回全图坐标，用 NMS / NMM（Non-Maximum Merging）按
   IoS (Intersection over Smaller) 合并重叠区内的重复框 —— 被切片边界截断的半个气孔
   与相邻片中完整的该气孔 IoS 很高，会被合并为一个，从而消除边界重复计数。
   Shift tile predictions back to image coordinates and merge duplicates in overlaps with NMS / NMM
   (Non-Maximum Merging) using IoS (Intersection over Smaller): half a stoma cut by a tile border has a
   high IoS with the complete stoma in the neighbouring tile, so the two are merged and the border
   duplicate disappears.

本模块提供两条路径：
This module offers two paths:
    · SlicedPredictor            —— 无 sahi 依赖的等价实现（任何 "tile → boxes" 的检测器都可接入），
                                     额外支持"中心归属"去重：每个框只由其中心所在切片的核心区负责。
                                     默认方案（见 docs/SAHI_COMPARISON.md）。
                                     dependency-free equivalent (any "tile → boxes" detector plugs in), with
                                     extra "centre ownership" de-duplication: each box belongs only to the
                                     tile whose core contains its centre. Default (see docs/SAHI_COMPARISON.md).
    · predict_with_sahi()        —— 调用官方 sahi 库（可选）
                                     calls the official sahi library (optional)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .dataset import iter_tiles


@dataclass
class BoxDetection:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def center(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def major_px(self):
        """轴对齐框估计长轴：正方形框（点标注生成）时 ≈ 长轴×pad；OBB 模型请直接使用其长边。
        Major axis from an axis-aligned box: ≈ major×pad for point-derived squares; for OBB models use the long side directly."""
        return max(self.x1 - self.x0, self.y1 - self.y0)


def ios_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoS = 交集面积 / 较小框面积。对"被截断的半框 vs 完整框"比 IoU 更敏感。
    IoS = intersection / area of the smaller box; more sensitive than IoU for "truncated half box vs complete box"."""
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (ix1 - ix0).clip(0) * (iy1 - iy0).clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(np.minimum(area_a[:, None], area_b[None, :]), 1e-6)


def greedy_nms_ios(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(boxes), bool)
    M = ios_matrix(boxes, boxes) if len(boxes) else np.zeros((0, 0))
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= M[i] > thr
    return np.asarray(keep, int)


class SlicedPredictor:
    """无外部依赖的 SAHI 等价实现。
    Dependency-free equivalent of SAHI.

    predict_tile: Callable[[np.ndarray], np.ndarray]
        输入切片 (h, w, 3)，返回 (K, 5) 数组 [x0, y0, x1, y1, score]（切片坐标）。
        例如包装 Ultralytics::
        Takes a tile (h, w, 3) and returns a (K, 5) array [x0, y0, x1, y1, score] in tile coordinates,
        e.g. wrapping Ultralytics::

            model = YOLO("runs/detect/stoma/weights/best.pt")
            def predict_tile(tile):
                r = model.predict(tile, imgsz=640, conf=0.25, verbose=False)[0]
                return np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]])
    """

    def __init__(self, predict_tile: Callable[[np.ndarray], np.ndarray], tile: int = 640, overlap: float = 0.2,
                 merge: str = "center", ios_thr: float = 0.5):
        assert merge in ("center", "nms")
        self.predict_tile, self.tile, self.overlap = predict_tile, tile, overlap
        self.merge, self.ios_thr = merge, ios_thr

    def __call__(self, image: np.ndarray) -> list[BoxDetection]:
        H, W = image.shape[:2]
        wins = list(iter_tiles(W, H, self.tile, self.overlap))
        all_boxes = []
        for (x0, y0, x1, y1) in wins:
            pred = np.asarray(self.predict_tile(image[y0:y1, x0:x1]), float).reshape(-1, 5)
            if not len(pred):
                continue
            pred[:, [0, 2]] += x0
            pred[:, [1, 3]] += y0
            if self.merge == "center":
                # 中心归属：只保留中心落在本切片"核心区"的框。核心区 = 切片去掉与邻片重叠的一半，
                # 相邻切片的核心区恰好无缝、不重叠地铺满整图 ⇒ 每个目标只被计数一次。
                # Centre ownership: keep only boxes whose centre is in this tile's core (the tile minus half
                # of each overlap). The cores tile the image without gaps or overlaps ⇒ each object counts once.
                cx, cy = (pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2
                cx0, cy0, cx1, cy1 = self._core(x0, y0, x1, y1, wins)
                pred = pred[(cx >= cx0) & (cx < cx1) & (cy >= cy0) & (cy < cy1)]
            all_boxes.append(pred)
        if not all_boxes:
            return []
        P = np.vstack(all_boxes)
        keep = greedy_nms_ios(P[:, :4], P[:, 4], self.ios_thr)  # 两种模式都再做一次全局 IoS-NMS 兜底 / both modes finish with a global IoS-NMS safety net
        return [BoxDetection(*P[i]) for i in keep]

    @staticmethod
    def _core(x0, y0, x1, y1, wins):
        """该切片的核心区：与相邻切片重叠部分各分一半。
        Core of this tile: each overlap with a neighbour is split in half."""
        xs = sorted({w[0] for w in wins})
        ys = sorted({w[1] for w in wins})
        xe = {w[0]: w[2] for w in wins}
        ye = {w[1]: w[3] for w in wins}
        i, j = xs.index(x0), ys.index(y0)
        cx0 = 0 if i == 0 else (xe[xs[i - 1]] + x0) / 2
        cx1 = float("inf") if i == len(xs) - 1 else (x1 + xs[i + 1]) / 2
        cy0 = 0 if j == 0 else (ye[ys[j - 1]] + y0) / 2
        cy1 = float("inf") if j == len(ys) - 1 else (y1 + ys[j + 1]) / 2
        return cx0, cy0, cx1, cy1


def load_sahi_model(weights: str, conf: float = 0.25, device: str = "cuda:0", image_size: int = 640):
    """加载 SAHI 检测模型；批量推理时只加载一次，传给 predict_with_sahi(model=...)。
    Load a SAHI detection model once for batch inference and pass it to predict_with_sahi(model=...)."""
    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=weights,
                                              confidence_threshold=conf, device=device, image_size=image_size)


def predict_with_sahi(image, weights: Optional[str] = None, tile: int = 640, overlap: float = 0.2, conf: float = 0.25,
                      device: str = "cuda:0", postprocess: str = "GREEDYNMM", match_metric: str = "IOS",
                      match_threshold: float = 0.5, standard_pred: bool = False, model=None,
                      force_postprocess: bool = True) -> list[BoxDetection]:
    """官方 SAHI 路径（pip install sahi ultralytics）。image 可为路径或 BGR 数组。
    Official SAHI path (pip install sahi ultralytics); image may be a path or a BGR array."""
    from sahi.predict import get_sliced_prediction

    if model is None:
        model = load_sahi_model(weights, conf, device)
    if isinstance(image, np.ndarray):
        image = image[:, :, ::-1].copy()                 # SAHI 期望 RGB 数组 / SAHI expects an RGB array
    res = get_sliced_prediction(
        image, model,
        slice_height=tile, slice_width=tile,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        perform_standard_pred=standard_pred,            # 气孔很小，整图预测通常只带来误检，默认关闭 / stomata are small; a whole-image pass mostly adds FPs, off by default
        postprocess_type=postprocess,                    # GREEDYNMM：合并而非简单丢弃重叠框 / GREEDYNMM merges overlapping boxes instead of discarding
        postprocess_match_metric=match_metric,           # IOS：对切片边界截断的半框更鲁棒 / IOS is more robust for half boxes cut by tile borders
        postprocess_match_threshold=match_threshold,
        force_postprocess_type=force_postprocess,        # 否则模型阈值低时 SAHI 会自动改用 NMS/IOU / otherwise SAHI silently switches to NMS/IOU at low model thresholds
        verbose=0,
    )
    out = []
    for p in res.object_prediction_list:
        b = p.bbox
        out.append(BoxDetection(b.minx, b.miny, b.maxx, b.maxy, float(p.score.value)))
    return out


def ultralytics_tile_predictor(weights: str, conf: float = 0.25, imgsz: int = 640, device: Optional[str] = None):
    """把 Ultralytics YOLO 权重包装为 SlicedPredictor 需要的 predict_tile 函数。
    Wrap Ultralytics YOLO weights as the predict_tile function expected by SlicedPredictor."""
    from ultralytics import YOLO

    model = YOLO(weights)

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        r = model.predict(tile, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5))
        return np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]])

    return predict_tile
