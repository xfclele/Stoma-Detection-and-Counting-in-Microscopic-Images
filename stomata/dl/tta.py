"""多尺度测试时增强（TTA）：在 1.0× 和放大后的整图上分别做切片推理，坐标还原后融合。

放大推理的目的是让 <40 px 的小气孔在网络里显得更大。融合方式：

* ``nms``：所有尺度的框合在一起，按分数做 IoS-NMS；
* ``wbf``：跨尺度聚类（IoS ≥ thr），框按分数加权平均。分数取各尺度平均，
  缺席的尺度记 0，所以多尺度一致的目标得分更高；
* ``small``：以 1.0× 结果为主，只用其他尺度中"原尺度长边 < small_px"的框：与 1.0× 框重叠时
  把该框分数提升为两者最大值，不重叠时作为新框补充。

数组格式统一为 (N, 5) = [x0, y0, x1, y1, score]（原图坐标）。

English
-------
Multi-scale test-time augmentation (TTA): sliced inference on the 1.0× image and on upscaled
copies, mapped back to original coordinates and fused.

Upscaling makes < 40 px stomata look larger to the network. Fusion modes:

* ``nms``: pool boxes from all scales and apply IoS-NMS by score;
* ``wbf``: cluster across scales (IoS ≥ thr) and average boxes weighted by score. The score is the
  mean over scales with missing scales counted as 0, so objects found at several scales score higher;
* ``small``: keep the 1.0× result and use only other-scale boxes whose long side at original scale is
  < small_px: if one overlaps a 1.0× box, that box's score becomes the maximum of the two; otherwise
  it is added as a new box.

Arrays are (N, 5) = [x0, y0, x1, y1, score] in original-image coordinates.
See docs/TTA_COMPARISON.md: not used by default (no F1 gain).
"""
from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np

from .sliced_inference import greedy_nms_ios, ios_matrix


def predict_at_scale(predict: Callable[[np.ndarray], list], image: np.ndarray, scale: float) -> np.ndarray:
    """把整图缩放 scale 倍后用 predict（如 SlicedPredictor）推理，返回原图坐标下的 (N, 5) 数组。
    Resize the image by `scale`, run `predict` (e.g. SlicedPredictor) and return an (N, 5) array in original coordinates."""
    img = image if scale == 1.0 else cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    d = predict(img)
    a = np.array([[x.x0, x.y0, x.x1, x.y1, x.score] for x in d], np.float32).reshape(-1, 5)
    a[:, :4] /= scale
    return a


def fuse_scales(preds: Sequence[np.ndarray], mode: str = "wbf", thr: float = 0.5, small_px: float = 40.0) -> np.ndarray:
    """融合多个尺度的预测；preds[0] 视为基准尺度（1.0×）。
    Fuse predictions from several scales; preds[0] is the reference (1.0×) scale."""
    preds = [np.asarray(p, np.float32).reshape(-1, 5) for p in preds]
    if mode == "small":
        # 只对小框做跨尺度融合：与基准框重叠 → 基准框分数取两者最大值；不重叠 → 作为新框补充 / fuse only small boxes: overlap with a base box → max score; otherwise → add as a new box
        base = preds[0].copy()
        add = []
        for p in preds[1:]:
            if not len(p):
                continue
            side = np.maximum(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
            p = p[side < small_px]
            if not len(p):
                continue
            if len(base):
                M = ios_matrix(p[:, :4], base[:, :4])
                j = M.argmax(1)
                hit = M[np.arange(len(p)), j] >= thr
                np.maximum.at(base[:, 4], j[hit], p[hit, 4])
                p = p[~hit]
            if len(p):
                add.append(p[greedy_nms_ios(p[:, :4], p[:, 4], thr)])
        return np.vstack([base] + add) if add else base

    allp = np.vstack(preds)
    if not len(allp):
        return allp
    if mode == "nms":
        return allp[greedy_nms_ios(allp[:, :4], allp[:, 4], thr)]

    # wbf：贪心聚类，每个簇每个尺度最多取一个框 / wbf: greedy clustering, at most one box per scale in each cluster
    src = np.concatenate([np.full(len(p), k) for k, p in enumerate(preds)])
    order = np.argsort(-allp[:, 4])
    M = ios_matrix(allp[:, :4], allp[:, :4])
    used = np.zeros(len(allp), bool)
    out = []
    for i in order:
        if used[i]:
            continue
        members = [i]
        used[i] = True
        taken = {src[i]}
        for j in order:
            if not used[j] and src[j] not in taken and M[i, j] >= thr:
                members.append(j)
                used[j] = True
                taken.add(src[j])
        m = allp[members]
        w = m[:, 4:5]
        box = (m[:, :4] * w).sum(0) / w.sum()
        out.append([*box, m[:, 4].sum() / len(preds)])
    return np.array(out, np.float32).reshape(-1, 5)
