"""多尺度测试时增强（TTA）：在 1.0× 和放大后的整图上分别做切片推理，坐标还原后融合。

放大推理的目的是让 <40 px 的小气孔在网络里显得更大。融合方式：

* ``nms``：所有尺度的框合在一起，按分数做 IoS-NMS；
* ``wbf``：跨尺度聚类（IoS ≥ thr），框按分数加权平均。分数取各尺度平均，
  缺席的尺度记 0，所以多尺度一致的目标得分更高；
* ``small``：以 1.0× 结果为主，只从其他尺度补充"原尺度长边 < small_px 且与 1.0× 结果不重叠"的框。

数组格式统一为 (N, 5) = [x0, y0, x1, y1, score]（原图坐标）。
"""
from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np

from .sliced_inference import greedy_nms_ios, ios_matrix


def predict_at_scale(predict: Callable[[np.ndarray], list], image: np.ndarray, scale: float) -> np.ndarray:
    """把整图缩放 scale 倍后用 predict（如 SlicedPredictor）推理，返回原图坐标下的 (N, 5) 数组。"""
    img = image if scale == 1.0 else cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    d = predict(img)
    a = np.array([[x.x0, x.y0, x.x1, x.y1, x.score] for x in d], np.float32).reshape(-1, 5)
    a[:, :4] /= scale
    return a


def fuse_scales(preds: Sequence[np.ndarray], mode: str = "wbf", thr: float = 0.5, small_px: float = 40.0) -> np.ndarray:
    """融合多个尺度的预测；preds[0] 视为基准尺度（1.0×）。"""
    preds = [np.asarray(p, np.float32).reshape(-1, 5) for p in preds]
    if mode == "small":
        base = preds[0]
        add = []
        for p in preds[1:]:
            if not len(p):
                continue
            side = np.maximum(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
            p = p[side < small_px]
            ref = np.vstack([base] + add) if add else base
            if len(p) and len(ref):
                p = p[ios_matrix(p[:, :4], ref[:, :4]).max(1) < thr]
            if len(p):
                add.append(p)
        return np.vstack([base] + add) if add else base

    allp = np.vstack(preds)
    if not len(allp):
        return allp
    if mode == "nms":
        return allp[greedy_nms_ios(allp[:, :4], allp[:, 4], thr)]

    # wbf：贪心聚类，每个簇每个尺度最多取一个框
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
