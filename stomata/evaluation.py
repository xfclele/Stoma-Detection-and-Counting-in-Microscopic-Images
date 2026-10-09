"""评估：Precision / Recall / F1（点匹配）。

气孔计数任务的评估以 *中心点* 为单位：
    · 预测点 p 与真值点 g 距离 ≤ tol 才可能匹配；
    · 用匈牙利算法 (linear_sum_assignment) 求最小代价一对一匹配，
      保证一个真值只能被一个预测"认领"——重复检测会被计为 FP，
      这正是检验"边界/分块重复计数"是否处理干净的关键；
    · 可选 ignore 点：模糊、半出视野、专家也无法判定的目标，
      落在其附近的预测既不算 TP 也不算 FP（COCO 中 iscrowd/ignore 的思想）。

    Precision = TP / (TP + FP)     Recall = TP / (TP + FN)
    F1 = 2PR / (P + R)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    count_pred: int
    count_gt: int
    mean_loc_error_px: float
    matches: list  # [(pred_idx, gt_idx, dist)]

    def summary(self) -> str:
        return (f"TP={self.tp} FP={self.fp} FN={self.fn} | P={self.precision:.3f} "
                f"R={self.recall:.3f} F1={self.f1:.3f} | pred={self.count_pred} gt={self.count_gt} "
                f"| loc_err={self.mean_loc_error_px:.1f}px")

    def to_dict(self):
        d = asdict(self)
        d.pop("matches")
        return d


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def match_points(
    pred: Sequence[Sequence[float]],
    gt: Sequence[Sequence[float]],
    tol: float = 12.0,
    ignore: Optional[Sequence[Sequence[float]]] = None,
) -> MatchResult:
    """一对一点匹配并计算指标。pred/gt/ignore 为 (x, y) 序列。"""
    P = np.asarray(pred, float).reshape(-1, 2)
    G = np.asarray(gt, float).reshape(-1, 2)
    matches = []
    if len(P) and len(G):
        D = np.hypot(P[:, None, 0] - G[None, :, 0], P[:, None, 1] - G[None, :, 1])
        cost = np.where(D <= tol, D, 1e6)
        ri, ci = linear_sum_assignment(cost)
        matches = [(int(i), int(j), float(D[i, j])) for i, j in zip(ri, ci) if D[i, j] <= tol]
    matched_p = {m[0] for m in matches}
    unmatched = [i for i in range(len(P)) if i not in matched_p]
    if ignore is not None and len(ignore) and unmatched:
        Ig = np.asarray(ignore, float).reshape(-1, 2)
        U = P[unmatched]
        d_ig = np.hypot(U[:, None, 0] - Ig[None, :, 0], U[:, None, 1] - Ig[None, :, 1]).min(1)
        unmatched = [u for u, d in zip(unmatched, d_ig) if d > tol]
    tp, fp, fn = len(matches), len(unmatched), len(G) - len(matches)
    p, r, f = precision_recall_f1(tp, fp, fn)
    err = float(np.mean([m[2] for m in matches])) if matches else float("nan")
    return MatchResult(tp, fp, fn, p, r, f, len(P), len(G), err, matches)


# ----------------------------------------------------------------------------
# 标注文件（JSON）：支持"部分区域标注"——只在若干矩形区域内完整标注
# ----------------------------------------------------------------------------
# {
#   "image": "leaf_epidermis_01.jpg",
#   "regions": [
#       {"box": [x0, y0, x1, y1],
#        "points": [[x, y], ...],        # 区域内全部气孔中心
#        "ignore": [[x, y], ...]}         # 可选：难以判定的目标
#   ]
# }
# 若不给 regions 而是给顶层 "points"，则视为整图完整标注。
def load_annotations(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_against_annotation(pred_xy: Iterable[Sequence[float]], ann: dict, tol: float = 12.0) -> MatchResult:
    """按标注文件评估；区域标注时只统计落在各区域内的预测与真值，最后合并。"""
    pred_xy = np.asarray(list(pred_xy), float).reshape(-1, 2)
    regions = ann.get("regions")
    if not regions:
        return match_points(pred_xy, ann.get("points", []), tol, ann.get("ignore"))
    tp = fp = fn = 0
    errs, n_pred, n_gt = [], 0, 0
    for reg in regions:
        x0, y0, x1, y1 = reg["box"]
        inside = (pred_xy[:, 0] >= x0) & (pred_xy[:, 0] < x1) & (pred_xy[:, 1] >= y0) & (pred_xy[:, 1] < y1)
        m = match_points(pred_xy[inside], reg.get("points", []), tol, reg.get("ignore"))
        tp, fp, fn = tp + m.tp, fp + m.fp, fn + m.fn
        n_pred, n_gt = n_pred + m.count_pred, n_gt + m.count_gt
        errs += [d for _, _, d in m.matches]
    p, r, f = precision_recall_f1(tp, fp, fn)
    return MatchResult(tp, fp, fn, p, r, f, n_pred, n_gt, float(np.mean(errs)) if errs else float("nan"), [])
