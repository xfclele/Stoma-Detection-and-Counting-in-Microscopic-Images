#!/usr/bin/env python
"""用切片推理在整图上评估预训练模型：与伪标签点做一对一匹配（P/R/F1），并对比计数。

    python scripts/eval_pretrain.py --weights runs/stomata/pretrain_v8n_p2/weights/best.pt \
        --images 137.1.jpg 49.2.jpg 573.2.jpg 9.3.jpg

只统计有效区域掩膜内的预测与标签（掩膜外本来就没有标注）。
注意：参考点本身是外部自动流程的伪标签，指标反映"与该流程的一致程度"，不等于对人工真值的精度。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata.dl.sliced_inference import SlicedPredictor, ultralytics_tile_predictor  # noqa: E402
from stomata.evaluation import match_points, precision_recall_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="pre-trainning/images 下的文件名")
    ap.add_argument("--src", default="data/samples/pre-trainning/images")
    ap.add_argument("--labels", default="data/pretraining_labels")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tol-factor", type=float, default=0.35, help="匹配容差 = factor × 该图长轴中位数")
    ap.add_argument("--out", default=None, help="可选：保存叠加图的目录")
    a = ap.parse_args()

    pred_fn = ultralytics_tile_predictor(a.weights, conf=a.conf, device="cpu")
    sp = SlicedPredictor(pred_fn, tile=640, overlap=0.2, merge="center")
    tot = dict(tp=0, fp=0, fn=0)
    for name in a.images:
        stem = os.path.splitext(name)[0]
        lab = json.load(open(os.path.join(a.labels, "labels", stem + ".json"), encoding="utf-8"))
        mask = cv2.imread(os.path.join(a.labels, "masks", stem + ".png"), cv2.IMREAD_GRAYSCALE) > 0
        img = cv2.imread(os.path.join(a.src, name))
        dets = sp(img)
        P = np.array([d.center for d in dets], float).reshape(-1, 2)
        H, W = mask.shape
        if len(P):
            inside = mask[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)]
            P = P[inside]
        G = np.asarray(lab["points"], float).reshape(-1, 2)
        tol = a.tol_factor * float(np.median(lab["major_px"]))
        m = match_points(P, G, tol=tol)
        tot["tp"] += m.tp; tot["fp"] += m.fp; tot["fn"] += m.fn
        print(f"{name:14s} pred={len(P):4d} ref={len(G):4d} tol={tol:4.1f}px  {m.summary()}")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            vis = img.copy()
            vis[~mask] = (vis[~mask] * 0.4).astype(np.uint8)
            for x, y in G:
                cv2.circle(vis, (int(x), int(y)), 4, (0, 200, 0), -1)
            for x, y in P:
                cv2.circle(vis, (int(x), int(y)), int(tol), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(a.out, stem + "_eval.jpg"), vis)
    p, r, f = precision_recall_f1(tot["tp"], tot["fp"], tot["fn"])
    print(f"ALL  TP={tot['tp']} FP={tot['fp']} FN={tot['fn']} | P={p:.3f} R={r:.3f} F1={f:.3f}")


if __name__ == "__main__":
    main()
