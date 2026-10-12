#!/usr/bin/env python
"""在 Stomata_Enhanced 人工标注验证集上做整图切片推理评估，并保存全部预测供误差分析。

    python scripts/eval_human.py --weights models/xxx/best.pt -o outputs/eval_xxx [--exclude 118.2]

输出：
    per_image.csv   每图：真值数、预测数、TP/FP/FN、P/R/F1、计数误差
    preds.npz       每图预测框 (x0,y0,x1,y1,conf)，可离线改阈值重算（--from-npz）
    summary.json    总体指标 + 置信度阈值扫描 + 计数误差统计
匹配：中心点一对一匈牙利匹配，容差 = tol_factor × 该真值框长边（默认 0.5，约 25 px）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata.dl import load_yolo_polygon_labels  # noqa: E402
from stomata.evaluation import match_points, precision_recall_f1  # noqa: E402


def load_gt(root):
    out = {}
    for ip in sorted(glob.glob(os.path.join(root, "images", "*.jpg"))):
        s = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(root, "labels", s + ".txt")
        if os.path.exists(lp):
            out[s] = (ip, load_yolo_polygon_labels(lp, ip, 2560, 1920))
    return out


def evaluate(gt, preds, conf, tol_factor):
    rows, tot = [], np.zeros(3, int)
    for s, (ip, ann) in gt.items():
        P = preds[s]
        P = P[P[:, 4] >= conf] if len(P) else P
        C = np.stack([(P[:, 0] + P[:, 2]) / 2, (P[:, 1] + P[:, 3]) / 2], 1) if len(P) else np.zeros((0, 2))
        tol = tol_factor * float(np.median(ann.majors)) if len(ann.majors) else 25.0
        m = match_points(C, ann.points, tol=tol)
        tot += [m.tp, m.fp, m.fn]
        rows.append(dict(image=s, gt=len(ann.points), pred=len(C), tp=m.tp, fp=m.fp, fn=m.fn,
                         precision=round(m.precision, 4), recall=round(m.recall, 4), f1=round(m.f1, 4),
                         count_err=len(C) - len(ann.points),
                         rel_count_err=round((len(C) - len(ann.points)) / max(len(ann.points), 1), 4)))
    p, r, f = precision_recall_f1(*tot)
    ce = np.array([x["count_err"] for x in rows]); rce = np.array([x["rel_count_err"] for x in rows])
    return rows, dict(conf=conf, tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), precision=round(p, 4),
                      recall=round(r, 4), f1=round(f, 4), count_mae=round(float(np.abs(ce).mean()), 2),
                      count_mape=round(float(np.abs(rce).mean() * 100), 2), count_bias=round(float(ce.mean()), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--root", default="data/samples/pre-trainning/Stomata_Enhanced/val")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="排除的图像（例如在该模型训练集中出现过的）")
    ap.add_argument("--tol-factor", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25, help="报告用阈值；扫描另见 summary.json")
    ap.add_argument("--from-npz", action="store_true", help="复用已保存的 preds.npz，不重新推理")
    ap.add_argument("--backend", choices=["sliced", "sahi"], default="sliced",
                    help="sliced = 本仓库 SlicedPredictor（中心归属去重）；sahi = 官方 SAHI get_sliced_prediction")
    ap.add_argument("--sahi-postprocess", default="GREEDYNMM", help="SAHI 合并方式：GREEDYNMM / NMM / NMS")
    ap.add_argument("--sahi-metric", default="IOS", help="SAHI 匹配度量：IOS / IOU")
    ap.add_argument("--sahi-thr", type=float, default=0.5, help="SAHI 匹配阈值")
    ap.add_argument("--infer-conf", type=float, default=0.05, help="推理时的最低置信度（之后再做阈值扫描）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gt = {k: v for k, v in load_gt(a.root).items() if k not in set(a.exclude)}

    npz = os.path.join(a.out, "preds.npz")
    if a.from_npz:
        z = np.load(npz)
        preds = {k: z[k] for k in z.files}
    else:
        from stomata.dl.sliced_inference import (SlicedPredictor, load_sahi_model, predict_with_sahi,
                                                 ultralytics_tile_predictor)
        if a.backend == "sahi":
            model = load_sahi_model(a.weights, conf=a.infer_conf, device="cpu", image_size=640)
            run = lambda img: predict_with_sahi(img, tile=640, overlap=0.2, postprocess=a.sahi_postprocess,  # noqa: E731
                                                match_metric=a.sahi_metric, match_threshold=a.sahi_thr, model=model)
        else:
            run = SlicedPredictor(ultralytics_tile_predictor(a.weights, conf=a.infer_conf, device="cpu"), 640, 0.2, "center")
        preds, times = {}, []
        for i, (s, (ip, _)) in enumerate(gt.items()):
            img = cv2.imread(ip)
            t0 = time.perf_counter()
            d = run(img)
            times.append(time.perf_counter() - t0)
            preds[s] = np.array([[x.x0, x.y0, x.x1, x.y1, x.score] for x in d], np.float32).reshape(-1, 5)
            print(f"[{i + 1}/{len(gt)}] {s}: {len(preds[s])} dets  {times[-1]:.1f}s", flush=True)
        t = np.array(times[1:] if len(times) > 1 else times)          # 第一张含模型预热，不计
        timing = dict(backend=a.backend, n=len(t), mean_s=round(float(t.mean()), 3),
                      median_s=round(float(np.median(t)), 3), total_s=round(float(np.sum(times)), 1))
        json.dump(timing, open(os.path.join(a.out, "timing.json"), "w"), indent=1)
        print("timing", json.dumps(timing))
        np.savez_compressed(npz, **preds)

    rows, summ = evaluate(gt, preds, a.conf, a.tol_factor)
    sweep = [evaluate(gt, preds, c, a.tol_factor)[1] for c in np.round(np.arange(0.1, 0.81, 0.05), 2)]
    best = max(sweep, key=lambda x: x["f1"])
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(dict(weights=a.weights, n_images=len(gt), excluded=a.exclude, tol_factor=a.tol_factor,
                   at_conf=summ, best_f1=best, sweep=sweep), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("conf=%.2f" % a.conf, json.dumps(summ))
    print("best  ", json.dumps(best))


if __name__ == "__main__":
    main()
