#!/usr/bin/env python
"""多尺度 TTA 实验：先按尺度推理并缓存，再离线融合；输出目录可直接用 eval_human.py --from-npz 评估。

    # 1) 在 1.25× 整图上切片推理（1.0× 可复用已有的 preds.npz）
    python scripts/tta_eval.py infer --scale 1.25 -o outputs/eval_human/tta/s1.25
    # 2) 融合
    python scripts/tta_eval.py fuse outputs/eval_human/sahi_cmp/sliced_ic0.05 outputs/eval_human/tta/s1.25 \
        --mode wbf -o outputs/eval_human/tta/wbf
    # 3) 评估
    python scripts/eval_human.py --from-npz -o outputs/eval_human/tta/wbf --exclude 118.2 221.2 --conf 0.40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.eval_human import load_gt  # noqa: E402
from stomata.dl.tta import fuse_scales, predict_at_scale  # noqa: E402


def cmd_infer(a):
    from stomata.dl.sliced_inference import SlicedPredictor, ultralytics_tile_predictor

    gt = {k: v for k, v in load_gt(a.root).items() if k not in set(a.exclude)}
    sp = SlicedPredictor(ultralytics_tile_predictor(a.weights, conf=a.infer_conf, device="cpu"), 640, 0.2, "center")
    os.makedirs(a.out, exist_ok=True)
    preds, times = {}, []
    for i, (s, (ip, _)) in enumerate(gt.items()):
        img = cv2.imread(ip)
        t0 = time.perf_counter()
        preds[s] = predict_at_scale(sp, img, a.scale)
        times.append(time.perf_counter() - t0)
        print(f"[{i + 1}/{len(gt)}] {s}: {len(preds[s])} dets  {times[-1]:.1f}s", flush=True)
    np.savez_compressed(os.path.join(a.out, "preds.npz"), **preds)
    t = np.array(times[1:])
    json.dump(dict(backend=f"sliced@{a.scale}x", n=len(t), mean_s=round(float(t.mean()), 3),
                   median_s=round(float(np.median(t)), 3), total_s=round(float(np.sum(times)), 1)),
              open(os.path.join(a.out, "timing.json"), "w"), indent=1)


def cmd_fuse(a):
    zs = [np.load(os.path.join(d, "preds.npz")) for d in a.runs]
    os.makedirs(a.out, exist_ok=True)
    fused = {k: fuse_scales([z[k] for z in zs], a.mode, a.thr, a.small_px) for k in zs[0].files}
    np.savez_compressed(os.path.join(a.out, "preds.npz"), **fused)
    print(f"fused {len(fused)} images -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("infer")
    p.add_argument("--weights", default="models/stomata_yolov8n_p2_human_v2/best.pt")
    p.add_argument("--root", default="data/samples/pre-trainning/Stomata_Enhanced/val")
    p.add_argument("--exclude", nargs="*", default=["118.2", "221.2"])
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--infer-conf", type=float, default=0.05)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_infer)
    p = sub.add_parser("fuse")
    p.add_argument("runs", nargs="+", help="各尺度的输出目录（第一个为 1.0× 基准）")
    p.add_argument("--mode", choices=["wbf", "nms", "small"], default="wbf")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--small-px", type=float, default=40.0)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_fuse)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
