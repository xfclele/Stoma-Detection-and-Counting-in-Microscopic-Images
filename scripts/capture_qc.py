#!/usr/bin/env python
"""拍摄后质检：检查新图是否与训练图条件一致，不合格的当场提示重拍。
Post-capture quality check: tells whether new images match the training conditions, so bad ones can be
retaken while the sample is still under the microscope.

    python scripts/capture_qc.py new_images/ -o qc.csv                 # 含模型检出数与气孔尺寸 / with model count and stoma size
    python scripts/capture_qc.py new_images/ --no-model                # 只看图像指标，几秒一张 / image metrics only, seconds per image
    python scripts/capture_qc.py --calibrate data/samples/pre-trainning/Stomata_Enhanced/train/images

检查项（阈值见 stomata/quality.py: QC_THRESHOLDS，依据见 docs/IMAGING_GUIDE.md）：
Checks (thresholds in stomata/quality.py: QC_THRESHOLDS; rationale in docs/IMAGING_GUIDE.md):
    · 清晰度、对焦均匀性 / sharpness and focus uniformity
    · 对比度、纯黑/纯白像素比例 / contrast and fraction of pure black/white pixels
    · 分辨率是否为训练用的 2560×1920 / whether the resolution is the training 2560×1920
    · 模型检出数与预测框长边中位数（判断放大倍率） / model count and median predicted box long side (magnification)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata import StomataCounter  # noqa: E402
from stomata.quality import CHECK, OK, RETAKE, assess, image_quality  # noqa: E402

TRAIN_SIZE = (2560, 1920)
METRICS = ["sharpness", "sharpness_low", "focus_uniformity", "contrast", "clipped", "dynamic_range"]


def calibrate(folder):
    rows = [image_quality(cv2.imread(p)) for p in StomataCounter.list_images(folder)]
    print(f"{len(rows)} 张参考图 / reference images")
    print("指标 metric".ljust(20) + "".join(f"{'p%d' % q:>9}" for q in (5, 10, 50, 95)))
    for m in METRICS:
        v = np.array([r[m] for r in rows])
        print(m.ljust(20) + "".join(f"{np.percentile(v, q):9.3f}" for q in (5, 10, 50, 95)))
    print("把 p5 / p10 填入 QC_THRESHOLDS 的 *_retake / *_check / Put p5 / p10 into the *_retake / *_check entries of QC_THRESHOLDS")


def main():
    ap = argparse.ArgumentParser(description="拍摄后质检 / post-capture quality check")
    ap.add_argument("images", nargs="?", help="图像文件或文件夹 / image file or folder")
    ap.add_argument("--weights", default="models/stomata_yolov8n_p2_human_v2/best.pt")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--no-model", action="store_true", help="不跑模型，跳过检出数和尺寸检查 / skip the model (no count or size check)")
    ap.add_argument("-o", "--out", help="结果 CSV / output CSV")
    ap.add_argument("--calibrate", metavar="DIR", help="从参考图（如训练图）计算阈值分位数 / compute threshold percentiles from reference images (e.g. training images)")
    a = ap.parse_args()
    if a.calibrate:
        return calibrate(a.calibrate)
    if not a.images:
        ap.error("需要 images 参数 / images is required")

    predictor = None
    if not a.no_model:
        if not os.path.exists(a.weights):
            sys.exit(f"找不到权重 / weights not found: {a.weights}（可加 --no-model / or use --no-model）")
        from stomata.dl.sliced_inference import SlicedPredictor, ultralytics_tile_predictor
        predictor = SlicedPredictor(ultralytics_tile_predictor(a.weights, conf=a.conf, device="cpu"), 640, 0.2, "center")

    rows, tally = [], {RETAKE: 0, CHECK: 0, OK: 0}
    for path in StomataCounter.list_images(a.images):
        img = cv2.imread(path)
        if img is None:
            print(f"{os.path.basename(path)}: 无法读取 / cannot read")
            continue
        q = image_quality(img)
        if predictor is not None:
            dets = predictor(img)
            q["n_stomata"] = len(dets)
            q["size_px"] = float(np.median([max(d.x1 - d.x0, d.y1 - d.y0) for d in dets])) if dets else None
        verdict, reasons = assess(q)
        h, w = img.shape[:2]
        if (w, h) != TRAIN_SIZE:
            reasons.append(f"分辨率 {w}×{h} 与训练图 2560×1920 不同，请确认放大倍率与像素尺寸 / "
                           f"resolution {w}×{h} differs from the training 2560×1920; check magnification and pixel size")
            verdict = CHECK if verdict == OK else verdict
        tally[verdict] += 1
        name = os.path.basename(path)
        extra = f"  n={q['n_stomata']}  size={q['size_px'] or 0:.0f}px" if predictor is not None else ""
        print(f"{verdict:<12} {name}  sharp={q['sharpness']:.3f}/{q['sharpness_low']:.3f}  "
              f"contrast={q['contrast']:.1f}  clipped={q['clipped']:.1%}{extra}")
        for r in reasons:
            print(f"    - {r}")
        rows.append({"image": name, "verdict": verdict, **{m: round(q[m], 4) for m in METRICS},
                     "n_stomata": q.get("n_stomata", ""), "size_px": round(q["size_px"], 1) if q.get("size_px") else "",
                     "reasons": " | ".join(reasons)})

    print(f"\n共 {len(rows)} 张 / {len(rows)} images: " + ", ".join(f"{k} {v}" for k, v in tally.items()))
    if a.out and rows:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print("CSV:", a.out)


if __name__ == "__main__":
    main()
