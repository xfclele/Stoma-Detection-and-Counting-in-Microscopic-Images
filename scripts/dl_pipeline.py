#!/usr/bin/env python
"""阶段二命令行：预标注 → 构建切片数据集 → 训练 YOLOv8-P2 → SAHI 推理计数。
Stage 2 command line: pre-label → build tile dataset → train YOLOv8-P2 → sliced inference and counting.

    # 1) 用阶段一给每张图生成预标注（点 + 长轴），人工在标注工具中增删改后覆盖保存 / pre-label each image with stage 1 (points + major axis), then correct in an annotation tool and save
    python scripts/dl_pipeline.py prelabel images/ -o labels_points/

    # 2) 点标注 → YOLO 切片数据集（640×640，重叠 20%，按整图划分训练/验证） / point labels → YOLO tile dataset (640×640, 20% overlap, split by whole image)
    python scripts/dl_pipeline.py build labels_points/ --images images/ -o datasets/stomata

    # 3) 小样本微调 YOLOv8s-P2（需 GPU 与 `pip install ultralytics`） / few-shot fine-tuning of YOLOv8s-P2 (GPU and `pip install ultralytics`)
    python scripts/dl_pipeline.py train datasets/stomata/data.yaml --epochs 150

    # 4) 切片推理 + CSV（默认内置 SlicedPredictor；加 --sahi 改用官方 SAHI，需 `pip install sahi`） / sliced inference + CSV (built-in SlicedPredictor by default; --sahi for official SAHI, needs `pip install sahi`)
    python scripts/dl_pipeline.py predict images/ --weights runs/stomata/yolov8s_p2/weights/best.pt -o outputs_dl
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata import DetectorConfig, StomataCounter  # noqa: E402
from stomata.dl import SlicedPredictor, build_yolo_dataset, load_point_annotations  # noqa: E402


def cmd_prelabel(a):
    counter = StomataCounter(DetectorConfig(expected_major_px=a.expected_major, margin_exclusion=False))
    os.makedirs(a.out, exist_ok=True)
    for path in counter.list_images(a.images):
        rep = counter.process(counter.read_image(path), os.path.basename(path))
        counter.export_points_json(rep, os.path.join(a.out, os.path.splitext(rep.image)[0] + ".json"))
        print(f"{rep.image}: {rep.count} 个预标注点 / pre-label points")


def cmd_build(a):
    anns = load_point_annotations(sorted(glob.glob(os.path.join(a.labels, "*.json"))), a.images)
    y = build_yolo_dataset(anns, a.out, tile=a.tile, overlap=a.overlap, val_fraction=a.val_fraction)
    print("data.yaml:", y)


def cmd_train(a):
    from stomata.dl.train import train_yolo_p2
    train_yolo_p2(a.data, model_cfg=a.model, pretrained=a.pretrained, epochs=a.epochs, imgsz=a.tile,
                  batch=a.batch, freeze=a.freeze, device=a.device)


def cmd_predict(a):
    from stomata.dl.sliced_inference import predict_with_sahi, ultralytics_tile_predictor
    files = StomataCounter.list_images(a.images)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    predictor = None if a.sahi else SlicedPredictor(
        ultralytics_tile_predictor(a.weights, conf=a.conf, imgsz=a.tile, device=a.device), a.tile, a.overlap)
    for path in files:
        if predictor is None:
            dets = predict_with_sahi(path, a.weights, a.tile, a.overlap, a.conf, device=a.device or "cpu")
        else:
            dets = predictor(cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR))
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        kept = []
        for d in dets:
            if a.margin_exclusion and (d.x0 <= 0 or d.y0 <= 0 or d.x1 >= W - 1 or d.y1 >= H - 1):
                continue
            kept.append(d)
        for i, d in enumerate(kept, 1):
            cx, cy = d.center
            rows.append({"image": os.path.basename(path), "id": i, "x": round(cx, 1), "y": round(cy, 1),
                         "major_axis_px": round(d.major_px / a.box_pad, 2), "score": round(d.score, 4)})
            cv2.rectangle(img, (int(d.x0), int(d.y0)), (int(d.x1), int(d.y1)), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(a.out, os.path.splitext(os.path.basename(path))[0] + "_dl.jpg"), img)
        print(f"{os.path.basename(path)}: {len(kept)} 个气孔 / stomata")
    with open(os.path.join(a.out, "stomata_dl.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "id", "x", "y", "major_axis_px", "score"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="气孔检测 · 阶段二 深度学习流水线 / stomata detection · stage 2 deep-learning pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prelabel", help="阶段一检测结果导出为点标注 JSON（供人工校正） / export stage-1 detections as point-annotation JSON (for manual correction)")
    p.add_argument("images"); p.add_argument("-o", "--out", default="labels_points")
    p.add_argument("--expected-major", type=float, default=34.0)
    p.set_defaults(fn=cmd_prelabel)

    p = sub.add_parser("build", help="点标注 → YOLO 切片数据集 / point labels → YOLO tile dataset")
    p.add_argument("labels"); p.add_argument("--images", required=True); p.add_argument("-o", "--out", default="datasets/stomata")
    p.add_argument("--tile", type=int, default=640); p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("train", help="微调 YOLOv8-P2 / fine-tune YOLOv8-P2")
    p.add_argument("data"); p.add_argument("--model", default="yolov8s-p2.yaml"); p.add_argument("--pretrained", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=150); p.add_argument("--tile", type=int, default=640)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--freeze", type=int, default=10)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict", help="切片推理计数 / sliced inference and counting")
    p.add_argument("images"); p.add_argument("--weights", required=True); p.add_argument("-o", "--out", default="outputs_dl")
    p.add_argument("--tile", type=int, default=640); p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25); p.add_argument("--device", default=None)
    p.add_argument("--sahi", action="store_true",
                   help="改用官方 SAHI（GREEDYNMM+IOS）。在验证集上 F1 未优于内置切片器且更慢，见 docs/SAHI_COMPARISON.md / use official SAHI (GREEDYNMM+IOS) instead; on validation it was not better in F1 and slower, see docs/SAHI_COMPARISON.md")
    p.add_argument("--no-sahi", action="store_true", help="（默认行为，保留以兼容旧命令）使用内置 SlicedPredictor / (default; kept for old commands) use the built-in SlicedPredictor")
    p.add_argument("--margin-exclusion", action="store_true", help="剔除触碰图像边缘的框 / drop boxes touching the image border")
    p.add_argument("--box-pad", type=float, default=1.1, help="构建数据集时的框扩张系数，用于换算长轴 / box padding factor used when building the dataset, for converting to major axis")
    p.set_defaults(fn=cmd_predict)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
