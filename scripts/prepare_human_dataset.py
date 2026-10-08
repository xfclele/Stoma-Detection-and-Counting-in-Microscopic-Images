#!/usr/bin/env python
"""Stomata_Enhanced（人工标注）→ 切片 YOLO 数据集 + 数据清单。

数据现状（见 docs/TRAINING_REPORT.md）：
    Stomata_Enhanced/train/labels/*.txt   234 个标签；对应图像散落在
        Stomata_Enhanced/train/*.jpg（6 张）与 pre-trainning/images/*.jpg（122 张）
        —— 其余 106 个标签暂无图像，跳过并记入清单
    Stomata_Enhanced/val/images + labels  59 对，完整
标签格式：每行 `0 x1 y1 x2 y1 x2 y2 x1 y2`（归一化 4 角点，轴对齐矩形），也兼容 `0 cx cy w h`。

    python scripts/prepare_human_dataset.py -o datasets/stomata_human
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata.dl import build_yolo_dataset, load_yolo_polygon_labels  # noqa: E402


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def find_pairs(label_dir, image_dirs):
    imgs = {}
    for d in image_dirs:                       # 靠前的目录优先（Stomata_Enhanced 自带图像优先）
        for p in sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))):
            imgs.setdefault(stem(p), p)
    pairs, missing = [], []
    for lp in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        (pairs.append((imgs[stem(lp)], lp)) if stem(lp) in imgs else missing.append(stem(lp)))
    return pairs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/samples/pre-trainning/Stomata_Enhanced")
    ap.add_argument("--extra-images", default="data/samples/pre-trainning/images")
    ap.add_argument("-o", "--out", default="datasets/stomata_human")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--train-overlap", type=float, default=0.0, help="2560×1920 恰好切成 4×3 个 640 块")
    a = ap.parse_args()

    tr, tr_missing = find_pairs(os.path.join(a.root, "train/labels"), [os.path.join(a.root, "train"), a.extra_images])
    va, va_missing = find_pairs(os.path.join(a.root, "val/labels"), [os.path.join(a.root, "val/images")])

    def load(pairs):
        out = []
        for ip, lp in pairs:
            h, w = cv2.imread(ip, cv2.IMREAD_REDUCED_GRAYSCALE_8).shape[:2]
            out.append(load_yolo_polygon_labels(lp, ip, w * 8, h * 8))
        return out

    tr_anns, va_anns = load(tr), load(va)
    val_names = [os.path.basename(x.image_path) for x in va_anns]
    build_yolo_dataset(tr_anns + va_anns, a.out, tile=a.tile, overlap=a.train_overlap, val_images=val_names,
                       min_visible=0.5, keep_empty=1.0)
    manifest = {
        "train": [{"image": i, "label": l} for i, l in tr],
        "val": [{"image": i, "label": l} for i, l in va],
        "train_labels_without_image": tr_missing,
        "val_labels_without_image": va_missing,
        "counts": {"train_images": len(tr), "train_boxes": int(sum(len(x.points) for x in tr_anns)),
                   "val_images": len(va), "val_boxes": int(sum(len(x.points) for x in va_anns))},
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps(manifest["counts"]), f"缺图像的训练标签: {len(tr_missing)}")


if __name__ == "__main__":
    main()
