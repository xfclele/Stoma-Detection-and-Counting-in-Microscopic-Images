#!/usr/bin/env python
"""在 v2 基础上加离焦模糊增强继续微调（其余超参与 v2_b 相同：lr0 0.0005、mosaic 关、cos_lr）。
Continue fine-tuning with defocus blur augmentation (other hyper-parameters as v2_b: lr0 0.0005, mosaic off, cos_lr).

    python scripts/train_defocus.py --name human_v3_defocus_a                       # 从 v2 开始 / start from v2
    python scripts/train_defocus.py --weights runs/stomata/human_v3_defocus_a/weights/last.pt \
        --name human_v3_defocus_b                                                   # 下一段 / next segment

CPU 每轮约 17 分钟，后台任务约 1 小时会被终止，所以每段用 time=0.85 小时。
On CPU one epoch takes ~17 min and background jobs are killed after ~1 h, so each segment uses time=0.85 h.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata.dl.defocus import install_defocus_augment  # noqa: E402

KEEP = ["imgsz", "batch", "workers", "optimizer", "cos_lr", "lrf", "momentum", "weight_decay", "warmup_epochs",
        "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
        "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "close_mosaic", "single_cls", "patience"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/stomata_yolov8n_p2_human_v2/best.pt")
    ap.add_argument("--base-args", default="models/stomata_yolov8n_p2_human_v2/args.yaml")
    ap.add_argument("--data", default="datasets/stomata_human_v2/data.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--time", type=float, default=0.85)
    ap.add_argument("--lr0", type=float, default=0.0005)
    ap.add_argument("--p", type=float, default=0.5, help="每张训练切片施加离焦的概率 / probability of defocusing each training tile")
    ap.add_argument("--radius", type=float, nargs=2, default=(1.0, 3.5), help="圆盘 PSF 半径范围（原分辨率像素） / disk PSF radius range (original-resolution px)")
    ap.add_argument("--local-p", type=float, default=0.7, help="局部（而非整片）失焦的比例 / fraction of local (rather than whole-tile) defocus")
    ap.add_argument("--contrast-p", type=float, default=0.5, help="模糊区域同时降低对比度的概率 / probability of also lowering contrast in blurred areas")
    a = ap.parse_args()

    base = yaml.safe_load(open(a.base_args))
    args = {k: base[k] for k in KEEP if k in base}
    args.update(data=a.data, epochs=a.epochs, time=a.time, lr0=a.lr0, project="runs/stomata", name=a.name,
                device="cpu", exist_ok=True, plots=False, amp=False)

    install_defocus_augment(p=a.p, radius=tuple(a.radius), local_p=a.local_p, contrast_p=a.contrast_p)
    from ultralytics import YOLO

    YOLO(a.weights).train(**args)


if __name__ == "__main__":
    main()
