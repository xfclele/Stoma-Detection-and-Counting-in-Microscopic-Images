#!/usr/bin/env python
"""命令行：批量气孔计数（阶段一，免训练）。

示例
----
    # 单张图，输出 CSV + 标注图 + 流水线对比图
    python scripts/count_stomata.py data/samples/leaf_epidermis_01.jpg -o outputs --visualize

    # 整个目录，已知像素尺寸 0.45 µm/px，边缘剔除改为"中心距边界 < 20 px"
    python scripts/count_stomata.py "images/*.tif" -o outputs --um-per-px 0.45 --margin-mode center --margin-px 20

    # 用测微尺图像标定 µm/px，并以物理尺寸给出气孔先验（20x 系列气孔长轴约 28 µm）
    python scripts/count_stomata.py "data/samples/test0*.tif" -o outputs \
        --micrometer "data/samples/stage micrometer 20x objectivetif.tif" --stoma-size-um 28

    # 与点标注对比，输出 Precision / Recall / F1
    python scripts/count_stomata.py data/samples/leaf_epidermis_01.jpg -o outputs \
        --eval data/annotations/leaf_epidermis_01_partial.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stomata import DetectorConfig, StomataCounter, draw_detections, evaluate_against_annotation, load_annotations  # noqa: E402
from stomata.calibration import calibrate_stage_micrometer  # noqa: E402
from stomata.templates import StomaTemplate  # noqa: E402
from stomata.visualize import visualize_pipeline  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="叶表皮显微图像气孔检测与计数（免训练基线）")
    ap.add_argument("input", help="图像文件 / 目录 / 通配符")
    ap.add_argument("-o", "--out", default="outputs", help="输出目录")
    ap.add_argument("--expected-major", type=float, default=34.0, help="气孔长轴的大致像素长度（20–60）")
    ap.add_argument("--um-per-px", type=float, default=None, help="像素物理尺寸 (µm/px)，给出后导出 µm 与密度")
    ap.add_argument("--micrometer", default=None, help="同一物镜拍摄的载物台测微尺图像，用于自动标定 µm/px")
    ap.add_argument("--division-um", type=float, default=10.0, help="测微尺每小格的长度 (µm)，常见 10")
    ap.add_argument("--stoma-size-um", type=float, default=None, help="气孔长轴的大致物理长度 (µm)；需 µm/px")
    ap.add_argument("--candidate-mode", choices=["ncc", "doh"], default="ncc")
    ap.add_argument("--no-margin-exclusion", action="store_true", help="保留触碰图像边缘的不完整气孔")
    ap.add_argument("--margin-mode", choices=["bbox", "center"], default="bbox")
    ap.add_argument("--margin-px", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=None, help="覆盖默认 NCC 得分阈值")
    ap.add_argument("--template", default=None, help="加载已保存的模板 .npz（跳过自举）")
    ap.add_argument("--save-template", default=None, help="把第一张图自举得到的模板保存为 .npz")
    ap.add_argument("--share-template", action="store_true", help="整批图像共用第一张图的模板")
    ap.add_argument("--visualize", action="store_true", help="保存流水线四联图")
    ap.add_argument("--eval", default=None, help="点标注 JSON，计算 Precision/Recall/F1")
    ap.add_argument("--tol", type=float, default=None, help="评估匹配容差（像素），默认 0.35 × 期望长轴")
    args = ap.parse_args(argv)

    um_per_px = args.um_per_px
    if args.micrometer:
        cal = calibrate_stage_micrometer(StomataCounter.read_image(args.micrometer), args.division_um)
        um_per_px = cal.um_per_px
        print(f"测微尺标定: {cal.um_per_px:.4f} µm/px（周期 {cal.period_px:.2f} px，{cal.n_ticks} 条刻线，"
              f"残差 {cal.residual_px:.2f} px）")
    cfg = DetectorConfig(expected_major_px=args.expected_major, um_per_px=um_per_px,
                         expected_major_um=args.stoma_size_um, candidate_mode=args.candidate_mode,
                         margin_exclusion=not args.no_margin_exclusion, margin_mode=args.margin_mode,
                         margin_px=args.margin_px, template_path=args.template)
    if cfg.expected_major_um and cfg.um_per_px:
        cfg.expected_major_px = cfg.expected_major_um / cfg.um_per_px
        print(f"期望气孔长轴: {cfg.expected_major_px:.1f} px")
    tol = args.tol if args.tol is not None else 0.35 * cfg.expected_major_px
    if args.min_score is not None:
        cfg.min_score = args.min_score
    template = StomaTemplate.load(args.template) if args.template else None
    counter = StomataCounter(cfg, template=template, share_template=args.share_template)
    os.makedirs(args.out, exist_ok=True)

    reports = []
    for path in counter.list_images(args.input):
        t0 = time.time()
        img = counter.read_image(path)
        rep = counter.process(img, os.path.basename(path))
        reports.append(rep)
        stem = os.path.splitext(rep.image)[0]
        cv2.imwrite(os.path.join(args.out, f"{stem}_detections.jpg"), draw_detections(img, rep.result))
        counter.export_points_json(rep, os.path.join(args.out, f"{stem}_points.json"))
        if args.visualize:
            visualize_pipeline(img, rep.result, save_path=os.path.join(args.out, f"{stem}_pipeline.png"))
        if args.save_template and len(reports) == 1:
            rep.result.template.save(args.save_template)
        msg = f"{rep.image}: {rep.count} 个气孔  ({time.time() - t0:.1f}s)"
        if args.eval:
            ann = load_annotations(args.eval)
            if ann.get("image") in (None, rep.image):
                m = evaluate_against_annotation([(d.x, d.y) for d in rep.result.detections], ann, tol)
                msg += "\n    评估: " + m.summary()
                with open(os.path.join(args.out, f"{stem}_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(m.to_dict(), f, ensure_ascii=False, indent=1)
        print(msg)

    if reports:
        detail, summary = counter.export_csv(reports, os.path.join(args.out, "stomata_detections.csv"))
        print(f"明细: {detail}\n汇总: {summary}")
    else:
        print("未找到图像。")


if __name__ == "__main__":
    main()
