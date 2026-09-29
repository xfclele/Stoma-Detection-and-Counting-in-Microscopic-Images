"""StomataCounter：批量读取、计数、坐标提取、统计与 CSV 导出。"""
from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import cv2
import numpy as np

from .detector import ClassicalStomataDetector, DetectionResult, DetectorConfig
from .templates import StomaTemplate

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

DETAIL_FIELDS = ["image", "id", "x", "y", "major_axis_px", "minor_axis_px", "angle_deg", "score",
                 "major_source", "major_axis_um", "minor_axis_um"]
SUMMARY_FIELDS = ["image", "width", "height", "count", "density_per_mm2", "major_mean_px", "major_std_px",
                  "major_median_px", "n_candidates", "n_rejected_margin"]


@dataclass
class ImageReport:
    image: str
    width: int
    height: int
    result: DetectionResult
    um_per_px: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return self.result.count

    def rows(self) -> list[dict]:
        out = []
        for i, d in enumerate(sorted(self.result.detections, key=lambda d: (d.y, d.x)), start=1):
            row = {
                "image": self.image, "id": i, "x": round(d.x, 1), "y": round(d.y, 1),
                "major_axis_px": round(d.major_px, 2), "minor_axis_px": round(d.minor_px, 2),
                "angle_deg": round(d.angle_deg, 1), "score": round(d.score, 4), "major_source": d.major_source,
                "major_axis_um": "", "minor_axis_um": "",
            }
            if self.um_per_px:
                row["major_axis_um"] = round(d.major_px * self.um_per_px, 2)
                row["minor_axis_um"] = round(d.minor_px * self.um_per_px, 2)
            out.append(row)
        return out

    def summary(self) -> dict:
        majors = np.array([d.major_px for d in self.result.detections], float)
        area_mm2 = (self.width * self.height * self.um_per_px ** 2 / 1e6) if self.um_per_px else None
        return {
            "image": self.image, "width": self.width, "height": self.height, "count": self.count,
            "density_per_mm2": round(self.count / area_mm2, 2) if area_mm2 else "",
            "major_mean_px": round(float(majors.mean()), 2) if len(majors) else "",
            "major_std_px": round(float(majors.std()), 2) if len(majors) else "",
            "major_median_px": round(float(np.median(majors)), 2) if len(majors) else "",
            "n_candidates": self.result.extras.get("n_candidates", ""),
            "n_rejected_margin": sum(1 for d in self.result.rejected if d.reason == "margin"),
        }


class StomataCounter:
    """批量气孔计数器。

    示例::

        counter = StomataCounter(DetectorConfig(expected_major_px=34, um_per_px=0.5))
        reports = counter.run("data/samples")            # 目录 / 通配符 / 文件列表
        counter.export_csv(reports, "outputs/stomata.csv")  # 逐气孔明细 + *_summary.csv 汇总
    """

    def __init__(self, config: Optional[DetectorConfig] = None, template: Optional[StomaTemplate] = None,
                 share_template: bool = False):
        """
        share_template=True：用第一张图自举的模板处理整批图像（同一批次拍摄条件一致时更快、更一致）；
        False（默认）：每张图单独自举，适应不同放大倍率/染色。
        """
        self.cfg = config or DetectorConfig()
        self.template = template
        self.share_template = share_template

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def list_images(source: Union[str, Iterable[str]]) -> list[str]:
        if isinstance(source, str):
            if os.path.isdir(source):
                files = [os.path.join(source, f) for f in sorted(os.listdir(source))]
            elif any(ch in source for ch in "*?["):
                files = sorted(glob.glob(source))
            else:
                files = [source]
        else:
            files = list(source)
        return [f for f in files if f.lower().endswith(IMAGE_EXTS)]

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        # imdecode 兼容中文路径；IMREAD_UNCHANGED 保留 16-bit 显微图像的动态范围
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        return img

    # ------------------------------------------------------------------ run
    def process(self, image: np.ndarray, name: str = "image") -> ImageReport:
        det = ClassicalStomataDetector(self.cfg, template=self.template)
        res = det.detect(image)
        if self.share_template and self.template is None:
            self.template = det.template
        h, w = image.shape[:2]
        return ImageReport(name, w, h, res, self.cfg.um_per_px)

    def run(self, source: Union[str, Iterable[str]], verbose: bool = True) -> list[ImageReport]:
        reports = []
        for path in self.list_images(source):
            rep = self.process(self.read_image(path), os.path.basename(path))
            reports.append(rep)
            if verbose:
                print(f"[StomataCounter] {rep.image}: {rep.count} 个气孔")
        return reports

    # --------------------------------------------------------------- export
    @staticmethod
    def export_csv(reports: list[ImageReport], csv_path: str) -> tuple[str, str]:
        """写出两份 CSV：逐气孔明细 (csv_path) 与逐图像汇总 (*_summary.csv)。UTF-8 BOM 便于 Excel 打开。"""
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
            w.writeheader()
            for rep in reports:
                w.writerows(rep.rows())
        root, ext = os.path.splitext(csv_path)
        summary_path = f"{root}_summary{ext or '.csv'}"
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            for rep in reports:
                w.writerow(rep.summary())
        return csv_path, summary_path

    @staticmethod
    def export_points_json(report: ImageReport, path: str) -> None:
        """导出点标注 JSON（与 evaluation.load_annotations 格式兼容），可作为人工校正/深度学习的预标注。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"image": report.image,
                       "points": [[round(d.x, 1), round(d.y, 1)] for d in report.result.detections],
                       "major_px": [round(d.major_px, 1) for d in report.result.detections]},
                      f, ensure_ascii=False, indent=1)
