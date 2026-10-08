"""阶段二：生产级深度学习方案（YOLOv8-P2 + SAHI 切片推理）。

本子包的 dataset / sliced_inference 模块只依赖 numpy + OpenCV；
训练与官方 SAHI 推理需额外安装：pip install ultralytics sahi
"""
from .dataset import (PointAnnotation, boxes_to_yolo_lines, build_yolo_dataset, clip_boxes_to_tile, iter_tiles,
                      load_point_annotations, load_yolo_polygon_labels, points_to_boxes)
from .sliced_inference import BoxDetection, SlicedPredictor, greedy_nms_ios, ios_matrix

__all__ = [
    "PointAnnotation", "boxes_to_yolo_lines", "build_yolo_dataset", "clip_boxes_to_tile", "iter_tiles",
    "load_point_annotations", "load_yolo_polygon_labels", "points_to_boxes",
    "BoxDetection", "SlicedPredictor", "greedy_nms_ios", "ios_matrix",
]
