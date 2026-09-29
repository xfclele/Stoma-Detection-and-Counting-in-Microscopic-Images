"""stomata —— 植物叶表皮显微图像气孔检测、定位与计数。

阶段一（免训练）: ClassicalStomataDetector / StomataCounter / visualize_pipeline
阶段二（深度学习）: stomata.dl （YOLO + P2 小目标头 + SAHI 切片推理，需 ultralytics / sahi）
"""
from .detector import ClassicalStomataDetector, Detection, DetectionResult, DetectorConfig
from .counter import StomataCounter, ImageReport
from .evaluation import match_points, evaluate_against_annotation, precision_recall_f1, load_annotations
from .templates import StomaTemplate
from .calibration import MicrometerCalibration, calibrate_stage_micrometer
from .visualize import visualize_pipeline, draw_detections

__all__ = [
    "ClassicalStomataDetector", "Detection", "DetectionResult", "DetectorConfig",
    "StomataCounter", "ImageReport",
    "match_points", "evaluate_against_annotation", "precision_recall_f1", "load_annotations",
    "StomaTemplate", "visualize_pipeline", "draw_detections",
    "MicrometerCalibration", "calibrate_stage_micrometer",
]
__version__ = "0.1.0"
