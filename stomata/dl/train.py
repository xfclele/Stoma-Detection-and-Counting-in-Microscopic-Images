"""阶段二 · YOLOv8-P2 小样本微调。

为什么选 "YOLOv8 + P2 检测头" 而非密度估计 (CSRNet/FIDTM)？
------------------------------------------------------------
* 任务要求 *逐个* 输出坐标与长轴 ⇒ 检测模型天然给出 (x, y, w, h)；密度图方法只擅长"总数"，
  坐标需再做局部极大值后处理，长轴无法直接得到。
* 气孔在本数据中 20–60 px、彼此分离度尚可、密度中等（非人群计数那种极端遮挡），
  检测模型的精度优势可以发挥；只有在"大量严重重叠、只关心总数"时才建议 FIDTM。
* 标准 YOLOv8 最高分辨率输出为 P3 (stride 8)，一个 20 px 的气孔只落在 ~2.5×2.5 的网格上；
  P2 头 (stride 4) 把特征图分辨率提升一倍，对 <32 px 小目标召回显著更好，代价是显存 ↑。
* 朝向任意 ⇒ 如需直接回归长轴与朝向，可换用 YOLOv8-OBB（旋转框），标注需为旋转框。

小样本 (few-shot) 策略
----------------------
* 冻结 backbone 前 10 层（freeze=10），只微调 neck + head，20–50 张切片即可收敛；
* 强几何增强：degrees=180（任意朝向）、flipud/fliplr=0.5、mosaic、scale=0.25；
  关闭色调/饱和度增强（灰度图）、仅保留亮度扰动 hsv_v 模拟光照不均；
* 负样本切片（叶脉、空白）按 ~30% 保留，直接压制阶段一中最主要的误检来源。
"""
from __future__ import annotations

from typing import Optional


def train_yolo_p2(data_yaml: str, model_cfg: str = "yolov8s-p2.yaml", pretrained: Optional[str] = "yolov8s.pt",
                  epochs: int = 150, imgsz: int = 640, batch: int = 16, freeze: int = 10, device=None,
                  project: str = "runs/stomata", name: str = "yolov8s_p2", **overrides):
    """微调 YOLOv8-P2；返回 Ultralytics 训练结果。需要 `pip install ultralytics`。"""
    from ultralytics import YOLO

    model = YOLO(model_cfg)                 # P2 结构（ultralytics 内置 yolov8{n,s,m,l,x}-p2.yaml）
    if pretrained:
        model = model.load(pretrained)      # 迁移 COCO 权重中形状匹配的层（P2 新增层随机初始化）
    args = dict(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, freeze=freeze, device=device,
        project=project, name=name,
        # ---- 增强：几何为主，颜色为辅 ----
        degrees=180.0, flipud=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=20, scale=0.25, translate=0.1,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        # ---- 小目标友好 ----
        rect=False, cache=True, patience=50, cos_lr=True,
        box=7.5, cls=0.5, dfl=1.5,
        single_cls=True,
    )
    args.update(overrides)
    return model.train(**args)
