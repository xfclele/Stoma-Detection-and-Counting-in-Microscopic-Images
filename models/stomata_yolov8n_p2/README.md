# stomata_yolov8n_p2 — 预训练气孔检测模型（初版）

| 项目 | 内容 |
|---|---|
| 结构 | YOLOv8n-P2（增加 stride-4 小目标检测头），2.9 M 参数，COCO 预训练初始化，不冻结 |
| 训练数据 | `data/samples/pre-trainning` 中 16 张 2560×1920 图像 → 279 个 640×640 切片，6,818 个框 |
| 验证数据 | 4 张整图（137.1、49.2、573.2、9.3），不参与训练 |
| 标签来源 | 由外部自动计数流程的叠加图还原的**伪标签**（`scripts/prepare_pretraining.py`），非人工真值 |
| 训练 | CPU，batch 8，计划 100 轮；因运行时限在第 61 轮停止，`best.pt` 取自第 54 轮（验证 mAP50-95 最高） |
| 文件 | `best.pt`（已去除优化器状态，6.3 MB，普通 git 文件）、`results.csv`（逐轮指标）、`args.yaml`（训练参数） |

## 指标

切片验证集（第 54 轮）：P 0.891，R 0.938，mAP50 0.963，mAP50-95 0.664。

整图切片推理（`scripts/eval_pretrain.py`，只统计有效表皮区域，匹配容差 0.35 × 长轴）：

| 置信度阈值 | Precision | Recall | F1 |
|---|---|---|---|
| 0.25 | 0.813 | 0.980 | 0.889 |
| 0.40 | 0.847 | 0.966 | 0.903 |
| 0.55 | 0.886 | 0.944 | 0.914 |

这些指标衡量的是"与原自动计数流程的一致程度"。目视检查发现，部分"误检"是原流程漏标的真实气孔，
因此真实精确率应高于表中数值，但需要人工标注一张整图才能确定。`49.2` 这类大面积被遮蔽、气孔分布零散的图像最弱（F1 ≈ 0.78）。

## 使用

```bash
# 内置切片推理 + 中心归属去重（无需 sahi）
python scripts/dl_pipeline.py predict images/ --weights models/stomata_yolov8n_p2/best.pt --no-sahi --conf 0.4 -o outputs_dl
```


## 继续训练

```python
from ultralytics import YOLO
YOLO("runs/stomata/pretrain_v8n_p2/weights/last.pt").train(resume=True)   # 需要原训练目录中的 last.pt
# 或以本权重为起点重新微调：
YOLO("models/stomata_yolov8n_p2/best.pt").train(data="datasets/stomata_pre/data.yaml", epochs=40, device="cpu")
```
