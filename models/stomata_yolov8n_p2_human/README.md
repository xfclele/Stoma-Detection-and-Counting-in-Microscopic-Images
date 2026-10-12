# stomata_yolov8n_p2_human — 人工标注微调模型（v1，已被 v2 取代） / model fine-tuned on human labels (v1, superseded by v2)

**语言 / Language：** 中文（下文） · [English](#english-version)

| 项目 | 内容 |
|---|---|
| 结构 | YOLOv8n-P2（stride-4 小目标检测头），2.9 M 参数 |
| 初始化 | `models/stomata_yolov8n_p2/best.pt`（伪标签预训练） |
| 训练数据 | `Stomata_Enhanced` 人工标注：126 张 2560×1920 图像 → 1,512 个 640 切片（剔除标注不完整的 `108.1_1`、`349.3`） |
| 训练 | CPU，三段共 18 轮（ft1→ft2→ft3），本权重为 ft3 的最佳轮；逐轮指标见 `results_all_runs.csv` |
| 推荐阈值 | **conf = 0.45**（验证集 F1 最优） |

## 指标（57 张人工标注验证图，整图切片推理）

| Precision | Recall | F1 | 计数 MAPE | 计数偏差 |
|---|---|---|---|---|
| 0.935 | 0.944 | 0.939 | 6.8% | +3.1 / 图 |

阈值与模型都在该验证集上选择，数字略偏乐观。详细分析见 [`docs/TRAINING_REPORT.md`](../../docs/TRAINING_REPORT.md)。

## 使用

```bash
python scripts/dl_pipeline.py predict images/ --weights models/stomata_yolov8n_p2_human/best.pt --no-sahi --conf 0.45 -o outputs_dl
```

---

# English version

| Item | Value |
|---|---|
| Architecture | YOLOv8n-P2 (stride-4 small-object head), 2.9 M parameters |
| Initialisation | `models/stomata_yolov8n_p2/best.pt` (pseudo-label pre-training) |
| Training data | human-labelled `Stomata_Enhanced`: 126 images of 2560×1920 → 1,512 tiles of 640 (incompletely labelled `108.1_1` and `349.3` removed) |
| Training | CPU, three segments with 18 epochs in total (ft1 → ft2 → ft3); these weights are the best epoch of ft3; per-epoch metrics in `results_all_runs.csv` |
| Recommended threshold | **conf = 0.45** (best F1 on validation) |

## Metrics (57 human-labelled validation images, whole-image sliced inference)

| Precision | Recall | F1 | Count MAPE | Count bias |
|---|---|---|---|---|
| 0.935 | 0.944 | 0.939 | 6.8% | +3.1 per image |

Both threshold and model were selected on this validation set, so the numbers are slightly optimistic. Detailed analysis: [`docs/TRAINING_REPORT.md`](../../docs/TRAINING_REPORT.md).

## Usage

```bash
python scripts/dl_pipeline.py predict images/ --weights models/stomata_yolov8n_p2_human/best.pt --no-sahi --conf 0.45 -o outputs_dl
```
