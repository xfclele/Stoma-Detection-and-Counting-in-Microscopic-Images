# stomata_yolov8n_p2_human_v2 — 完整人工标注数据微调（当前推荐） / fine-tuned on the complete human-labelled data (currently recommended)

**语言 / Language：** 中文（下文） · [English](#english-version)

| 项目 | 内容 |
|---|---|
| 结构 | YOLOv8n-P2（stride-4 小目标检测头），2.9 M 参数 |
| 初始化 | `models/stomata_yolov8n_p2_human/best.pt`（v1） |
| 训练数据 | 整理补全后的 `Stomata_Enhanced`：232 张 2560×1920 图像 → 2,784 个 640 切片、71,525 个框（剔除标注不完整的 `108.1_1`、`349.3`） |
| 训练 | CPU，lr0 0.0005，关闭 mosaic，4 轮（run `human_v2_b`）；全部训练段的逐轮指标见 `results_all_runs.csv` |
| 推荐阈值 | **conf = 0.40** |

## 指标（57 张人工标注验证图，整图切片推理）

| Precision | Recall | F1 | 计数 MAPE | 计数偏差 |
|---|---|---|---|---|
| 0.937 | 0.945 | 0.941 | 6.6% | +2.7 / 图 |

与 v1 相比逐图 F1 提升显著（39/57 张更好，p = 0.011），计数误差略降但不显著。阈值在验证集上选择；v1 在 106 张独立新图上的结果（F1 0.939）与其验证集结果一致，可作为这一偏差很小的参考。

验证集中重复的 `23.2_1` 已删除。去重后的 56 张上为 P 0.937 / R 0.948 / F1 0.942，计数误差 6.4%，偏差 +3.5 / 图，推荐阈值仍为 0.40。
详见 [`docs/TRAINING_REPORT.md`](../../docs/TRAINING_REPORT.md) 第 6 节。

## 使用

```bash
python scripts/dl_pipeline.py predict images/ --weights models/stomata_yolov8n_p2_human_v2/best.pt --no-sahi --conf 0.4 -o outputs_dl
```

---

# English version

| Item | Value |
|---|---|
| Architecture | YOLOv8n-P2 (stride-4 small-object head), 2.9 M parameters |
| Initialisation | `models/stomata_yolov8n_p2_human/best.pt` (v1) |
| Training data | the reorganised, complete `Stomata_Enhanced`: 232 images of 2560×1920 → 2,784 tiles of 640, 71,525 boxes (incompletely labelled `108.1_1` and `349.3` removed) |
| Training | CPU, lr0 0.0005, mosaic off, 4 epochs (run `human_v2_b`); per-epoch metrics of all training segments in `results_all_runs.csv` |
| Recommended threshold | **conf = 0.40** |

## Metrics (57 human-labelled validation images, whole-image sliced inference)

| Precision | Recall | F1 | Count MAPE | Count bias |
|---|---|---|---|---|
| 0.937 | 0.945 | 0.941 | 6.6% | +2.7 per image |

Per-image F1 is significantly better than v1 (39/57 images better, p = 0.011); count error is slightly lower but not significantly. The threshold was chosen on the validation set; v1's result on 106 independent new images (F1 0.939) matched its validation result, suggesting this bias is small.

The duplicate `23.2_1` has since been removed from the validation set. On the 56 deduplicated images: P 0.937 / R 0.948 / F1 0.942, count error 6.4%, bias +3.5 per image; the recommended threshold is still 0.40.
See section 6 of [`docs/TRAINING_REPORT.md`](../../docs/TRAINING_REPORT.md).

## Usage

```bash
python scripts/dl_pipeline.py predict images/ --weights models/stomata_yolov8n_p2_human_v2/best.pt --no-sahi --conf 0.4 -o outputs_dl
```
