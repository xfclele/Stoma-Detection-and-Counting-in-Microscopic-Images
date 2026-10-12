# stomata_yolov8n_p2_human_v3_defocus：离焦模糊增强版（存档，非推荐） / defocus blur augmented model (archived, not recommended)

**语言 / Language：** 中文（下文） · [English](#english-version)

| 项目 | 内容 |
|---|---|
| 结构 | YOLOv8n-P2，2.9 M 参数 |
| 初始化 | `models/stomata_yolov8n_p2_human/best.pt`（v1），训练流程与 v2 相同 |
| 增强 | v2 的全部增强，另加离焦模糊（`stomata/dl/defocus.py`）：每张切片以 0.5 的概率模糊，圆盘或高斯 PSF，半径 1–3.5 px，70% 为局部模糊，可能同时降低对比度 |
| 训练 | CPU，lr0 0.0005，关闭 mosaic，3 轮（run `human_v3_defocus_from_v1`）。`results_all_runs.csv` 含全部 v3 实验段，包括对照组 |

## 指标（57 张验证图，整图切片推理）

| 阈值 | P | R | F1 | 计数 MAPE | 计数偏差 |
|---|---|---|---|---|---|
| 0.40 | 0.928 | 0.947 | 0.937 | 7.5% | +6.3 |
| 0.45 | 0.939 | 0.931 | 0.935 | 6.5% | −2.3 |

与 v2 相比，召回的变化：

- 最模糊 1/4 的气孔：+0.7 个百分点；
- <40 px 的小气孔：+3.7 个百分点；
- 40–50 px 的气孔：+1.0 个百分点。

但精确率低约 1 个百分点，整体 F1 和计数误差都不如 v2。推荐模型仍为 v2，详见 [`docs/DEFOCUS_AUGMENT.md`](../../docs/DEFOCUS_AUGMENT.md)。

---

# English version

| Item | Value |
|---|---|
| Architecture | YOLOv8n-P2, 2.9 M parameters |
| Initialisation | `models/stomata_yolov8n_p2_human/best.pt` (v1), same training recipe as v2 |
| Augmentation | all of v2's augmentation plus defocus blur (`stomata/dl/defocus.py`): each tile is blurred with probability 0.5, disk or Gaussian PSF, radius 1–3.5 px, 70% local blur, sometimes with reduced contrast |
| Training | CPU, lr0 0.0005, mosaic off, 3 epochs (run `human_v3_defocus_from_v1`). `results_all_runs.csv` contains all v3 experiment segments, including the control |

## Metrics (57 validation images, whole-image sliced inference)

| Threshold | P | R | F1 | Count MAPE | Count bias |
|---|---|---|---|---|---|
| 0.40 | 0.928 | 0.947 | 0.937 | 7.5% | +6.3 |
| 0.45 | 0.939 | 0.931 | 0.935 | 6.5% | −2.3 |

Recall changes relative to v2:

- blurriest quarter of stomata: +0.7 percentage points;
- small stomata < 40 px: +3.7 points;
- stomata of 40–50 px: +1.0 point.

But precision is about 1 point lower, and overall F1 and count error are both worse than v2. The recommended model remains v2; see [`docs/DEFOCUS_AUGMENT.md`](../../docs/DEFOCUS_AUGMENT.md).
