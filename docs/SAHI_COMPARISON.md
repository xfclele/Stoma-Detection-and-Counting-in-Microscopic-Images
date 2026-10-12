# SAHI 与内置 SlicedPredictor 的对照实验 / Controlled Comparison: SAHI vs Built-in SlicedPredictor

**语言 / Language：** 中文（下文） · [English](#english-version)

**结论：SAHI 的 F1 没有提升（−0.05 个百分点），远未达到保留门槛（> 0.5 个百分点），而且更慢。继续使用内置 `SlicedPredictor`；`dl_pipeline.py predict` 的默认后端也已改为它，需要 SAHI 时加 `--sahi`。**

## 设置

- 模型：`models/stomata_yolov8n_p2_human_v2/best.pt`。
- 验证图：57 张，排除 `118.2`、`221.2`。
- 切片：640 × 640，重叠 20%，CPU 4 核。
- 匹配：中心点一对一，容差为 0.5 × 每图中位框长边。
- 后端：
  - **SlicedPredictor**：本仓库实现，按"中心归属"去重，即每个框只由其中心所在的那块切片负责。
  - **SAHI**：0.12.8 版 `get_sliced_prediction`，不做整图预测。比较了两种合并方式：
    - GREEDYNMM + IOS 0.5，本仓库原来的 SAHI 配置；
    - NMS + IOU 0.5，SAHI 在模型阈值较低时会自动切换到这种方式。
- 推理阈值：每个后端各跑两次。一次以阈值 0.05 推理，再在 0.10–0.80 之间扫描；另一次直接以 0.40 推理。两次在 0.40 处的结果一致，误差在 0.0001 以内。
- 计时：每张图的推理时间，第一张含模型预热，不计入。

## 结果

| 后端 | 阈值 | P | R | F1 | 计数 MAE | 计数 MAPE | 计数偏差 | 每图耗时 |
|---|---|---|---|---|---|---|---|---|
| **SlicedPredictor** | 0.40 | **0.9368** | 0.9452 | **0.9410** | **18.5** | **6.60%** | **+2.7** | **1.26–1.34 s** |
| SAHI GREEDYNMM + IOS | 0.40 | 0.9341 | 0.9471 | 0.9405 | 18.9 | 6.77% | +4.2 | 1.39–1.47 s（约慢 10%） |
| SAHI NMS + IOU | 0.45（其最优） | — | — | 0.9232 | — | 8.35% | — | 1.42–1.45 s |
| SAHI NMS + IOU | 0.40 | 0.8929 | 0.9473 | 0.9193 | — | 10.28% | +18.3 | |

- 每个后端的最优阈值都是 0.40，NMS + IOU 例外，其最优阈值为 0.45。所以按各自最优阈值比较，结论与上表相同。
- **SlicedPredictor 与 SAHI GREEDYNMM 的逐图配对比较**（[对比表](sahi/compare_sliced_vs_sahi_gnmm.md)）：
  - F1：SAHI 有 17 张更好、23 张更差、17 张持平，Wilcoxon p = 0.18，差异不显著；
  - 计数误差：p = 0.16，同样不显著。
  - SAHI 的召回略高（+0.2 个百分点），但精确率更低（−0.3 个百分点）。原因是切片边界上被截断的半个气孔，有时没能和完整框合并，结果被计数两次，所以计数偏差从 +2.7 增大到 +4.2。
- **SAHI NMS + IOU** 明显更差：57 张图中 55 张的 F1 更低（p = 7e-11）。截断的半框和完整框之间的 IoU 很低，NMS 去不掉这类重复框，所以每张图约多计 18 个。

## 注意

SAHI 在模型置信度阈值较低时，会自动把合并方式改成 NMS/IOU，只在日志里打一条警告。这正是上表中较差的那种配置。`predict_with_sahi()` 现在默认传 `force_postprocess_type=True`，以保持指定的合并方式。

## 复现

```bash
python scripts/eval_human.py --weights models/stomata_yolov8n_p2_human_v2/best.pt --exclude 118.2 221.2 \
    --conf 0.40 --backend sliced -o outputs/eval_human/sahi_cmp/sliced_ic0.05
python scripts/eval_human.py --weights models/stomata_yolov8n_p2_human_v2/best.pt --exclude 118.2 221.2 \
    --conf 0.40 --backend sahi -o outputs/eval_human/sahi_cmp/sahi_gnmm_ic0.05      # 加 --sahi-postprocess NMS --sahi-metric IOU 得到 NMS 版
python scripts/compare_runs.py outputs/eval_human/sahi_cmp/sliced_ic0.05 0.40 outputs/eval_human/sahi_cmp/sahi_gnmm_ic0.05 0.40
```

---

# English version

**Conclusion: SAHI did not improve F1 (−0.05 percentage points), far from the retention threshold (> 0.5 points), and it is slower. The built-in `SlicedPredictor` stays; it is now also the default backend of `dl_pipeline.py predict`, with `--sahi` to opt into SAHI.**

## Setup

- Model: `models/stomata_yolov8n_p2_human_v2/best.pt`.
- Validation images: 57, excluding `118.2` and `221.2`.
- Tiles: 640 × 640, 20% overlap, 4 CPU cores.
- Matching: one-to-one centre matching, tolerance 0.5 × the image's median box long side.
- Backends:
  - **SlicedPredictor**: this repository's implementation with "centre ownership" de-duplication, i.e. each box belongs only to the tile containing its centre.
  - **SAHI**: version 0.12.8 `get_sliced_prediction`, without a whole-image pass. Two merge modes were compared:
    - GREEDYNMM + IOS 0.5, this repository's original SAHI configuration;
    - NMS + IOU 0.5, which SAHI switches to automatically when the model threshold is low.
- Inference thresholds: each backend was run twice — once at threshold 0.05 followed by a sweep over 0.10–0.80, and once directly at 0.40. The two agree at 0.40 to within 0.0001.
- Timing: inference time per image; the first image includes model warm-up and is excluded.

## Results

| Backend | Threshold | P | R | F1 | Count MAE | Count MAPE | Count bias | Time per image |
|---|---|---|---|---|---|---|---|---|
| **SlicedPredictor** | 0.40 | **0.9368** | 0.9452 | **0.9410** | **18.5** | **6.60%** | **+2.7** | **1.26–1.34 s** |
| SAHI GREEDYNMM + IOS | 0.40 | 0.9341 | 0.9471 | 0.9405 | 18.9 | 6.77% | +4.2 | 1.39–1.47 s (about 10% slower) |
| SAHI NMS + IOU | 0.45 (its best) | — | — | 0.9232 | — | 8.35% | — | 1.42–1.45 s |
| SAHI NMS + IOU | 0.40 | 0.8929 | 0.9473 | 0.9193 | — | 10.28% | +18.3 | |

- The best threshold of every backend is 0.40, except NMS + IOU at 0.45, so comparing at each backend's best threshold gives the same conclusion.
- **Per-image paired comparison of SlicedPredictor and SAHI GREEDYNMM** ([comparison table](sahi/compare_sliced_vs_sahi_gnmm.md)):
  - F1: SAHI better on 17 images, worse on 23, tied on 17; Wilcoxon p = 0.18, not significant;
  - count error: p = 0.16, also not significant.
  - SAHI's recall is slightly higher (+0.2 points) but its precision is lower (−0.3 points): half stomata cut by tile borders sometimes fail to merge with the complete box and are counted twice, so the count bias rises from +2.7 to +4.2.
- **SAHI NMS + IOU** is clearly worse: lower F1 on 55 of 57 images (p = 7e-11). The IoU between a truncated half box and the complete box is low, so NMS cannot remove these duplicates, adding about 18 counts per image.

## Note

When the model confidence threshold is low, SAHI silently switches the merge mode to NMS/IOU, logging only a warning — exactly the poor configuration in the table. `predict_with_sahi()` now passes `force_postprocess_type=True` by default to keep the specified merge mode.

## Reproduction

```bash
python scripts/eval_human.py --weights models/stomata_yolov8n_p2_human_v2/best.pt --exclude 118.2 221.2 \
    --conf 0.40 --backend sliced -o outputs/eval_human/sahi_cmp/sliced_ic0.05
python scripts/eval_human.py --weights models/stomata_yolov8n_p2_human_v2/best.pt --exclude 118.2 221.2 \
    --conf 0.40 --backend sahi -o outputs/eval_human/sahi_cmp/sahi_gnmm_ic0.05      # add --sahi-postprocess NMS --sahi-metric IOU for the NMS variant
python scripts/compare_runs.py outputs/eval_human/sahi_cmp/sliced_ic0.05 0.40 outputs/eval_human/sahi_cmp/sahi_gnmm_ic0.05 0.40
```
