# 多尺度 TTA（1.0× + 1.25×）实验 / Multi-Scale TTA (1.0× + 1.25×) Experiment

**语言 / Language：** 中文（下文） · [English](#english-version)

**结论：不采用。** 多尺度 TTA 确实把 <40 px 小气孔的召回提高了 2.5–3.3 个百分点，但所有融合方式的整体 F1 都不高于单尺度（最好的 0.9409，基线 0.9410），每图耗时增加约 1.5 倍。默认推理仍为单尺度 `SlicedPredictor`。`stomata/dl/tta.py` 保留为可选工具。

## 设置

- 模型：`models/stomata_yolov8n_p2_human_v2/best.pt`。
- 验证图：57 张，排除 `118.2`、`221.2`。
- 切片：640 × 640，重叠 20%，CPU 4 核。
- 匹配：中心点一对一，容差为 0.5 × 每图中位框长边。
- 推理：整图先放大到 1.25× 或 1.5×（双三次插值），再用同样的 640 切片推理，最后把坐标除以缩放倍数还原。1.25× 在训练时 ±25% 尺度增强的范围内。
- 融合方式（`stomata/dl/tta.py: fuse_scales`）：
  - `nms`：所有尺度的框合在一起，做 IoS-NMS。
  - `wbf`：跨尺度聚类后，框按分数加权平均。分数取各尺度的平均值，某个尺度没有检出时按 0 计。
  - `small{px}`：以 1.0× 的结果为主，只用其他尺度中原尺度长边小于 px 的框。与 1.0× 的框重叠时，把那个框的分数提升为两者的较大值；不重叠时作为新框补充。
- 每种方案都先以阈值 0.05 推理，再扫描阈值 0.10–0.80；表中报告阈值 0.40 和各自的 F1 最优阈值。

## 结果

| 方案 | 最优阈值 | P | R | F1 | 计数 MAPE | 计数偏差 | <40 px 召回 | 每图耗时 |
|---|---|---|---|---|---|---|---|---|
| **单尺度 1.0×（现行）** | 0.40 | 0.9368 | 0.9452 | **0.9410** | **6.60%** | +2.7 | 0.704 | **1.3 s** |
| 单尺度 1.25× | 0.30 | 0.9214 | 0.9456 | 0.9334 | 7.64% | +7.9 | 0.738 | 2.1 s |
| 1.0 + 1.25 `small40` | 0.40 | 0.9366 | 0.9452 | 0.9409 | 6.63% | +2.8 | 0.704 | 3.4 s |
| 1.0 + 1.25 `small60` | 0.40 | 0.9321 | 0.9487 | 0.9403 | 7.12% | — | 0.729 | 3.4 s |
| 1.0 + 1.25 `nms` | 0.40 | 0.9311 | 0.9490 | 0.9400 | 7.23% | +5.8 | 0.729 | 3.4 s |
| 1.0 + 1.25 `wbf` | 0.35 | 0.9307 | 0.9474 | 0.9390 | 7.04% | +5.4 | 0.738 | 3.4 s |
| 1.0 + 1.5 `small40` | 0.40 | 0.9354 | 0.9453 | 0.9403 | 6.75% | — | — | 4.6 s |
| 1.0 + 1.5 `nms` | 0.40 | 0.9310 | 0.9482 | 0.9395 | 7.27% | +5.5 | — | 4.6 s |
| 1.0 + 1.5 `wbf` | 0.25 | 0.9157 | 0.9555 | 0.9352 | 8.61% | +13.0 | — | 4.6 s |

逐图配对比较（[`tta/`](tta/)）：

- `nms`：F1 有 17 张更好、32 张更差（p = 0.08）。
- `wbf`：15 张更好、36 张更差（p = 0.005）。
- `small60`：17 张更好、28 张更差（p = 0.25）。
- 单独用 1.25×：50/57 张更差。

## 为什么没有提升

1. **小气孔太少，无论怎么改进都难以撬动 F1。** <40 px 的气孔只有 240 个，占全部 17,092 个的 1.4%，当前漏检 71 个。即使把这 71 个全部找回、而且不增加任何误检，F1 也只能从 0.9410 升到 0.9432（+0.22 个百分点）。所以只针对小气孔，理论上就达不到"F1 提升 > 0.5 个百分点"的门槛。
2. **放大后，对中等大小气孔的误检会增加。** 单独用 1.25× 时，精确率从 0.937 降到 0.921。原来 45–60 px 的气孔放大后变成 56–75 px，接近训练分布的上沿；同时背景纹理被放大，误检随之增加。融合后找回的小气孔，不足以抵消新增的误检。
3. **小气孔通常已经被检出，只是分数低。** 基准推理会保留低至 0.05 的框，1.25× 的小框几乎总能和某个低分的 1.0× 框重合。所以"只补充新框"的方式没有效果，必须改成"提升已有框的分数"（即现在的 `small` 实现）。另外，模型给小气孔预测的框往往比人工标注的框大，所以按预测框尺寸筛选小目标时，门槛要放宽到 50–60 px 才有作用；而放宽后，中等大小气孔的误检也会混进来。

## 用法（可选）

```bash
python scripts/tta_eval.py infer --scale 1.25 -o outputs/eval_human/tta/s1.25
python scripts/tta_eval.py fuse outputs/eval_human/sahi_cmp/sliced_ic0.05 outputs/eval_human/tta/s1.25 \
    --mode nms -o outputs/eval_human/tta/f1.25_nms
python scripts/eval_human.py --from-npz -o outputs/eval_human/tta/f1.25_nms --exclude 118.2 221.2 --conf 0.40
```

只有在更关心小气孔召回、并且能接受约 0.5–0.7 个百分点精确率损失和约 1.5 倍额外推理时间时，才值得用 `nms` 或 `small60` 融合。

---

# English version

**Conclusion: not adopted.** Multi-scale TTA did raise recall of < 40 px stomata by 2.5–3.3 percentage points, but no fusion mode has a higher overall F1 than single-scale inference (best 0.9409 vs baseline 0.9410), while time per image grows by about 1.5×. Default inference stays single-scale `SlicedPredictor`; `stomata/dl/tta.py` is kept as an optional tool.

## Setup

- Model: `models/stomata_yolov8n_p2_human_v2/best.pt`.
- Validation images: 57, excluding `118.2` and `221.2`.
- Tiles: 640 × 640, 20% overlap, 4 CPU cores.
- Matching: one-to-one centre matching, tolerance 0.5 × the image's median box long side.
- Inference: the whole image is first upscaled to 1.25× or 1.5× (bicubic), tiled into the same 640 tiles, and coordinates are divided by the scale factor. 1.25× lies within the ±25% scale augmentation used in training.
- Fusion modes (`stomata/dl/tta.py: fuse_scales`):
  - `nms`: pool boxes from all scales and apply IoS-NMS.
  - `wbf`: cluster across scales and average boxes weighted by score. The score is the mean over scales, counting a scale with no detection as 0.
  - `small{px}`: keep the 1.0× result and use only other-scale boxes whose long side at original scale is below px. If such a box overlaps a 1.0× box, that box's score becomes the larger of the two; otherwise it is added as a new box.
- Every variant is first inferred at threshold 0.05 and then swept over 0.10–0.80; the table reports threshold 0.40 and each variant's best-F1 threshold.

## Results

| Variant | Best threshold | P | R | F1 | Count MAPE | Count bias | Recall < 40 px | Time per image |
|---|---|---|---|---|---|---|---|---|
| **single scale 1.0× (current)** | 0.40 | 0.9368 | 0.9452 | **0.9410** | **6.60%** | +2.7 | 0.704 | **1.3 s** |
| single scale 1.25× | 0.30 | 0.9214 | 0.9456 | 0.9334 | 7.64% | +7.9 | 0.738 | 2.1 s |
| 1.0 + 1.25 `small40` | 0.40 | 0.9366 | 0.9452 | 0.9409 | 6.63% | +2.8 | 0.704 | 3.4 s |
| 1.0 + 1.25 `small60` | 0.40 | 0.9321 | 0.9487 | 0.9403 | 7.12% | — | 0.729 | 3.4 s |
| 1.0 + 1.25 `nms` | 0.40 | 0.9311 | 0.9490 | 0.9400 | 7.23% | +5.8 | 0.729 | 3.4 s |
| 1.0 + 1.25 `wbf` | 0.35 | 0.9307 | 0.9474 | 0.9390 | 7.04% | +5.4 | 0.738 | 3.4 s |
| 1.0 + 1.5 `small40` | 0.40 | 0.9354 | 0.9453 | 0.9403 | 6.75% | — | — | 4.6 s |
| 1.0 + 1.5 `nms` | 0.40 | 0.9310 | 0.9482 | 0.9395 | 7.27% | +5.5 | — | 4.6 s |
| 1.0 + 1.5 `wbf` | 0.25 | 0.9157 | 0.9555 | 0.9352 | 8.61% | +13.0 | — | 4.6 s |

Per-image paired comparisons ([`tta/`](tta/)):

- `nms`: F1 better on 17 images, worse on 32 (p = 0.08).
- `wbf`: 15 better, 36 worse (p = 0.005).
- `small60`: 17 better, 28 worse (p = 0.25).
- 1.25× alone: worse on 50/57 images.

## Why there is no gain

1. **There are too few small stomata to move F1, whatever the method.** Stomata under 40 px number only 240, 1.4% of all 17,092, of which 71 are currently missed. Even recovering all 71 without a single new false positive would raise F1 only from 0.9410 to 0.9432 (+0.22 points). Targeting small stomata alone therefore cannot, even in theory, reach the "F1 gain > 0.5 points" threshold.
2. **Upscaling adds false positives among medium-sized stomata.** With 1.25× alone, precision falls from 0.937 to 0.921. Stomata of 45–60 px become 56–75 px, near the upper edge of the training distribution, and background texture is magnified too, adding false positives. The small stomata recovered by fusion do not offset the new false positives.
3. **Small stomata are usually already detected, just with low scores.** The baseline keeps boxes down to 0.05, and a small 1.25× box almost always coincides with some low-scored 1.0× box. Merely adding new boxes therefore has no effect; the scores of existing boxes have to be raised instead (the current `small` implementation). Moreover, the model's predicted boxes for small stomata tend to be larger than the human boxes, so selecting small objects by predicted size works only with a 50–60 px threshold — which also lets in false positives from medium-sized stomata.

## Usage (optional)

```bash
python scripts/tta_eval.py infer --scale 1.25 -o outputs/eval_human/tta/s1.25
python scripts/tta_eval.py fuse outputs/eval_human/sahi_cmp/sliced_ic0.05 outputs/eval_human/tta/s1.25 \
    --mode nms -o outputs/eval_human/tta/f1.25_nms
python scripts/eval_human.py --from-npz -o outputs/eval_human/tta/f1.25_nms --exclude 118.2 221.2 --conf 0.40
```

`nms` or `small60` fusion is worthwhile only if small-stoma recall matters more and a precision loss of about 0.5–0.7 points plus about 1.5× extra inference time is acceptable.
