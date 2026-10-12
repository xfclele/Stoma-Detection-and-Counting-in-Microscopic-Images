# 极简标注指南（小样本 / Few-shot） / Minimal Annotation Guide

**语言 / Language：** 中文（下文） · [English](#english-version)

目标：用尽量少的人工（几张整图的校正）得到足以微调 YOLOv8-P2 的训练集。

## 1. 标什么：点，而不是框
* 在每个气孔**中心（裂隙中点）**点一下，不需要画框、多边形或朝向；
* 框由程序自动生成：`边长 = 长轴 × 1.1` 的正方形，长轴来自阶段一的测量或全图中位数（`points_to_boxes`）；
* 点标注比画框快得多，而且一致性更好（中心点远比框边界明确）。

## 2. 标注判定规则（统一口径最重要）

| 情形 | 处理 |
|---|---|
| 清晰、完整的气孔 | 标注 |
| 轻度失焦但能看出两个保卫细胞 | 标注 |
| 严重失焦，只有一团暗影 | 放入 `ignore`（评估时中性） |
| 被图像边缘截断超过一半 | 不标（训练时 `clip_boxes_to_tile` 会自动丢弃截断目标） |
| 两个气孔粘连 | 分别标注两个中心 |
| 细胞壁围成的椭圆小室、气泡、碎屑 | 不标，这些正是需要模型学会拒绝的负样本 |

## 3. 推荐流程："预标注 → 校正"

```bash
# 1) 阶段一生成预标注（点 + 长轴），关闭边缘剔除以免漏掉可见的边缘气孔
python scripts/dl_pipeline.py prelabel images/ -o labels_points/
```

2) 在任意支持点标注的工具中打开 `labels_points/*.json`（格式如下）逐图校正：**删**掉误检，**补**上漏检。
   阶段一的检测点精确率较高、召回有限（全图约一半的明显气孔能检出，见 README），所以校正以"补漏"为主，已有的点大多只需确认。
   CVAT / Label Studio 均支持导入点，可用简单脚本互转：

```json
{"image": "leaf_01.jpg", "points": [[1407.0, 467.0], [1483.0, 486.0]], "major_px": [36.2, 41.0]}
```

3) 构建切片数据集并训练：

```bash
python scripts/dl_pipeline.py build labels_points/ --images images/ -o datasets/stomata
python scripts/dl_pipeline.py train datasets/stomata/data.yaml --epochs 150 --freeze 10
```

## 4. 标多少才够？

| 规模 | 预期 |
|---|---|
| 1 张 2000×1500 整图（≈ 300 个气孔 → ≈ 12 个 640 切片） | 能跑通流程；对新样本（不同叶片/倍率）泛化有限 |
| 3–5 张，覆盖不同叶片 / 焦平面 / 光照 | 小样本微调的推荐起点 |
| 10–20 张，含不同放大倍率 | 可以解冻 backbone 全量训练，作为生产模型 |

单张图也能训练：`build_yolo_dataset` 会自动按空间划分（最右一列切片作验证，与之重叠的训练切片丢弃，防止泄漏）。

## 5. 主动学习迭代
1. 用第 1 轮模型预标注下一批图像；
2. 优先校正**置信度在 0.25–0.5 之间**的检测，以及阶段一/阶段二结果差异大的区域；
3. 合并数据重新训练，重复到新一批图像的校正量足够小为止。

## 6. 评估集
必须留出**至少 1 张完全独立、完整标注**的图像作为测试集（不参与任何调参），
用 `python scripts/count_stomata.py ... --eval test_points.json` 或 `stomata.evaluation.match_points` 计算 P/R/F1。
仓库自带的 `data/annotations/*_partial.json` 只标注了少数小区域，而且参与了阶段一的调参，**不能**当独立测试集用。

---

# English version

Goal: get a training set good enough to fine-tune YOLOv8-P2 with as little manual work as possible (correcting a few whole images).

## 1. What to label: points, not boxes
* Click once at the **centre of each stoma (the middle of the slit)**; no boxes, polygons or orientation are needed;
* boxes are generated automatically as squares with `side = major axis × 1.1`, where the major axis comes from the stage-1 measurement or the image median (`points_to_boxes`);
* point labelling is much faster than drawing boxes and more consistent (a centre is far less ambiguous than a box edge).

## 2. Labelling rules (a consistent convention matters most)

| Case | Action |
|---|---|
| sharp, complete stoma | label it |
| slightly defocused but both guard cells visible | label it |
| badly defocused, only a dark blob | put it in `ignore` (neutral in evaluation) |
| more than half cut off by the image border | do not label (`clip_boxes_to_tile` drops truncated objects in training) |
| two touching stomata | label both centres |
| ellipse-like cells enclosed by cell walls, bubbles, debris | do not label — these are exactly the negatives the model must learn to reject |

## 3. Recommended workflow: "pre-label → correct"

```bash
# 1) generate stage-1 pre-labels (points + major axis); margin exclusion off so visible border stomata are not lost
python scripts/dl_pipeline.py prelabel images/ -o labels_points/
```

2) Open `labels_points/*.json` (format below) in any tool that supports point labels and correct each image: **delete** false positives and **add** misses.
   Stage-1 points have fairly high precision but limited recall (about half of the obvious stomata are found in whole images, see README), so correction is mostly adding misses; existing points mostly just need confirming.
   CVAT and Label Studio both import points; a small script converts between formats:

```json
{"image": "leaf_01.jpg", "points": [[1407.0, 467.0], [1483.0, 486.0]], "major_px": [36.2, 41.0]}
```

3) Build the tile dataset and train:

```bash
python scripts/dl_pipeline.py build labels_points/ --images images/ -o datasets/stomata
python scripts/dl_pipeline.py train datasets/stomata/data.yaml --epochs 150 --freeze 10
```

## 4. How much is enough?

| Amount | Expectation |
|---|---|
| 1 whole 2000×1500 image (≈ 300 stomata → ≈ 12 tiles of 640) | the workflow runs end to end; limited generalisation to new samples (other leaves/magnifications) |
| 3–5 images covering different leaves / focal planes / lighting | recommended starting point for few-shot fine-tuning |
| 10–20 images including different magnifications | the backbone can be unfrozen for full training, as a production model |

A single image also works: `build_yolo_dataset` falls back to a spatial split (the rightmost tile column is validation, and overlapping training tiles are dropped to prevent leakage).

## 5. Active-learning iterations
1. pre-label the next batch with the round-1 model;
2. correct detections with **confidence between 0.25 and 0.5** first, and regions where stage 1 and stage 2 disagree a lot;
3. merge the data and retrain; repeat until a new batch needs little correction.

## 6. Evaluation set
Set aside **at least one fully independent, completely labelled** image as a test set (never used for any tuning),
and compute P/R/F1 with `python scripts/count_stomata.py ... --eval test_points.json` or `stomata.evaluation.match_points`.
The repository's `data/annotations/*_partial.json` label only a few small regions and were used to tune stage 1, so they **cannot** serve as an independent test set.
