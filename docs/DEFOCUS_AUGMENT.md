# 离焦模糊增强实验（v3） / Defocus Blur Augmentation Experiment (v3)

**语言 / Language：** 中文（下文） · [English](#english-version)

**结论：离焦增强在同等训练条件下确实提高了模糊和小气孔的召回，但会让精确率下降约 1 个百分点，整体 F1 和计数误差都不如 v2。推荐模型仍为 `models/stomata_yolov8n_p2_human_v2/best.pt`（conf 0.40）。**

## 1. 动机

分层诊断（[`error_stratify_v2.md`](error_stratify_v2.md)、[`error_stratify_v2_threeaxis.md`](error_stratify_v2_threeaxis.md)）表明：在同一尺寸档内、甚至同一张图内，清晰度最低的 1/4 气孔召回为 0.899，其余约 0.96，而且这一效应与对比度相互独立。

## 2. 增强方法

实现见 `stomata/dl/defocus.py`，训练脚本为 `scripts/train_defocus.py`。增强只作用于训练集，不改标签。

- **点扩散函数（PSF）**：70% 用圆盘核（几何光学离焦），30% 用高斯核。
- **局部与整片**：70% 只对平滑随机掩膜覆盖的局部区域施加模糊，模拟叶面起伏造成的部分离焦；30% 整片模糊。
- **附带退化**：可选降低对比度，并加入少量噪声。
- **半径标定**：以清晰度指标（细频带与粗频带的能量比）为准。验证集中最模糊的几张图，清晰度相当于在清晰训练图上施加半径 1.5–2 px 的圆盘模糊，因此半径上限取 2.5–3.5 px。

## 3. 实验

所有实验都在 CPU 上进行，每段限时 0.85 小时，lr0 0.0005，关闭 mosaic，其余超参与 v2 相同。评估在 57 张验证图上做整图推理，阈值 0.40。

| 运行 | 起点 | 离焦增强 | P | R | F1 | 计数 MAPE | 计数偏差 | 最模糊 1/4 召回 | <40 px 召回 |
|---|---|---|---|---|---|---|---|---|---|
| **v2（现行）** | v1 | 无 | **0.937** | 0.945 | **0.941** | **6.6%** | +2.7 | 0.899 | 0.704 |
| 对照 | v2 | 无 | 0.931 | 0.942 | 0.936 | 7.2% | +3.8 | 0.890 | 0.721 |
| v3a | v2 | p 0.5，半径 1–3.5，降对比度 | 0.928 | 0.947 | 0.937 | 7.3% | +6.1 | 0.905 | 0.725 |
| v3b | v2 | p 0.25，半径 1–2.5，不降对比度 | 0.925 | 0.948 | 0.937 | 7.7% | +7.3 | 0.902 | 0.742 |
| v3（v1 起，与 v2 同流程） | v1 | 同 v3a | 0.928 | 0.947 | 0.937 | 7.5% | +6.3 | **0.906** | **0.742** |

完整对比表见 [`defocus/`](defocus/)，由 `scripts/compare_runs.py` 生成，其中给出逐图 Wilcoxon 检验和按图像 bootstrap 的 95% 置信区间。

## 4. 分析

1. **从 v2 继续训练本身就会让指标回落。**对照组不加增强，F1 也从 0.941 降到 0.936（47/57 张图更差，p = 6e-7）。v2 是在同一验证集上挑出的最佳检查点，因此带有选择偏差，它的 0.941 略偏乐观。
2. **和同等条件的对照相比，增强有净效应。**
   - 最模糊 1/4 的召回：v3a +1.4、v3b +1.2 个百分点，95% CI 均不含 0。
   - 其他清晰度档和尺寸档的召回不降反升，清晰目标的召回没有受损。
   - v3a 的逐图 F1 优于对照（37 张更好、17 张更差，p = 0.012），计数误差与对照持平（p = 0.70）。
3. **和 v2 直接比较，增强模型仍然更差。**按 v2 的流程从 v1 训练并加入增强，相对 v2 的结果是：
   - 召回：最模糊 1/4 +0.7 个百分点（CI +0.2 到 +1.3），<40 px +3.7 个百分点（CI +1.6 到 +6.1），40–50 px +1.0 个百分点；
   - 精确率：0.937 → 0.928；
   - F1 更差（42/57 张图，p = 3e-5），计数误差更差（p = 0.004），计数偏差从 +2.7 增大到 +6.3。
   - 换任何阈值都不能消除这一差距：召回相同时，精确率始终低约 1 个百分点。
4. **新增误检的来源。**v3a 比 v2 多出 252 个误检，其中一半位于最模糊的 1/4 区域。抽查 48 个，大多是模糊的纹理、条纹和杂质，只有约四分之一到三分之一像是人工漏标的模糊气孔。也就是说，增强把模型对"模糊目标"的判定放宽了，召回和精确率之间的取舍并没有改善。

## 5. 结论与建议

- 推荐模型仍为 **v2**。增强版存档在 `models/stomata_yolov8n_p2_human_v3_defocus/`，适合更看重模糊或小气孔召回、能接受约 1 个百分点精确率损失的场景。如果用它计数，建议把阈值提高到 0.45，计数偏差会接近 0（MAPE 6.5%）。
- 失焦对召回的独立效应是真实的，但在当前数据上只影响约 1/4 目标中的几个百分点。CPU 训练预算下，离焦增强带来的收益抵不过它造成的精确率损失。
- 若要进一步改进，有以下几个方向：
  - 补标模糊区域的漏标气孔。如果误检中确有一部分是漏标的真气孔，精确率被低估了。
  - 用一组独立测试图消除 v2 的选择偏差，再比较各个模型。
  - 分辨率与小目标：增强版在 <40 px 上的稳定提升（+3.7 个百分点），说明小目标仍有改进空间。
- 所有阈值和模型选择都基于同一个验证集，以上差异只适合做相对比较。

---

# English version

**Conclusion: under matched training conditions, defocus augmentation does raise recall for blurred and small stomata, but it lowers precision by about 1 percentage point, and overall F1 and count error are both worse than v2. The recommended model remains `models/stomata_yolov8n_p2_human_v2/best.pt` (conf 0.40).**

## 1. Motivation

The stratified diagnosis ([`error_stratify_v2.md`](error_stratify_v2.md), [`error_stratify_v2_threeaxis.md`](error_stratify_v2_threeaxis.md)) showed that within the same size bin, and even within the same image, the blurriest quarter of stomata has a recall of 0.899 versus about 0.96 for the rest, and that this effect is independent of contrast.

## 2. Augmentation

Implemented in `stomata/dl/defocus.py`; training script `scripts/train_defocus.py`. The augmentation applies only to the training set and does not change labels.

- **Point-spread function (PSF)**: a disk kernel (geometric-optics defocus) 70% of the time, a Gaussian kernel 30%.
- **Local vs whole tile**: 70% blur only the local area covered by a smooth random mask, mimicking partial defocus from an uneven leaf surface; 30% blur the whole tile.
- **Extra degradation**: optional contrast reduction plus a little noise.
- **Radius calibration**: based on the sharpness metric (fine-band to coarse-band energy ratio). The blurriest validation images have the sharpness of a sharp training image blurred with a 1.5–2 px disk, so the maximum radius is 2.5–3.5 px.

## 3. Experiments

All runs on CPU, 0.85 h per segment, lr0 0.0005, mosaic off, other hyper-parameters as for v2. Evaluation: whole-image inference on the 57 validation images, threshold 0.40.

| Run | Start | Defocus augmentation | P | R | F1 | Count MAPE | Count bias | Recall, blurriest quarter | Recall, < 40 px |
|---|---|---|---|---|---|---|---|---|---|
| **v2 (current)** | v1 | none | **0.937** | 0.945 | **0.941** | **6.6%** | +2.7 | 0.899 | 0.704 |
| control | v2 | none | 0.931 | 0.942 | 0.936 | 7.2% | +3.8 | 0.890 | 0.721 |
| v3a | v2 | p 0.5, radius 1–3.5, contrast reduction | 0.928 | 0.947 | 0.937 | 7.3% | +6.1 | 0.905 | 0.725 |
| v3b | v2 | p 0.25, radius 1–2.5, no contrast reduction | 0.925 | 0.948 | 0.937 | 7.7% | +7.3 | 0.902 | 0.742 |
| v3 (from v1, same recipe as v2) | v1 | as v3a | 0.928 | 0.947 | 0.937 | 7.5% | +6.3 | **0.906** | **0.742** |

Full comparison tables are in [`defocus/`](defocus/), generated by `scripts/compare_runs.py`, with per-image Wilcoxon tests and image-level bootstrap 95% confidence intervals.

## 4. Analysis

1. **Continuing training from v2 by itself lowers the metrics.** The control without augmentation also falls from F1 0.941 to 0.936 (worse on 47/57 images, p = 6e-7). v2 is the best checkpoint selected on the same validation set, so it carries selection bias and its 0.941 is slightly optimistic.
2. **Against a matched control, the augmentation has a net effect.**
   - Recall of the blurriest quarter: v3a +1.4 and v3b +1.2 percentage points, 95% CIs both excluding 0.
   - Recall in the other sharpness and size bins does not drop and even rises slightly; recall of sharp objects is not harmed.
   - v3a's per-image F1 beats the control (37 better, 17 worse, p = 0.012), with count error equal to the control (p = 0.70).
3. **Compared directly with v2, the augmented model is still worse.** Trained from v1 with v2's recipe plus augmentation, relative to v2:
   - recall: blurriest quarter +0.7 points (CI +0.2 to +1.3), < 40 px +3.7 points (CI +1.6 to +6.1), 40–50 px +1.0 point;
   - precision: 0.937 → 0.928;
   - F1 worse (42/57 images, p = 3e-5), count error worse (p = 0.004), count bias up from +2.7 to +6.3.
   - No threshold removes the gap: at equal recall, precision is always about 1 point lower.
4. **Where the new false positives come from.** v3a has 252 more false positives than v2, half of them in the blurriest quarter. Of 48 inspected, most were blurred texture, stripes and debris; only about a quarter to a third looked like blurred stomata the annotators missed. In other words, the augmentation loosened the model's criterion for blurred objects without improving the recall–precision trade-off.

## 5. Conclusions and recommendations

- The recommended model remains **v2**. The augmented model is archived in `models/stomata_yolov8n_p2_human_v3_defocus/`, for cases that value recall of blurred or small stomata and can accept about 1 point lower precision. For counting with it, raise the threshold to 0.45, which brings the count bias close to 0 (MAPE 6.5%).
- The independent effect of defocus on recall is real, but on the current data it affects only a few points within about a quarter of objects. Under a CPU training budget, the gain from defocus augmentation does not outweigh the precision loss.
- Possible further directions:
  - label the missed stomata in blurred areas: if some false positives really are unlabelled stomata, precision is underestimated;
  - use an independent test set to remove v2's selection bias before comparing models;
  - resolution and small objects: the augmented model's consistent gain on < 40 px (+3.7 points) shows there is still room to improve small objects.
- All thresholds and model choices are based on the same validation set; the differences above are suitable only for relative comparison.
