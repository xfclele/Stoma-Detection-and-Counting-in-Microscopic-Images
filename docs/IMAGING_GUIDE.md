# 拍摄规范：让新图像适合 v2 模型 / Imaging Guide: Capturing Images Suited to the v2 Model

**语言 / Language：** 中文（下文） · [English](#english-version)

v2 模型（`models/stomata_yolov8n_p2_human_v2/best.pt`）只见过一种拍摄条件：2560×1920 像素，灰度为主，经过增强处理的 `Stomata_Enhanced` 图像。新图越接近这种条件，验证集上的 P 0.937 / R 0.945 / F1 0.941 就越有参考价值。

本文把建议分为两类：

- **【实测】**：有本项目数据支持；
- **【常规】**：显微拍摄的通用做法，没有在本数据上单独验证。

除特别说明外，数字来自 v2 在 57 张验证图上的诊断（[`error_stratify_v2_threeaxis.md`](error_stratify_v2_threeaxis.md)），以及训练图像和标注的统计。

## 1. 放大倍率与分辨率（最重要）

**【实测】** 训练标注共 92,255 个气孔框，长边中位数 55 px，5%–95% 分位为 42–72 px。v2 按气孔长边分档的召回率：

| 气孔长边 | 占标注 | 召回率 |
|---|---|---|
| < 40 px | 2.4% | **0.68** |
| 40–70 px | 约 91% | **0.95** |
| ≥ 70 px | 6.7% | 0.91 |

建议：

- **尺寸目标：**选放大倍率时，让气孔长边落在 **50–60 px**。按仓库中 20× 测微尺的标定（0.503 µm/px），约 28 µm 长的气孔约为 56 px。
- **换物种：**气孔明显更小就提高倍率，明显更大就降低倍率。不要默认沿用同一个物镜。
- **分辨率：**保持相机原始分辨率 2560×1920，不要缩小或裁小后再保存。
- **标定：**每换一次物镜、相机或适配镜，都拍一张测微尺图（`count_stomata.py --micrometer`），否则无法换算成气孔密度（个/mm²）。

## 2. 对焦（独立影响召回）

**【实测】**

- **同尺寸比较：**气孔大小相同时，清晰度较低的一半召回率低 4–7 个百分点（40–70 px：0.933 对 0.969；≥ 70 px：0.878 对 0.945；p < 1e-4）。
- **训练补不回来：**离焦模糊增强训练没能在不增加误检的前提下补回这部分召回（[`DEFOCUS_AUGMENT.md`](DEFOCUS_AUGMENT.md)），所以只能在拍摄时解决。

建议：

- **焦平面：**对准表皮气孔所在的平面，以保卫细胞轮廓和中间的孔缝清楚为准。
- **样品放平（常规）：**叶片或印膜要压平，盖玻片下不要有气泡和褶皱。叶面弯曲、厚度不均是"中间清楚、边缘模糊"的主要原因。
- **拍完检查：**拍完放大看四个角和中心。如果某一侧明显模糊，换一个更平的视野。
- **景深合成：**能解决对焦不均，但这种合成图模型没见过，使用前先用几张图验证效果。
- **避免运动模糊（常规）：**曝光时间尽量短，用软件或快门线触发，不要碰载物台。

## 3. 照明、曝光与图像处理（对比度独立影响召回）

**【实测】**

- **对比度：**气孔大小相同时，局部对比度较低的一半召回率低 2.5–6 个百分点（p ≤ 1e-4）。
- **召回最低的图对比度都很低：**`16.1`、`23.2` 的召回率只有 0.71–0.75。它们的对比度只有 13–15，训练图的中位数是 44；直方图只占 145–155 个灰阶，训练图通常占满 0–255。它们看起来像**没有做过与其他图相同的增强处理**。
- **训练图本来就有截断：**训练图经过增强后，通常约 10% 的像素是纯黑或纯白（中位数 10.2%，5%–95% 分位为 1.3%–16.7%）。所以对于增强后的图像，截断本身不是问题。

建议：

- **照明（常规）：**用柯勒照明（Köhler）把光调匀。聚光镜孔径光阑开到物镜数值孔径的约 70%–80%。
- **原始图的曝光（常规）：**这里说的是增强之前的原图。背景在中高亮度，气孔轮廓深而不纯黑。原图中纯黑和纯白的像素尽量 < 0.5%，给后续增强留出余量。
- **锁定参数（常规）：**同一批样品锁定曝光时间、增益、白平衡和光源亮度，不要用自动曝光。
- **增强处理一致：**新图必须经过与 `Stomata_Enhanced` 相同的增强处理，不要另外叠加锐化、HDR 或"自动增强"。⚠ 仓库中没有记录训练图用的是哪种增强方法和参数，请向制作 `Stomata_Enhanced` 的人确认，并把方法记录下来。
- **保存格式（常规）：**原始图存 TIFF/PNG 无损格式。只能存 JPEG 时质量 ≥ 90，并关闭相机的降噪和锐化，或固定在同一档。

## 4. 选视野

**【实测】**

- **气孔挤在一起：**小气孔紧挨邻居时召回率只有 0.58，不挨着时为 0.79（p = 0.002）。
- **遮挡多：**大面积被遮挡、气孔零散的图（如 `49.2`）是早期伪标签模型评估中最弱的一张（F1 约 0.78，见 `models/stomata_yolov8n_p2/README.md`）。v2 没有在这张图上评估过。
- **叶脉密集：**`292.1` 的清晰度、对比度都正常，但叶脉和纹理很密，召回率 0.79。这类问题图像指标看不出来。
- **开口大小不影响：**同尺寸下，气孔开口大小对召回没有显著影响（p ≥ 0.09），不必挑气孔张开的时间拍。

建议（常规）：

- **避开干扰：**避开主叶脉、表皮毛、碎屑、气泡和破损区域。
- **有效面积：**每个视野中能看清气孔的表皮面积最好超过 70%。
- **随机取样：**每片叶取 3–5 个互不重叠、随机分布的视野，不要专挑气孔多、清楚的区域，否则密度估计会偏高。

## 5. 拍摄流程

1. **开机准备：**调柯勒照明，拍测微尺图，记录物镜和 µm/px。
2. **试拍定参数：**拍 1 张代表性视野，确认气孔长边约 50–60 px、直方图不截断，然后锁定曝光、增益和白平衡。
3. **逐个视野拍摄：**对焦到表皮，检查四角清晰度，保存无损原图。文件名包含样品、叶片和视野编号。
4. **增强：**按与训练图相同的方法增强。
5. **当场质检：**`python scripts/capture_qc.py 新图文件夹/ -o qc.csv`。标为"重拍"的图，趁样品还在显微镜下重拍。
6. **计数：**`python scripts/dl_pipeline.py predict 新图文件夹/ --weights models/stomata_yolov8n_p2_human_v2/best.pt --conf 0.4 -o outputs_dl`
7. **换条件后先小规模验证：**换物种、显微镜、倍率或制样方法时，先人工标注 5–10 张评估。如果明显下降，就把这批标注加入训练。

## 6. 拍摄后质检脚本 `scripts/capture_qc.py`

| 检查项 | 计算方法 | "重拍"阈值 | "注意"阈值 |
|---|---|---|---|
| 清晰度 | 4×4 分块，细频带/粗频带能量比，与亮度、对比度无关 | 较差分块（25% 分位）< 0.150 | 中位数 < 0.191 |
| 对比度 | 分块粗频带幅度 std(G2−G8) 的中位数 | < 26.6 | < 34.6 |
| 对焦均匀性 | 较差分块 / 中位数 | — | < 0.76 |
| 纯黑/纯白像素 | 灰度 ≤ 2 或 ≥ 253 的比例 | — | > 20% |
| 放大倍率 | 模型预测框长边中位数（检出 ≥ 10 个时才判断） | 超出 45–78 px，或未检出气孔 | — |
| 分辨率 | 是否为 2560×1920 | — | 不同时提示 |

- **阈值来源：**"重拍"阈值取 234 张训练图的 5% 分位，"注意"阈值取 10% 分位。换了参考数据后，可以用 `--calibrate 训练图文件夹` 重新计算。
- **尺寸范围：**由训练标注长边的 5%–95% 分位（42–72 px）乘以 1.08 得到。在验证集上，预测框比人工框平均大约 8%。
- **不跑模型：**加 `--no-model` 只计算图像指标，几秒一张；不加时还会给出检出数和气孔尺寸，每张约 2 s（CPU）。

**【实测】在 59 张验证图上的效果**（其中 57 张有可靠标注，按 v2 的逐图结果统计）：

| 质检结论 | 图数 | 召回率 | 精确率 | F1 | 平均计数误差 |
|---|---|---|---|---|---|
| 合格 | 40 | 0.955 | 0.941 | 0.948 | 5.2% |
| 注意 | 10 | 0.954 | 0.925 | 0.939 | 7.7% |
| **重拍** | 7 | **0.846** | 0.927 | **0.884** | **13.0%** |

说明：

- **"注意"类：**这一类的召回与合格图相近，只有精确率和计数误差略差，所以只作提醒，不要求重拍。
- **样本量小：**被判"重拍"的只有 7 张，而且 `23.2` 和 `23.2_1` 是同一个图像文件，在验证集里被计了两次。阈值本身来自训练图，但选 5% 和 10% 两档时参考过验证集的结果，所以上表可能略偏乐观。
- **看不出的问题：**叶脉密集（如 `292.1`）、气孔挤在一起这类问题，图像指标看不出来，仍需目视检查模型的叠加图。

---

# English version

The v2 model (`models/stomata_yolov8n_p2_human_v2/best.pt`) has only seen one capture condition: 2560×1920-pixel, mostly greyscale, enhanced `Stomata_Enhanced` images. The closer new images are to this condition, the more the validation scores P 0.937 / R 0.945 / F1 0.941 apply to them.

The recommendations fall into two kinds:

- **[measured]**: backed by this project's data;
- **[standard]**: standard microscopy practice, not separately validated on this data.

Unless stated otherwise, numbers come from the diagnosis of v2 on the 57 validation images ([`error_stratify_v2_threeaxis.md`](error_stratify_v2_threeaxis.md)) and from statistics of the training images and labels.

## 1. Magnification and resolution (most important)

**[measured]** The training labels contain 92,255 stoma boxes with a median long side of 55 px and a 5th–95th percentile range of 42–72 px. v2 recall by stoma long side:

| Stoma long side | Share of labels | Recall |
|---|---|---|
| < 40 px | 2.4% | **0.68** |
| 40–70 px | about 91% | **0.95** |
| ≥ 70 px | 6.7% | 0.91 |

Recommendations:

- **Size target:** choose the magnification so that the stoma long side is **50–60 px**. With the repository's 20× micrometer calibration (0.503 µm/px), a stoma about 28 µm long is about 56 px.
- **New species:** use a higher magnification for clearly smaller stomata and a lower one for clearly larger stomata. Do not simply keep the same objective.
- **Resolution:** keep the camera's native 2560×1920 resolution; do not downscale or crop before saving.
- **Calibration:** after every change of objective, camera or adapter, capture a micrometer image (`count_stomata.py --micrometer`); otherwise counts cannot be converted to stomatal density (per mm²).

## 2. Focus (independently affects recall)

**[measured]**

- **Same-size comparison:** for stomata of the same size, the less sharp half has 4–7 points lower recall (40–70 px: 0.933 vs 0.969; ≥ 70 px: 0.878 vs 0.945; p < 1e-4).
- **Training cannot recover it:** defocus blur augmentation did not recover this recall without adding false positives ([`DEFOCUS_AUGMENT.md`](DEFOCUS_AUGMENT.md)), so it has to be solved at capture time.

Recommendations:

- **Focal plane:** focus on the epidermal plane where the stomata lie, judging by sharp guard-cell outlines and pores.
- **Flat sample [standard]:** press leaves or impressions flat, with no bubbles or folds under the coverslip. Curved or uneven samples are the main cause of "sharp centre, blurred edge".
- **Check after capture:** zoom into the four corners and the centre. If one side is clearly blurred, choose a flatter field.
- **Focus stacking:** fixes uneven focus, but the model has never seen stacked images, so validate on a few images before relying on it.
- **Avoid motion blur [standard]:** keep exposure short, trigger from software or a cable release, and do not touch the stage.

## 3. Illumination, exposure and image processing (contrast independently affects recall)

**[measured]**

- **Contrast:** for stomata of the same size, the lower-contrast half has 2.5–6 points lower recall (p ≤ 1e-4).
- **The lowest-recall images all have very low contrast:** `16.1` and `23.2` have recall of only 0.71–0.75. Their contrast is only 13–15, against a training median of 44, and their histograms span only 145–155 grey levels, whereas training images usually span 0–255. They look as if they **did not receive the same enhancement** as the other images.
- **Training images are clipped anyway:** after enhancement, about 10% of the pixels of a typical training image are pure black or white (median 10.2%; 5th–95th percentile 1.3%–16.7%). So for enhanced images, clipping in itself is not a problem.

Recommendations:

- **Illumination [standard]:** use Köhler illumination for even lighting, with the condenser aperture at about 70%–80% of the objective's numerical aperture.
- **Raw exposure [standard]:** this refers to the raw image before enhancement. Keep the background mid-to-bright and the stoma outlines dark but not pure black. Keep pure black plus pure white pixels in the raw image below about 0.5%, leaving headroom for the enhancement.
- **Lock the settings [standard]:** within a batch, lock exposure time, gain, white balance and lamp brightness; do not use auto-exposure.
- **Consistent enhancement:** new images must receive the same enhancement as `Stomata_Enhanced`, with no additional sharpening, HDR or "auto enhance". ⚠ The repository does not record which enhancement method and parameters were used for the training images; please confirm them with whoever produced `Stomata_Enhanced` and document them.
- **File format [standard]:** save raw images as lossless TIFF/PNG. If only JPEG is available, use quality ≥ 90 and turn off in-camera noise reduction and sharpening, or fix them at one setting.

## 4. Choosing fields of view

**[measured]**

- **Crowded stomata:** small stomata touching a neighbour have recall of only 0.58, against 0.79 for those not touching (p = 0.002).
- **Heavy occlusion:** an image with large occluded areas and scattered stomata (`49.2`) was the weakest in the early pseudo-label model's evaluation (F1 about 0.78, see `models/stomata_yolov8n_p2/README.md`); v2 has not been evaluated on it.
- **Dense veins:** `292.1` has normal sharpness and contrast, but dense veins and texture, and its recall is 0.79. Image metrics cannot detect this kind of problem.
- **Opening does not matter:** for stomata of the same size, how open a stoma is has no significant effect on recall (p ≥ 0.09), so there is no need to time captures for open stomata.

Recommendations [standard]:

- **Avoid clutter:** avoid main veins, trichomes, debris, bubbles and damaged areas.
- **Usable area:** in each field, the epidermis with clearly visible stomata should preferably cover more than 70% of the image.
- **Random sampling:** take 3–5 non-overlapping, randomly placed fields per leaf. Do not pick only areas with many clear stomata, or density estimates will be biased upwards.

## 5. Capture workflow

1. **Setup:** adjust Köhler illumination, capture the micrometer, record the objective and µm/px.
2. **Test shot:** capture one representative field, confirm that the stoma long side is about 50–60 px and the histogram is not clipped, then lock exposure, gain and white balance.
3. **Capture each field:** focus on the epidermis, check corner sharpness and save a lossless raw image. Include sample, leaf and field numbers in the file name.
4. **Enhance:** apply the same enhancement as the training images.
5. **Check on the spot:** `python scripts/capture_qc.py new_images/ -o qc.csv`. Retake images marked "retake" while the sample is still under the microscope.
6. **Count:** `python scripts/dl_pipeline.py predict new_images/ --weights models/stomata_yolov8n_p2_human_v2/best.pt --conf 0.4 -o outputs_dl`
7. **Pilot after any change:** when the species, microscope, magnification or preparation changes, first label 5–10 images by hand and evaluate. If performance drops clearly, add these labels to training.

## 6. Post-capture check script `scripts/capture_qc.py`

| Check | How it is computed | "Retake" threshold | "Check" threshold |
|---|---|---|---|
| Sharpness | 4×4 cells, fine-band / coarse-band energy ratio, independent of brightness and contrast | worse cells (25th percentile) < 0.150 | median < 0.191 |
| Contrast | median over cells of the coarse-band amplitude std(G2−G8) | < 26.6 | < 34.6 |
| Focus uniformity | worse cells / median | — | < 0.76 |
| Pure black/white pixels | fraction with grey level ≤ 2 or ≥ 253 | — | > 20% |
| Magnification | median predicted box long side (judged only when ≥ 10 stomata are detected) | outside 45–78 px, or no stomata detected | — |
| Resolution | whether it is 2560×1920 | — | flagged when different |

- **Where the thresholds come from:** "retake" thresholds are the 5th percentile of the 234 training images, and "check" thresholds the 10th percentile. With different reference data, recompute them with `--calibrate training_image_folder`.
- **Size range:** the 5th–95th percentile of training label long sides (42–72 px) multiplied by 1.08, because on the validation set predicted boxes are about 8% larger than human boxes on average.
- **Without the model:** `--no-model` computes only the image metrics, in seconds per image. Without the flag, the script also reports the count and stoma size, at about 2 s per image on CPU.

**[measured] Results on the 59 validation images** (57 of them reliably labelled, using v2's per-image results):

| QC verdict | Images | Recall | Precision | F1 | Mean count error |
|---|---|---|---|---|---|
| OK | 40 | 0.955 | 0.941 | 0.948 | 5.2% |
| Check | 10 | 0.954 | 0.925 | 0.939 | 7.7% |
| **Retake** | 7 | **0.846** | 0.927 | **0.884** | **13.0%** |

Notes:

- **"Check" images:** their recall is similar to OK images; only precision and count error are slightly worse, so this verdict is a reminder and does not require a retake.
- **Small sample:** only 7 images were marked "retake", and `23.2` and `23.2_1` are the same image file, counted twice in the validation set. The thresholds come from the training images, but the choice of the 5% and 10% levels was made after looking at the validation results, so the table may be slightly optimistic.
- **Problems the metrics cannot see:** dense veins (such as `292.1`) and crowded stomata are invisible to image metrics; the model's overlay images still need a visual check.
