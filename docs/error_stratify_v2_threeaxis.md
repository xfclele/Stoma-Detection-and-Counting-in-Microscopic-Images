# 气孔漏检三轴解耦诊断（v2，57 张验证图，conf 0.40，容差 0.35 × 每图中位长轴） / Three-axis diagnosis of missed stomata (v2, 57 validation images, conf 0.40, tolerance 0.35 × image median major axis)

- GT 总数 / total：**17093**，命中 / hits 16149，整体召回 / overall recall **0.945**
- 匹配容差模式 / tolerance mode：`image`

> **怎么读这张表**：在每个尺寸档内按中位数二分。只有当同一尺寸档内，「差」的那一半召回**显著**低于「好」的那一半（Fisher p 小），该因子才算被证实为独立漏检原因，值得为它单独做增强。
> **How to read the tables**: each size bin is split at the median. A factor is confirmed as an independent cause of misses only if, within the same size bin, the "bad" half has a **significantly** lower recall than the "good" half (small Fisher p); only then is a dedicated augmentation worthwhile.
> 多重比较：4 个指标 × 3 个尺寸档 = 12 次检验，Bonferroni 校正后阈值约 p < 0.0042。
> Multiple comparisons: 4 metrics × 3 size bins = 12 tests; the Bonferroni-corrected threshold is about p < 0.0042.

## 1. 按尺寸档的召回（与训练报告口径对齐） / Recall by size bin (as in the training report)

| 尺寸档 size bin | n | 命中 hits | 召回 recall | 95% Wilson CI |
|---|---|---|---|---|
| <40px | 214 | 146 | 0.682 | [0.617, 0.741] |
| 40-70px | 15485 | 14732 | 0.951 | [0.948, 0.955] |
| >=70px | 1394 | 1271 | 0.912 | [0.896, 0.926] |

## 2. 尺寸档内二分：因子是否独立影响召回 / Within-bin split: does the factor independently affect recall?

### 局部清晰度(Laplacian方差) / local sharpness (Laplacian variance)　（低=失焦 / low = defocused）

| 尺寸档 size bin | 低值组 召回 low-group recall (n) | 高值组 召回 high-group recall (n) | 差值 diff | Fisher p |
|---|---|---|---|---|
| <40px | 0.717 (99) | 0.758 (99) | -0.040 | 0.6283 |
| 40-70px | 0.933 (7743) | 0.969 (7742) | -0.036 | <1e-4 |
| >=70px | 0.878 (697) | 0.945 (697) | -0.067 | <1e-4 |

### 局部对比度 / local contrast　（低=灰视野 / low = grey field）

| 尺寸档 size bin | 低值组 召回 low-group recall (n) | 高值组 召回 high-group recall (n) | 差值 diff | Fisher p |
|---|---|---|---|---|
| <40px | 0.730 (100) | 0.745 (98) | -0.015 | 0.8723 |
| 40-70px | 0.941 (9229) | 0.966 (6256) | -0.025 | <1e-4 |
| >=70px | 0.890 (883) | 0.949 (511) | -0.059 | 0.0001 |

### 开口度(短轴/长轴) / opening (minor/major)　（低=细缝 / low = narrow slit）

| 尺寸档 size bin | 低值组 召回 low-group recall (n) | 高值组 召回 high-group recall (n) | 差值 diff | Fisher p |
|---|---|---|---|---|
| <40px | 0.676 (108) | 0.689 (106) | -0.013 | 0.8839 |
| 40-70px | 0.948 (7778) | 0.954 (7707) | -0.006 | 0.0927 |
| >=70px | 0.913 (711) | 0.911 (683) | +0.002 | 0.9249 |

### 邻近度(最近邻距/长轴) / proximity (nearest-neighbour distance/major)　（低=粘连 / low = touching）

| 尺寸档 size bin | 低值组 召回 low-group recall (n) | 高值组 召回 high-group recall (n) | 差值 diff | Fisher p |
|---|---|---|---|---|
| <40px | 0.579 (107) | 0.785 (107) | -0.206 | 0.0019 |
| 40-70px | 0.945 (7743) | 0.958 (7742) | -0.012 | 0.0004 |
| >=70px | 0.895 (697) | 0.928 (697) | -0.033 | 0.0374 |

## 3. 不分档：命中组 vs 漏检组的分布差异（Mann-Whitney U，双侧） / No binning: hit vs missed distributions (two-sided Mann-Whitney U)

| 指标 metric | 命中组中位数 hit median | 漏检组中位数 miss median | p |
|---|---|---|---|
| 局部清晰度(Laplacian方差) / local sharpness (Laplacian variance) | 5.65e+03 (n=16149) | 4.48e+03 (n=928) | <1e-4 |
| 局部对比度 / local contrast | 1 (n=16149) | 0.996 (n=928) | <1e-4 |
| 开口度(短轴/长轴) / opening (minor/major) | 0.804 (n=16149) | 0.794 (n=944) | 0.0026 |
| 邻近度(最近邻距/长轴) / proximity (nearest-neighbour distance/major) | 1.38 (n=16149) | 1.29 (n=944) | <1e-4 |
| 相对清晰度(窗口/本图中位) / relative sharpness (window/image median) | 1 (n=16149) | 0.824 (n=928) | <1e-4 |

## 4. 召回最低的图 / Images with the lowest recall (top 10)

| 图像 image | GT 数 GT | 命中 hits | 召回 recall | 中位长轴 median major | 中位清晰度 median sharpness |
|---|---|---|---|---|---|
| 16.1 | 236 | 167 | 0.708 | 60.5 | 1295.81 |
| 23.2_1 | 246 | 175 | 0.711 | 70.0 | 809.2 |
| 23.2 | 223 | 167 | 0.749 | 52.0 | 1039.58 |
| 292.1 | 423 | 332 | 0.785 | 57.0 | 6233.57 |
| 43.3 | 370 | 307 | 0.830 | 56.0 | 6850.36 |
| 35.2 | 448 | 385 | 0.859 | 53.0 | 4063.63 |
| 46.2 | 288 | 250 | 0.868 | 60.0 | 6580.08 |
| 2.2 | 140 | 122 | 0.871 | 46.0 | 3643.58 |
| 263.2 | 299 | 264 | 0.883 | 59.0 | 5339.11 |
| 127.2 | 274 | 242 | 0.883 | 53.0 | 5734.31 |
