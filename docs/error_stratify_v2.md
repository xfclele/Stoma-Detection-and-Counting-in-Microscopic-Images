# 尺寸 × 清晰度分层诊断（v2，57 张验证图，conf 0.40） / Size × sharpness stratified diagnosis (v2, 57 validation images, conf 0.40)

置信度阈值 / threshold 0.4，57 张图 / images，17092 个真值 / GT，总体召回 / overall recall 0.945。
清晰度 = 细频带/粗频带能量比（与对比度无关） / sharpness = fine/coarse band energy ratio (contrast-independent)；四分位界（log） / quartile bounds (log)：-1.57, -1.38, -1.20。

清晰度与尺寸的 / sharpness vs size Spearman ρ = -0.03；与对比度 / vs contrast ρ = -0.06。

## 1. 召回交叉表（单元格：召回 / n） / Recall cross-table (cell: recall / n)

| 尺寸 size \ 清晰度 sharpness | Q1 最模糊 blurriest | Q2 | Q3 | Q4 最清晰 sharpest | 合计 total |
|---|---|---|---|---|---|
| <40 | 0.745 / 51 | 0.767 / 43 | 0.787 / 75 | 0.549 / 71 | 0.704 / 240 |
| 40–50 | 0.882 / 906 | 0.943 / 879 | 0.956 / 944 | 0.956 / 884 | 0.934 / 3613 |
| 50–60 | 0.916 / 1777 | 0.966 / 1865 | 0.966 / 1923 | 0.976 / 1923 | 0.957 / 7488 |
| 60–70 | 0.910 / 1101 | 0.974 / 1137 | 0.972 / 1092 | 0.976 / 1132 | 0.958 / 4462 |
| ≥70 | 0.852 / 438 | 0.931 / 349 | 0.929 / 239 | 0.958 / 263 | 0.909 / 1289 |
| 合计 total | 0.899 / 4273 | 0.959 / 4273 | 0.960 / 4273 | 0.963 / 4273 | 0.945 / 17092 |

## 2. 按尺寸分层的清晰度效应（Q1 最模糊 vs Q2–Q4） / Sharpness effect by size (Q1 blurriest vs Q2–Q4)

| 尺寸 size | Q1 召回 recall | Q2–Q4 召回 recall | 差值 diff | 漏检 OR miss OR |
|---|---|---|---|---|
| <40 | 0.745 | 0.693 | +0.052 | 0.79 |
| 40–50 | 0.882 | 0.952 | -0.070 | 2.63 |
| 50–60 | 0.916 | 0.969 | -0.053 | 2.90 |
| 60–70 | 0.910 | 0.974 | -0.064 | 3.67 |
| ≥70 | 0.852 | 0.939 | -0.087 | 2.67 |

Mantel–Haenszel 合并漏检 OR（控制尺寸） / pooled miss OR (size-adjusted) = **2.76**，CMH p = 1.7e-52。
注意：CMH 检验假定各目标相互独立，而同一张图的目标是相关的，所以这个 p 偏乐观，以下面的聚类稳健模型为准。 Note: CMH assumes independent objects, but objects in one image are correlated, so this p is optimistic; rely on the cluster-robust model below.

## 3. 多因素逻辑回归（因变量 = 漏检，聚类稳健 SE，按图像聚类） / Multivariable logistic regression (outcome = miss, image-clustered robust SE)

| 变量 variable | 漏检 OR miss OR | 95% CI | p |
|---|---|---|---|
| 尺寸 size <40（vs 50–60） | 6.67 | 4.29–10.36 | 3.0e-17 |
| 尺寸 size 40–50（vs 50–60） | 1.46 | 1.20–1.78 | 1.8e-04 |
| 尺寸 size 60–70（vs 50–60） | 0.89 | 0.71–1.12 | 3.3e-01 |
| 尺寸 size ≥70（vs 50–60） | 1.46 | 0.88–2.43 | 1.4e-01 |
| 清晰度 sharpness（每 per 1 SD，log） | 0.59 | 0.48–0.73 | 1.1e-06 |
| 对比度 contrast（每 per 1 SD，log） | 0.65 | 0.58–0.74 | 6.0e-12 |
| 密度 density（每 per 1 SD） | 1.47 | 1.23–1.75 | 1.5e-05 |
| 距边缘 border distance < 25 px | 2.32 | 1.54–3.49 | 5.6e-05 |

## 4. 图像固定效应模型（只用同一张图内部的差异） / Image fixed-effects model (within-image differences only)

| 变量 variable | 漏检 OR miss OR | 95% CI | p |
|---|---|---|---|
| 尺寸 size <40（vs 50–60） | 10.22 | 5.89–17.72 | 1.3e-16 |
| 尺寸 size 40–50（vs 50–60） | 1.88 | 1.52–2.33 | 8.7e-09 |
| 尺寸 size 60–70（vs 50–60） | 0.75 | 0.58–0.97 | 2.7e-02 |
| 尺寸 size ≥70（vs 50–60） | 1.18 | 0.70–1.97 | 5.4e-01 |
| 清晰度 sharpness（每 per 1 SD，log） | 0.47 | 0.41–0.54 | 1.3e-24 |
| 对比度 contrast（每 per 1 SD，log） | 0.37 | 0.30–0.46 | 9.8e-21 |
| 密度 density（每 per 1 SD） | 1.35 | 1.25–1.47 | 1.7e-13 |
| 距边缘 border distance < 25 px | 2.66 | 1.66–4.24 | 4.3e-05 |

## 5. 图像层面 / Image level

每图清晰度中位数 与 每图召回 / per-image median sharpness vs per-image recall：Spearman ρ = 0.23（p = 8.4e-02，n = 57）。

最模糊的 8 张图 / 8 blurriest images：

| 图像 image | 清晰度中位数（log） median sharpness | 召回 recall | 真值数 GT |
|---|---|---|---|
| 309.3 | -1.82 | 0.988 | 246 |
| 23.2_1 | -1.73 | 0.711 | 246 |
| 23.2 | -1.70 | 0.749 | 223 |
| 283.1 | -1.69 | 0.964 | 249 |
| 16.1 | -1.68 | 0.708 | 236 |
| 2.2 | -1.66 | 0.871 | 140 |
| 342.1 | -1.62 | 0.981 | 308 |
| 19.3 | -1.61 | 0.958 | 406 |
