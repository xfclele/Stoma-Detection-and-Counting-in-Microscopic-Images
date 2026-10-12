# 模型对比 / Model comparison: sliced_ic0.05（conf 0.4） vs sahi_gnmm_ic0.05（conf 0.4）

57 张验证图 / validation images.

| 模型 Model | P | R | F1 | 计数 Count MAE | 计数 Count MAPE | 计数偏差 Count bias |
|---|---|---|---|---|---|---|
| sliced_ic0.05 | 0.9368 | 0.9452 | 0.9410 | 18.53 | 6.6% | +2.67 |
| sahi_gnmm_ic0.05 | 0.9341 | 0.9471 | 0.9405 | 18.89 | 6.77% | +4.16 |

- 逐图 F1 / per-image F1：sahi_gnmm_ic0.05 更好 better 17 / 更差 worse 23 / 持平 tied 17，Wilcoxon p = 0.177
- 逐图 |相对计数误差| / per-image |relative count error|：sahi_gnmm_ic0.05 更小 smaller 13 / 更大 larger 27，Wilcoxon p = 0.158

## 按清晰度四分位的召回 / Recall by sharpness quartile

| 分组 Group | n | sliced_ic0.05 召回 recall | sahi_gnmm_ic0.05 召回 recall | 差值 Diff | 95% CI（按图 bootstrap / image bootstrap） |
|---|---|---|---|---|---|
| Q1 最模糊 blurriest | 4273 | 0.899 | 0.902 | +0.004 | [+0.001, +0.006] |
| Q2 | 4273 | 0.959 | 0.960 | +0.002 | [+0.000, +0.003] |
| Q3 | 4273 | 0.960 | 0.961 | +0.001 | [+0.000, +0.002] |
| Q4 最清晰 sharpest | 4273 | 0.963 | 0.965 | +0.001 | [+0.000, +0.003] |

## 按尺寸档的召回 / Recall by size bin

| 分组 Group | n | sliced_ic0.05 召回 recall | sahi_gnmm_ic0.05 召回 recall | 差值 Diff | 95% CI（按图 bootstrap / image bootstrap） |
|---|---|---|---|---|---|
| <40 | 240 | 0.704 | 0.717 | +0.012 | [+0.000, +0.027] |
| 40–50 | 3613 | 0.934 | 0.936 | +0.002 | [+0.001, +0.004] |
| 50–60 | 7488 | 0.957 | 0.958 | +0.002 | [+0.001, +0.003] |
| 60–70 | 4462 | 0.958 | 0.959 | +0.001 | [-0.000, +0.002] |
| ≥70 | 1289 | 0.909 | 0.913 | +0.004 | [+0.000, +0.009] |

## Q1 最模糊 × 尺寸 / Q1 blurriest × size

| 分组 Group | n | sliced_ic0.05 召回 recall | sahi_gnmm_ic0.05 召回 recall | 差值 Diff | 95% CI（按图 bootstrap / image bootstrap） |
|---|---|---|---|---|---|
| Q1, <40 | 51 | 0.745 | 0.765 | +0.020 | [+0.000, +0.068] |
| Q1, 40–50 | 906 | 0.882 | 0.884 | +0.002 | [+0.000, +0.006] |
| Q1, 50–60 | 1777 | 0.916 | 0.919 | +0.003 | [+0.000, +0.006] |
| Q1, 60–70 | 1101 | 0.910 | 0.914 | +0.004 | [+0.000, +0.008] |
| Q1, ≥70 | 438 | 0.852 | 0.858 | +0.007 | [+0.000, +0.018] |
