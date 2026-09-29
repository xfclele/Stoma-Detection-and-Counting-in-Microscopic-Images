# 阶段二：生产级深度学习方案设计

## 1. 架构选择

### 1.1 检测 vs 密度估计

| 维度 | 小目标检测（YOLOv8/v11 + P2） | 点监督密度估计（CSRNet / FIDTM） |
|---|---|---|
| 输出 | 每个气孔的框 → 中心 (x, y)、尺寸 | 密度图 → 积分得总数；FIDTM 可取局部极大得到点 |
| 长轴直径 | 框尺寸直接换算；OBB 版本还能给出朝向 | 无法直接得到，需要另做分割 |
| 标注 | 点 + 典型尺寸可自动转框（见 `stomata/dl/dataset.py`） | 纯点 |
| 适用密度 | 中低密度、目标边界可分 | 极高密度、严重遮挡（人群、细菌菌落） |
| 推理 | 实时，生态成熟（Ultralytics、SAHI、ONNX/TensorRT） | 较慢，工具链少 |

本任务需要**逐个坐标 + 长轴**，气孔之间基本可分，只有局部粘连，所以推荐**检测路线**。
FIDTM 仅作为备选：极高密度、只关心总数、而且检测器召回明显饱和时再考虑。

### 1.2 为什么需要 P2 检测头

YOLOv8 默认输出 P3/P4/P5（stride 8/16/32）。35 px 的气孔在 P3 上只占约 4×4 个格点，
20 px 的小气孔只有约 2.5×2.5 个格点，正样本分配和定位都受影响。
`yolov8{n,s,m}-p2.yaml` 增加 stride-4 的 P2 输出，小目标特征图分辨率翻倍：

```python
from ultralytics import YOLO
model = YOLO("yolov8s-p2.yaml").load("yolov8s.pt")   # 结构用 P2，形状匹配的层迁移 COCO 权重
```

代价是显存和计算量上升；只有 CPU 或小显存时可以用 `yolov8n-p2`。YOLO11 目前没有官方 P2 配置，
可以仿照 v8-p2 的 head 自行添加。

### 1.3 旋转框（可选）

需要直接回归**朝向 + 长短轴**时，改用 `yolov8s-obb.yaml`（Ultralytics OBB 任务），标注改为旋转框。
阶段一的 `angle_deg / major_px / minor_px` 可直接生成旋转框预标注。

---

## 2. SAHI：切片辅助推理

### 2.1 问题
2000×1500 的图若整体缩放到 640，缩放比约 1/3，35 px 的气孔变成 11 px，中央裂隙（2–3 px）完全消失。
输入必须保持原始分辨率。

### 2.2 机制
1. **切片**：按 640×640、重叠 20% 滑窗切片（`iter_tiles`：最后一行/列贴边对齐，保证覆盖全图）；
2. **逐片推理**：每片都在原始分辨率上，网络看到的气孔尺度与训练一致；
3. **坐标还原**：每片预测加上切片偏移 (x0, y0)，得到全图坐标；
4. **合并**：
   * 相邻切片的重叠区会各自给出同一个气孔 → 需要去重；
   * 切片边界会把气孔截成"半个框"，它和相邻片中的完整框 **IoU 可能只有 0.4–0.5**，但
     **IoS = 交集 / 较小框面积 ≈ 1** → 用 `postprocess_match_metric="IOS"` 更可靠；
   * `GREEDYNMM`（非极大值**合并**）把重复框融合成一个外接框，比 NMS 直接丢弃更稳；
   * 本仓库的 `SlicedPredictor(merge="center")` 还提供**中心归属**规则：每片只保留中心落在其"核心区"
     （与邻片重叠部分各分一半）的框。各片核心区无缝且互不重叠地铺满整图，所以理论上每个目标只计一次，
     再用一次全局 IoS-NMS 兜底（见 `tests/test_dl_utils.py`）。
5. **边缘剔除**：推理后对触碰原图边界的框做 Margin Exclusion（`dl_pipeline.py predict --margin-exclusion`）。

```python
from stomata.dl.sliced_inference import predict_with_sahi
dets = predict_with_sahi("leaf.jpg", "best.pt", tile=640, overlap=0.2, conf=0.25,
                         postprocess="GREEDYNMM", match_metric="IOS", match_threshold=0.5)
```

训练与推理的切片尺寸必须一致（都是 640，原始分辨率），否则尺度分布会不匹配。

---

## 3. 训练配方（小样本）

| 项目 | 取值 | 理由 |
|---|---|---|
| 模型 | yolov8s-p2 + COCO 预训练 | 小目标友好；s 规模在少量数据上不易过拟合 |
| 冻结 | `freeze=10`（backbone） | 20–50 张切片即可收敛；数据增多后可解冻全量微调 |
| 输入 | 640 切片，原始分辨率 | 见 §2 |
| 几何增强 | `degrees=180, flipud=0.5, fliplr=0.5, mosaic=1.0, scale=0.25` | 气孔朝向任意、上下左右对称 |
| 颜色增强 | `hsv_h=0, hsv_s=0, hsv_v=0.3` | 灰度图；只模拟亮度 / 光照不均 |
| 负样本 | 约 30% 无目标切片 | 叶脉、空白区是主要误检来源 |
| 划分 | **按整图**划分训练/验证 | 同一张图的重叠切片分到两边会造成验证集泄漏 |
| 早停 | `patience=50`，`close_mosaic=20` | 最后 20 轮关闭 mosaic，贴近真实分布 |

评估使用本仓库的 `stomata.evaluation`（点匹配 P/R/F1），在**独立的整图**上计算，不要只看 mAP：
计数任务关心的是一对一匹配后的 TP/FP/FN。

---

## 4. 部署建议
* 导出：`model.export(format="onnx", imgsz=640)`，配合内置 `SlicedPredictor` 可以脱离 PyTorch 推理；
* 批量显微图：先按放大倍率分组，不同倍率分别训练或统一重采样到相同 µm/px 后再切片（这是按物理尺度归一化，不是粗暴缩小）；
* 质量控制：把阶段一与阶段二的计数差异 > 20% 的图像列入人工复核队列。
