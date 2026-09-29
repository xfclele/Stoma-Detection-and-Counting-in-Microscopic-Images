"""可视化：流水线对比多子图 + 检测结果叠加图。"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .detector import ClassicalStomataDetector, DetectionResult, DetectorConfig
from .preprocessing import to_gray_float


def draw_detections(image: np.ndarray, result: DetectionResult, show_rejected: bool = False,
                    thickness: int = 2, label: bool = False) -> np.ndarray:
    """在 BGR 图上画出检测椭圆（红）与可选的被拒候选（黄色小点）。"""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if vis.dtype != np.uint8:
        vis = (to_gray_float(vis) * 255).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if show_rejected:
        for d in result.rejected:
            cv2.circle(vis, (int(d.x), int(d.y)), 2, (0, 200, 255), -1)
    for i, d in enumerate(result.detections, start=1):
        # 长轴方向 u=(sinθ, cosθ) 与 x 轴夹角 = 90° - θ
        cv2.ellipse(vis, (int(round(d.x)), int(round(d.y))), (int(d.major_px / 2), int(d.minor_px / 2)),
                    90.0 - d.angle_deg, 0, 360, (0, 0, 255), thickness, cv2.LINE_AA)
        cv2.circle(vis, (int(round(d.x)), int(round(d.y))), 2, (0, 255, 0), -1)
        if label:
            cv2.putText(vis, str(i), (int(d.x) + 4, int(d.y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    return vis


def visualize_pipeline(image: np.ndarray, result: Optional[DetectionResult] = None,
                       config: Optional[DetectorConfig] = None, save_path: Optional[str] = None,
                       zoom: Optional[Sequence[int]] = None, dpi: int = 130, show: bool = False):
    """原图 / 预处理增强图 / 候选热力图+二值掩膜 / 最终检测标注图 的对比多子图。

    参数
    ----
    image   : BGR 或灰度图
    result  : 已有的 DetectionResult；为 None 时内部运行检测
    zoom    : 可选 (x0, y0, x1, y1)，额外增加一行局部放大子图，便于观察气孔裂隙细节
    返回 (fig, result)
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    if result is None:
        result = ClassicalStomataDetector(config).detect(image)
    pre = result.pre
    gray = pre.gray
    overlay = draw_detections(image, result, show_rejected=False, thickness=2)

    # 候选热力图 + 自适应阈值掩膜轮廓 + 分水岭分界线
    heat = cv2.applyColorMap((np.clip(result.response / max(result.response.max(), 1e-6), 0, 1) * 255)
                             .astype(np.uint8), cv2.COLORMAP_INFERNO)
    bnd = find_boundaries(result.labels, mode="inner")
    heat[bnd] = (255, 255, 255)
    heat_rgb = heat[..., ::-1]

    panels = [
        ("① 原图（灰度）", gray, "gray"),
        ("② 预处理：平场校正 + CLAHE", pre.enhanced, "gray"),
        ("③ 候选热力图 (NCC×能量) + 分水岭分区", heat_rgb, None),
        (f"④ 最终检测：{result.count} 个气孔", overlay[..., ::-1], None),
    ]
    _set_cjk_font(plt)
    rows = 2 if zoom is None else 3
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5.4 * rows))
    axes = np.atleast_2d(axes)
    for ax, (title, img, cmap) in zip(axes[:2].ravel(), panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    if zoom is not None:
        x0, y0, x1, y1 = zoom
        axes[2, 0].imshow(pre.enhanced[y0:y1, x0:x1], cmap="gray")
        axes[2, 0].set_title(f"局部放大（增强图）[{x0}:{x1}, {y0}:{y1}]")
        axes[2, 1].imshow(overlay[y0:y1, x0:x1, ::-1])
        axes[2, 1].set_title("局部放大（检测结果）")
        for ax in axes[2]:
            ax.axis("off")
        for ax in (axes[0, 0], axes[1, 1]):
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1.5))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, result


def _set_cjk_font(plt):
    """尽量选用可显示中文的字体；找不到时 matplotlib 会退化为方框，不影响图像内容。"""
    from matplotlib import font_manager
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei",
                 "PingFang SC", "Heiti SC", "Source Han Sans SC"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return
