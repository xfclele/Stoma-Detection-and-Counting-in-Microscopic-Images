"""阶段一：免训练传统图像处理气孔检测器。

流水线
------
    原图
     │  preprocess(): 平场校正 → CLAHE → DoG 带通 / 黑顶帽暗结构图 / 暗结构能量
     ▼
    自举气孔模板 (templates.bootstrap_template, 完全无监督)
     ▼
    旋转×尺度×长宽比 NCC 响应图   S(x) = max_{θ,s,ρ} NCC(I, T_{θ,s,ρ})(x) · g(x)
     │   g(x) = clip(E(x)/E_90, 0, 1)^γ   暗结构能量门控，压制空白背景上的噪声相关
     │   ※ 大图按重叠分块计算，"拼接响应图"后再全图统一找峰 ⇒ 分块边界上不会重复计数
     ▼
    候选定位：响应图局部极大值
    候选分割：响应图局部自适应阈值 → 距离变换 → 标记控制分水岭（分离粘连气孔）
     ▼
    逐候选精修（±4 px、±5°、细尺度/长宽比）→ 精确中心、方向、尺度
     ▼
    多维硬过滤（全部可配置，见 DetectorConfig）
       ① 分水岭区域几何：面积比、偏心率、凸性 solidity、惯性比 λmin/λmax
       ② 轴向延续比 end_ratio：沿长轴方向 ±0.8L 处响应 / 中心响应。
          气孔是"有端点"的闭合结构 ⇒ 响应在两端衰减；叶脉/平行细胞壁条纹
          沿长轴无限延伸 ⇒ 比值 ≈ 1。这是剔除叶脉误检最有效的判据。
       ③ 结构张量相干度 coherence：C = (λ1-λ2)/(λ1+λ2)，在 σ≈16 px 的邻域积分。
          叶脉/纤维束为单一方向纹理 C→1；气孔及其周围表皮细胞方向杂乱 C 较低。
       ④ 内部暗度对比 darkness：椭圆内平均灰度 - 外环平均灰度（气孔 < 0）。
       ⑤ 射线-椭圆闭合验证：从中心发出 32 条射线，每条寻找
          "内侧亮(保卫细胞) → 暗谷(外壁) → 外侧回升" 的剖面；
          有效射线比例 = 闭合度；对谷点做最小二乘椭圆拟合，拟合残差衡量"是否为椭圆"。
          细胞壁交界点只在少数方向有暗谷 ⇒ 闭合度低 / 残差大。
          椭圆拟合同时给出长轴直径的测量值。
     ▼
    NMS 去重 → 边缘剔除 (margin exclusion) → 结果
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

from .preprocessing import PreprocessResult, preprocess
from .templates import StomaTemplate, bootstrap_template, build_bank, multi_orientation_match, rotate_template


# ============================================================================
# 配置
# ============================================================================
@dataclass
class DetectorConfig:
    # ---- 尺度先验 ----
    expected_major_px: float = 34.0          # 气孔长轴的大致像素长度（20–60 均可），用于自举与种子间距
    expected_major_um: Optional[float] = None  # 若同时给出 um_per_px，则以物理尺寸换算 expected_major_px
    # ---- 预处理 ----
    sigma_bg: float = 40.0
    clahe_clip: float = 2.0
    clahe_tile: int = 16
    dog_sigmas: tuple = (1.0, 10.0)
    line_width: int = 9
    # ---- 模板 ----
    template_path: Optional[str] = None      # 加载已保存的模板（.npz）；None ⇒ 在每张图上自举
    bootstrap_seeds: int = 60
    bootstrap_iters: int = 2
    angle_step: float = 10.0                 # 气孔关于中心 180° 对称，只需搜索 [0,180)
    scales: tuple = (0.9, 1.1, 1.3)          # 相对模板的尺度
    aspects: tuple = (1.0, 1.35)             # 长宽比拉伸（开放/闭合程度不同的气孔）
    crop_factor: float = 0.52                # 模板裁剪半径 = factor × 模板长轴
    match_sigma: float = 1.5                 # 匹配前对增强图的高斯平滑（抑制 CLAHE 放大的噪声）
    energy_gamma: float = 0.5
    # ---- 分块 (大图) ----
    tile_size: int = 2048
    tile_overlap: int = 96                   # 需 > 最大模板半径
    # ---- 候选 ----
    candidate_mode: str = "ncc"              # "ncc": 自举模板旋转匹配响应图（默认）
                                             # "doh": 多尺度 Hessian 暗斑点图（召回高、定位偏差较大）
    doh_scale_factors: tuple = (1 / 4.5, 1 / 3.6, 1 / 3.0)   # σ = factor × 期望长轴
    candidate_threshold: float = 0.20        # 候选图找峰阈值（宽松，保证召回）
    min_distance_factor: float = 0.45        # 种子最小间距 = factor × 期望长轴
    adaptive_sigma_factor: float = 2.0       # 自适应阈值局部均值的高斯 σ = factor × 期望长轴
    adaptive_offset: float = 0.05
    mask_floor: float = 0.15
    # ---- 精修 ----
    refine_scales: tuple = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
    refine_aspects: tuple = (1.0, 1.2, 1.4, 1.6)
    refine_angles: tuple = (-5.0, 0.0, 5.0)
    refine_shift: int = 4
    # ---- 硬过滤（默认值在 data/annotations 中 3 种成像风格的参考区域上联合标定，见 README） ----
    min_score: float = 0.30                  # 精修后的 NCC × 能量门控
    min_blob: float = 0.15                   # DoH 暗斑点强度下限（按全图 99.9 百分位归一化）：气孔整体是"暗椭圆斑"
    area_range: tuple = (0.0, 20.0)          # 分水岭区域面积 / (π/4·长轴·短轴)（响应图区域，宽松）
    max_eccentricity: float = 0.99
    min_solidity: float = 0.30
    min_inertia_ratio: float = 0.0
    min_region_fraction: float = 0.10        # 区域面积比低于此值时跳过①的形状过滤
    max_end_ratio: float = 0.85              # 轴向延续比上限（叶脉 ≈ 1）
    coherence_sigma: float = 16.0
    max_coherence: float = 0.70              # 结构张量相干度上限（平行纹理 → 1）
    max_darkness: float = -0.10              # 内部 - 外环 灰度差上限（需为负，即内部更暗）
    ray_count: int = 32
    ray_contrast: float = 0.06               # 射线剖面 谷深 的最小灰度差
    min_ray_coverage: float = 0.0            # 有效射线比例下限
    max_ellipse_residual: float = 0.30       # 椭圆拟合归一化残差上限
    ellipse_major_range: tuple = (0.5, 2.2)  # 拟合长轴 / 期望长轴 的允许范围
    require_ellipse_fit: bool = False        # True ⇒ 射线椭圆拟合失败的候选直接剔除（默认只用于长轴测量）
    nms_factor: float = 0.75                 # 两检测中心距 < factor × 短轴 ⇒ 视为同一气孔
    # ---- 边界 ----
    margin_exclusion: bool = True
    margin_px: Optional[int] = None          # 额外安全边距（像素）；None ⇒ 0（bbox 模式）/ 0.5×长轴（center 模式）
    margin_mode: str = "bbox"                # "bbox": 旋转椭圆外接框越出图像(或安全边距)即剔除
                                             # "center": 中心距边界 < margin_px 即剔除
    # ---- 物理尺度（可选） ----
    um_per_px: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Detection:
    x: float
    y: float
    score: float
    angle_deg: float          # 长轴相对图像竖直方向的旋转角（逆时针，度）
    major_px: float           # 长轴直径（外轮廓）
    minor_px: float           # 短轴直径
    major_source: str = "template"   # "ellipse"=射线椭圆拟合测量；"template"=模板尺度估计
    blob: float = 0.0         # DoH 斑点强度
    end_ratio: float = 0.0
    coherence: float = 0.0
    darkness: float = 0.0
    ray_coverage: float = 0.0
    ellipse_residual: float = 1.0
    area_ratio: float = 0.0
    eccentricity: float = 0.0
    solidity: float = 0.0
    inertia_ratio: float = 0.0
    edge: bool = False        # 是否触碰图像边缘（margin 规则）
    reason: str = ""          # 被剔除的原因（通过则为空）


@dataclass
class DetectionResult:
    detections: list                    # 通过全部过滤的 Detection
    rejected: list                      # 被过滤掉的候选（Detection.reason 注明原因）
    response: np.ndarray                # 候选热力图
    candidate_mask: np.ndarray          # 自适应阈值掩膜
    labels: np.ndarray                  # 分水岭标签
    pre: PreprocessResult
    template: StomaTemplate
    image_shape: tuple
    extras: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.detections)


# ============================================================================
# 检测器
# ============================================================================
class ClassicalStomataDetector:
    """免训练气孔检测器（见模块文档）。

    用法::

        det = ClassicalStomataDetector(DetectorConfig(expected_major_px=34))
        res = det.detect(cv2.imread("leaf.jpg"))
        print(res.count, [(d.x, d.y, d.major_px) for d in res.detections])
    """

    def __init__(self, config: Optional[DetectorConfig] = None, template: Optional[StomaTemplate] = None):
        self.cfg = config or DetectorConfig()
        self.template = template
        if self.template is None and self.cfg.template_path:
            self.template = StomaTemplate.load(self.cfg.template_path)
        self._fixed_template = self.template is not None
        if self.cfg.expected_major_um and self.cfg.um_per_px:
            self.cfg.expected_major_px = float(self.cfg.expected_major_um / self.cfg.um_per_px)

    # ------------------------------------------------------------------ utils
    def _preprocess(self, image):
        c = self.cfg
        return preprocess(image, c.sigma_bg, c.clahe_clip, c.clahe_tile, tuple(c.dog_sigmas), c.line_width)

    def _match_image(self, pre):
        return cv2.GaussianBlur(pre.enhanced, (0, 0), self.cfg.match_sigma) if self.cfg.match_sigma > 0 else pre.enhanced

    def _crop_radius(self, template):
        return self.cfg.crop_factor * template.major_px

    def _gate(self, energy):
        e90 = np.percentile(energy, 90) + 1e-6
        return np.clip(energy / e90, 0, 1) ** self.cfg.energy_gamma

    def compute_blob_map(self, pre: PreprocessResult):
        """多尺度、尺度归一化 Hessian 行列式 (DoH) 暗斑点图。

        在反相的平场校正图 (1 - flat) 上，气孔整体表现为"暗的椭圆斑"（保卫细胞 + 外壁）。
        DoH(σ) = σ⁴ (Lxx·Lyy - Lxy²)，仅保留 Lxx+Lyy < 0（亮斑）的位置；
        两个主曲率同号且相近 ⇒ 斑点；一大一小（细胞壁、叶脉）⇒ det ≈ 0 被抑制。
        σ 取期望长轴的 1/4.5 ~ 1/3，对应气孔短轴半径附近的尺度。
        返回 (归一化 DoH, 最佳 σ)。
        """
        c = self.cfg
        inv = (1.0 - pre.flat).astype(np.float32)
        best = np.zeros_like(inv)
        best_sig = np.zeros_like(inv)
        for f in c.doh_scale_factors:
            s = float(f * c.expected_major_px)
            g = cv2.GaussianBlur(inv, (0, 0), s)
            dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
            dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
            dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
            r = np.where(dxx + dyy < 0, np.clip((dxx * dyy - dxy ** 2) * s ** 4, 0, None), 0).astype(np.float32)
            upd = r > best
            best[upd], best_sig[upd] = r[upd], s
        return best / (np.percentile(best, 99.9) + 1e-12), best_sig

    # ------------------------------------------------------- response (tiled)
    def compute_response(self, pre: PreprocessResult, template: StomaTemplate):
        """分块计算旋转/尺度/长宽比不变 NCC 响应图并拼接。

        每块向外扩展 overlap 像素计算，只把"核心区"写回全局图，
        拼接结果与整图计算等价，因此不存在块间重复峰（边界目标不会被计两次）。
        """
        c = self.cfg
        img = self._match_image(pre)
        H, W = img.shape
        angles = np.arange(0.0, 180.0, c.angle_step)
        resp = np.full((H, W), -1.0, np.float32)
        best_a = np.zeros((H, W), np.float32)
        best_s = np.ones((H, W), np.float32)
        best_r = np.ones((H, W), np.float32)
        ts, ov = int(c.tile_size), int(c.tile_overlap)
        for y0 in range(0, H, ts):
            for x0 in range(0, W, ts):
                y1, x1 = min(y0 + ts, H), min(x0 + ts, W)
                ya, xa = max(0, y0 - ov), max(0, x0 - ov)
                yb, xb = min(H, y1 + ov), min(W, x1 + ov)
                r, a, s, q = multi_orientation_match(img[ya:yb, xa:xb], template.image, angles,
                                                     c.scales, c.aspects, self._crop_radius(template))
                dst = (slice(y0, y1), slice(x0, x1))
                src = (slice(y0 - ya, y1 - ya), slice(x0 - xa, x1 - xa))
                resp[dst], best_a[dst], best_s[dst], best_r[dst] = r[src], a[src], s[src], q[src]
        resp = np.clip(resp, 0, None) * self._gate(pre.energy)
        return resp.astype(np.float32), best_a, best_s, best_r

    # --------------------------------------------------- candidates/watershed
    def segment_candidates(self, response):
        """局部极大值种子 + 自适应阈值掩膜 + 距离变换分水岭。"""
        c = self.cfg
        L = c.expected_major_px
        min_dist = max(3, int(round(c.min_distance_factor * L)))
        peaks = peak_local_max(response, min_distance=min_dist, threshold_abs=c.candidate_threshold,
                               exclude_border=False)
        # 局部自适应阈值（作用于响应图而非灰度图）：S > G_σ*S + offset 且 S > floor
        local_mean = cv2.GaussianBlur(response, (0, 0), c.adaptive_sigma_factor * L)
        mask = (response > local_mean + c.adaptive_offset) & (response > c.mask_floor)
        markers = np.zeros(response.shape, np.int32)
        for i, (r, cc) in enumerate(peaks, start=1):
            markers[r, cc] = i
        mask |= cv2.dilate((markers > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        # 以 -距离 为地形、响应峰为标记：两个粘连气孔在"颈部"距离值小，分水岭线落在颈部
        dist = ndi.distance_transform_edt(mask)
        labels = watershed(-dist, markers, mask=mask)
        return peaks, mask, labels

    # -------------------------------------------------- per-candidate checks
    def _refine(self, img, template, x, y, ang0):
        """局部精修：±refine_shift 像素内搜索 角度×尺度×长宽比。
        返回 (ncc, x, y, angle, scale, aspect)；越界时返回 None。"""
        c = self.cfg
        H, W = img.shape
        d = int(c.refine_shift)
        best = None
        for s, asp, Ts in build_bank(template.image, c.refine_scales, c.refine_aspects, self._crop_radius(template)):
            h = Ts.shape[0] // 2
            y0, y1, x0, x1 = int(y) - h - d, int(y) + h + d + 1, int(x) - h - d, int(x) + h + d + 1
            if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
                continue
            roi = img[y0:y1, x0:x1]
            for da in c.refine_angles:
                r = cv2.matchTemplate(roi, rotate_template(Ts, ang0 + da), cv2.TM_CCOEFF_NORMED)
                r[~np.isfinite(r)] = -1
                k = int(r.argmax())
                v = float(r.flat[k])
                if best is None or v > best[0]:
                    dy, dx = divmod(k, r.shape[1])
                    best = (v, x0 + h + dx, y0 + h + dy, ang0 + da, s, asp)
        return best

    @staticmethod
    def axes_dirs(angle_deg):
        """模板旋转 θ 后：长轴单位向量 u=(sinθ, cosθ)，短轴 v=(cosθ, -sinθ)（图像坐标，y 向下）。"""
        t = np.radians(angle_deg)
        return (np.sin(t), np.cos(t)), (np.cos(t), -np.sin(t))

    def _end_ratio(self, response, d):
        H, W = response.shape
        (ux, uy), _ = self.axes_dirs(d.angle_deg)
        L = 0.8 * d.major_px

        def at(x, y):
            return response[int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))]

        c0 = at(d.x, d.y) + 1e-6
        return float(max(at(d.x + L * ux, d.y + L * uy), at(d.x - L * ux, d.y - L * uy)) / c0)

    def _darkness(self, img, d):
        H, W = img.shape
        R = int(d.major_px) + 4
        x0, x1, y0, y1 = max(0, int(d.x) - R), min(W, int(d.x) + R + 1), max(0, int(d.y) - R), min(H, int(d.y) + R + 1)
        m_in = np.zeros((y1 - y0, x1 - x0), np.uint8)
        m_out = m_in.copy()
        ctr = (int(d.x) - x0, int(d.y) - y0)
        ax = (max(1, int(d.minor_px / 2)), max(1, int(d.major_px / 2)))
        cv2.ellipse(m_in, ctr, ax, -d.angle_deg, 0, 360, 1, -1)
        cv2.ellipse(m_out, ctr, (int(ax[0] * 1.5), int(ax[1] * 1.4)), -d.angle_deg, 0, 360, 1, -1)
        m_out[m_in > 0] = 0
        p = img[y0:y1, x0:x1]
        if not m_in.any() or not m_out.any():
            return 0.0
        return float(p[m_in > 0].mean() - p[m_out > 0].mean())

    def _ray_ellipse(self, img_s, x, y, r_max):
        """射线剖面 + 椭圆拟合（见模块文档 ⑤）。

        返回 (coverage, fitted_major, fitted_minor, residual)；拟合失败时 major=0, residual=1。
        残差 = mean | sqrt((u/a)^2 + (v/b)^2) - 1 |，即谷点到拟合椭圆的归一化径向偏差。
        """
        c = self.cfg
        n = c.ray_count
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rad = np.arange(3.0, r_max, 0.5)
        px = (x + np.cos(phi)[:, None] * rad[None]).astype(np.float32)
        py = (y + np.sin(phi)[:, None] * rad[None]).astype(np.float32)
        P = cv2.remap(img_s, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        lo = int(np.searchsorted(rad, 7.0))     # 跳过中心裂隙
        pts = []
        for k in range(n):
            p = P[k]
            j = lo + int(np.argmin(p[lo:]))
            if j >= len(rad) - 2:
                continue
            inside = p[2:j].max() if j > 2 else p[j]
            outside = p[j:min(len(rad), j + 16)].max()
            if inside - p[j] > c.ray_contrast and outside - p[j] > c.ray_contrast:
                pts.append((px[k, j], py[k, j]))
        cov = len(pts) / n
        if len(pts) < 8:
            return cov, 0.0, 0.0, 1.0
        pts = np.asarray(pts, np.float32)
        (ex, ey), (d1, d2), ang = cv2.fitEllipse(pts)
        if not np.all(np.isfinite([ex, ey, d1, d2])) or min(d1, d2) < 2:
            return cov, 0.0, 0.0, 1.0
        t = np.radians(ang)
        u = (pts[:, 0] - ex) * np.cos(t) + (pts[:, 1] - ey) * np.sin(t)
        v = -(pts[:, 0] - ex) * np.sin(t) + (pts[:, 1] - ey) * np.cos(t)
        res = float(np.mean(np.abs(np.sqrt((u / (d1 / 2)) ** 2 + (v / (d2 / 2)) ** 2) - 1)))
        return cov, float(max(d1, d2)), float(min(d1, d2)), res

    # ----------------------------------------------------------------- detect
    def detect(self, image) -> DetectionResult:
        c = self.cfg
        pre = self._preprocess(image)
        template = self.template if self._fixed_template else bootstrap_template(
            pre.enhanced, pre.ridge, pre.energy, c.expected_major_px,
            c.bootstrap_seeds, c.bootstrap_iters, angle_step=15.0)
        self.template = template

        response, best_a, best_s, best_r = self.compute_response(pre, template)
        blob, _ = self.compute_blob_map(pre)
        cand_map = blob if c.candidate_mode == "doh" else response
        peaks, mask, labels = self.segment_candidates(cand_map)
        props = {p.label: p for p in regionprops(labels)}

        gate = self._gate(pre.energy)
        img_m = self._match_image(pre)
        img_s = cv2.GaussianBlur(pre.enhanced, (0, 0), 1.5)
        coh = structure_coherence(cv2.GaussianBlur(pre.enhanced, (0, 0), 1.0), c.coherence_sigma)
        H, W = pre.enhanced.shape
        L = c.expected_major_px
        cands = []
        for i, (r, cc) in enumerate(peaks, start=1):
            ref = self._refine(img_m, template, float(cc), float(r), float(best_a[r, cc]))
            if ref is None:  # 太靠近图像边缘无法精修 ⇒ 用粗匹配结果
                x, y, ang, s, asp = float(cc), float(r), float(best_a[r, cc]), float(best_s[r, cc]), float(best_r[r, cc])
                score = float(response[r, cc])
            else:
                v, x, y, ang, s, asp = ref
                x, y = float(x), float(y)
                score = float(max(v, 0.0) * gate[int(y), int(x)])
            d = Detection(x, y, score, float(ang) % 180.0, template.major_px * s, template.minor_px * s / asp)
            d.blob = float(blob[int(r), int(cc)])
            d.end_ratio = self._end_ratio(response, d)
            d.coherence = float(coh[int(y), int(x)])
            d.darkness = self._darkness(pre.enhanced, d)
            cov, emaj, emin, eres = self._ray_ellipse(img_s, x, y, r_max=0.9 * max(d.major_px, L))
            d.ray_coverage, d.ellipse_residual = cov, eres
            ell_ok = emaj > 0 and eres <= c.max_ellipse_residual
            if ell_ok and c.ellipse_major_range[0] * L <= emaj <= c.ellipse_major_range[1] * L:
                d.major_px, d.minor_px, d.major_source = emaj, emin, "ellipse"

            p = props.get(i)
            if p is not None:
                d.area_ratio = float(p.area / (np.pi / 4.0 * d.major_px * d.minor_px))
                d.eccentricity = float(p.eccentricity)
                d.solidity = float(p.solidity)
                ev = p.inertia_tensor_eigvals
                d.inertia_ratio = float(ev[1] / ev[0]) if ev[0] > 0 else 0.0
            d.reason = self._check(d, ell_ok or not c.require_ellipse_fit, W, H)
            cands.append(d)

        kept = self._nms([d for d in cands if not d.reason])
        kept_ids = {id(d) for d in kept}
        for d in cands:
            if not d.reason and id(d) not in kept_ids:
                d.reason = "duplicate"
        rejected = [d for d in cands if d.reason]
        return DetectionResult(kept, rejected, cand_map, mask, labels, pre, template, (H, W),
                               extras={"best_angle": best_a, "best_scale": best_s, "best_aspect": best_r,
                                       "coherence": coh, "n_candidates": len(peaks),
                                       "ncc_response": response, "blob_map": blob})

    # ----------------------------------------------------------------- helpers
    def _check(self, d: Detection, ell_ok: bool, W: int, H: int) -> str:
        """按顺序执行硬过滤，返回第一个未通过的原因；全部通过返回空串。"""
        c = self.cfg
        # 分水岭区域过小（< 0.1 × 期望椭圆面积）时其形状统计不可靠（退化为线段 ⇒ 偏心率≡1），不参与几何过滤
        has_region = d.area_ratio >= c.min_region_fraction
        rules = [
            ("score", d.score >= c.min_score),
            ("blob", d.blob >= c.min_blob),
            ("area", not has_region or c.area_range[0] <= d.area_ratio <= c.area_range[1]),
            ("eccentricity", not has_region or d.eccentricity <= c.max_eccentricity),
            ("solidity", not has_region or d.solidity >= c.min_solidity),
            ("inertia", not has_region or d.inertia_ratio >= c.min_inertia_ratio),
            ("end_ratio", d.end_ratio <= c.max_end_ratio),
            ("coherence", d.coherence <= c.max_coherence),
            ("darkness", d.darkness <= c.max_darkness),
            ("ray_coverage", d.ray_coverage >= c.min_ray_coverage),
            ("ellipse_fit", ell_ok),
        ]
        for name, ok in rules:
            if not ok:
                return name
        if self._touches_edge(d, W, H):
            d.edge = True
            if c.margin_exclusion:
                return "margin"
        return ""

    def _touches_edge(self, d: Detection, W: int, H: int) -> bool:
        c = self.cfg
        if c.margin_mode == "center":
            m = c.margin_px if c.margin_px is not None else 0.5 * d.major_px
            return d.x < m or d.y < m or d.x > W - 1 - m or d.y > H - 1 - m
        # 旋转椭圆的轴对齐外接框半宽/半高：长轴沿 u=(sinθ, cosθ)
        t = np.radians(d.angle_deg)
        a, b = d.major_px / 2, d.minor_px / 2
        hx = np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)
        hy = np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
        m = c.margin_px if c.margin_px is not None else 0
        return d.x - hx < m or d.y - hy < m or d.x + hx > W - 1 - m or d.y + hy > H - 1 - m

    def _nms(self, dets):
        """中心距离 NMS：两检测中心距 < nms_factor × 较小短轴 ⇒ 同一气孔，保留得分高者。"""
        dets = sorted(dets, key=lambda d: -d.score)
        out = []
        f = self.cfg.nms_factor
        for d in dets:
            if all((d.x - o.x) ** 2 + (d.y - o.y) ** 2 >= (f * min(d.minor_px, o.minor_px)) ** 2 for o in out):
                out.append(d)
        return out


def structure_coherence(img: np.ndarray, sigma: float) -> np.ndarray:
    """结构张量相干度 C = sqrt((Jxx-Jyy)^2 + 4Jxy^2) / (Jxx+Jyy)  ∈ [0, 1]。"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    return (np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-6)).astype(np.float32)
