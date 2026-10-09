import numpy as np
import pytest

from stomata.dl import SlicedPredictor, clip_boxes_to_tile, iter_tiles, points_to_boxes


def test_tiles_cover_image():
    W, H = 1500, 1100
    cover = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in iter_tiles(W, H, 640, 0.2):
        assert x1 - x0 == 640 and y1 - y0 == 640
        cover[y0:y1, x0:x1] = True
    assert cover.all()


def test_clip_boxes_drops_truncated():
    boxes = np.array([[10, 10, 40, 40], [630, 10, 660, 40]], float)
    tb = clip_boxes_to_tile(boxes, (0, 0, 640, 640), min_visible=0.6)
    assert len(tb) == 1


@pytest.mark.parametrize("merge", ["center", "nms"])
def test_sliced_predictor_counts_each_object_once(merge):
    rng = np.random.default_rng(1)
    W, H = 1500, 1100
    pts = []
    while len(pts) < 300:  # 真实气孔互不重叠：最小间距 40 px
        q = rng.uniform([20, 20], [W - 20, H - 20])
        if all(np.hypot(*(q - p)) > 40 for p in pts):
            pts.append(q)
    pts = np.array(pts)
    boxes = points_to_boxes(pts, default_major=36, pad=1.0)
    # 每个像素存放其全局坐标 (x, y)，预测函数据此得知切片位置
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([xx, yy], -1).astype(np.int32)

    def predict_tile(tile):
        """模拟完美检测器：返回切片内可见面积 ≥ 30% 的目标（含被切片边界截断者）。"""
        x0, y0 = int(tile[0, 0, 0]), int(tile[0, 0, 1])
        th, tw = tile.shape[:2]
        b = boxes.copy()
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b[:, [0, 2]] = np.clip(b[:, [0, 2]] - x0, 0, tw)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]] - y0, 0, th)
        vis = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) >= 0.3 * area
        return np.hstack([b[vis], rng.uniform(0.5, 1.0, (vis.sum(), 1))])

    dets = SlicedPredictor(predict_tile, tile=640, overlap=0.2, merge=merge)(img)
    assert len(dets) == len(pts), (merge, len(dets))


def test_build_dataset_single_image(tmp_path):
    import cv2

    from stomata.dl import PointAnnotation, build_yolo_dataset

    img = np.full((1500, 2000, 3), 200, np.uint8)
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), img)
    rng = np.random.default_rng(0)
    pts = rng.uniform([30, 30], [1970, 1470], size=(200, 2))
    y = build_yolo_dataset([PointAnnotation(str(p), pts, None)], str(tmp_path / "ds"), keep_empty=1.0)
    n_train = len(list((tmp_path / "ds/labels/train").iterdir()))
    n_val = len(list((tmp_path / "ds/labels/val").iterdir()))
    assert n_train > 0 and n_val > 0
    assert open(y).read().count("stoma") == 1
    # 标签格式：cls cx cy w h，均归一化到 [0, 1]
    for f in (tmp_path / "ds/labels/train").iterdir():
        for line in f.read_text().splitlines():
            v = [float(t) for t in line.split()]
            assert v[0] == 0 and all(0 <= t <= 1 for t in v[1:])


def test_build_dataset_with_mask(tmp_path):
    import cv2

    from stomata.dl import PointAnnotation, build_yolo_dataset

    img = np.full((1280, 1280, 3), 200, np.uint8)
    img[:, 640:] = 30                        # 右半边是"未标注区域"
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), img)
    mask = np.zeros((1280, 1280), np.uint8)
    mask[:, :640] = 255
    mp = tmp_path / "m.png"
    cv2.imwrite(str(mp), mask)
    pts = np.array([[100.0, 100.0], [300.0, 500.0], [900.0, 300.0]])   # 第三个点在掩膜外
    out = tmp_path / "ds"
    build_yolo_dataset([PointAnnotation(str(p), pts, None, str(mp))], str(out), keep_empty=1.0, overlap=0.0,
                       min_valid_frac=0.3)
    names = sorted(f.stem for f in (out / "labels/train").iterdir()) + sorted(f.stem for f in (out / "labels/val").iterdir())
    assert all(int(n.split("_")[1]) < 640 for n in names)          # 全掩膜的右侧切片被丢弃
    n_boxes = sum(len(f.read_text().splitlines()) for d in ("train", "val") for f in (out / "labels" / d).iterdir())
    assert n_boxes == 2                                             # 掩膜外的点不生成框
