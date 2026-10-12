import numpy as np

from stomata.dl.sliced_inference import BoxDetection
from stomata.dl.tta import fuse_scales, predict_at_scale

A = np.array([[100, 100, 150, 150, 0.9], [300, 300, 330, 330, 0.5]], np.float32)   # 1.0×
B = np.array([[102, 101, 151, 149, 0.7], [500, 500, 530, 530, 0.6],                  # 1.25×：1 个重叠 + 2 个新框
              [800, 800, 880, 880, 0.8]], np.float32)


def test_wbf_merges_across_scales_and_averages_scores():
    f = fuse_scales([A, B], "wbf")
    assert len(f) == 4
    merged = f[np.argmin(np.abs(f[:, 0] - 101))]
    assert np.isclose(merged[4], (0.9 + 0.7) / 2)          # 两个尺度都检出 → 分数取平均
    single = f[np.argmin(np.abs(f[:, 0] - 300))]
    assert np.isclose(single[4], 0.25)                      # 只有一个尺度检出 → 分数减半


def test_nms_keeps_highest():
    f = fuse_scales([A, B], "nms")
    assert len(f) == 4 and f[:, 4].max() == np.float32(0.9)


def test_small_only_adds_small_new_boxes():
    f = fuse_scales([A, B], "small", small_px=40)
    assert len(f) == 3                                      # 只补充 500–530 这个 30 px 新框
    assert any(np.isclose(f[:, 0], 500))


def test_small_boosts_score_of_overlapping_base_box():
    b = np.array([[300, 300, 332, 331, 0.7]], np.float32)  # 小框，与 A 的第二个框重叠
    f = fuse_scales([A, b], "small", small_px=40)
    assert len(f) == 2 and np.isclose(f[1, 4], 0.7)        # 分数 0.5 → 0.7，不新增框


def test_predict_at_scale_maps_back_to_original_coords():
    def fake(img):                                          # 在缩放后的图上固定返回中心附近一个框
        h, w = img.shape[:2]
        return [BoxDetection(w / 2 - 10, h / 2 - 10, w / 2 + 10, h / 2 + 10, 0.8)]

    img = np.zeros((400, 600, 3), np.uint8)
    a = predict_at_scale(fake, img, 1.25)
    cx, cy = (a[0, 0] + a[0, 2]) / 2, (a[0, 1] + a[0, 3]) / 2
    assert abs(cx - 300) < 1 and abs(cy - 200) < 1
    assert abs((a[0, 2] - a[0, 0]) - 16) < 0.1              # 20 px / 1.25
