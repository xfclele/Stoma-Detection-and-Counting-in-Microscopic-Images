import numpy as np

from stomata.calibration import calibrate_stage_micrometer


def _ruler(period=19.9, w=1360, h=400, vertical=True, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 200.0)
    x = np.arange(w)
    for k in range(int(w / period)):
        c = 30 + k * period
        prof = 120 * np.exp(-0.5 * ((x - c) / 1.2) ** 2)
        img[h // 3: 2 * h // 3] -= prof
        if k % 5 == 0:
            img[h // 6: 5 * h // 6] -= prof * 0.5
    img += rng.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img if vertical else img.T.copy()


def test_micrometer_vertical_ticks():
    cal = calibrate_stage_micrometer(_ruler(19.9), division_um=10.0)
    assert cal.orientation == "vertical"
    assert abs(cal.period_px - 19.9) < 0.1
    assert abs(cal.um_per_px - 10 / 19.9) < 0.005


def test_micrometer_horizontal_ticks():
    cal = calibrate_stage_micrometer(_ruler(31.3, vertical=False), division_um=10.0)
    assert cal.orientation == "horizontal"
    assert abs(cal.period_px - 31.3) < 0.15
