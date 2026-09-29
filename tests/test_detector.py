import os

import numpy as np
import pytest

from stomata import DetectorConfig, StomataCounter, match_points
from stomata.detector import ClassicalStomataDetector

from .synthetic import make_synthetic


@pytest.fixture(scope="module")
def synth():
    return make_synthetic(seed=3)


@pytest.fixture(scope="module")
def result(synth):
    img, _, _ = synth
    return ClassicalStomataDetector(DetectorConfig(expected_major_px=35, bootstrap_seeds=15)).detect(img)


def test_synthetic_detection_quality(synth, result):
    _, pts, _ = synth
    m = match_points([(d.x, d.y) for d in result.detections], pts, tol=10)
    assert m.recall >= 0.8, m.summary()
    assert m.precision >= 0.8, m.summary()


def test_major_axis_estimate(synth, result):
    _, pts, majors = synth
    m = match_points([(d.x, d.y) for d in result.detections], pts, tol=10)
    est = np.array([result.detections[i].major_px for i, _, _ in m.matches])
    true = majors[[j for _, j, _ in m.matches]]
    rel = np.median(np.abs(est - true) / true)
    assert rel < 0.25, rel


def test_no_detections_on_veins(synth, result):
    img, _, _ = synth
    h, w = img.shape
    # 叶脉条纹位于右下三角区，远离所有气孔
    in_vein = [d for d in result.detections if d.x > w * 0.75 and d.y > h * 0.8]
    assert len(in_vein) == 0


def test_tiled_response_equals_full(synth):
    img, _, _ = synth
    det_full = ClassicalStomataDetector(DetectorConfig(expected_major_px=35, bootstrap_seeds=15))
    r1 = det_full.detect(img)
    det_tiled = ClassicalStomataDetector(DetectorConfig(expected_major_px=35, tile_size=200, tile_overlap=96),
                                         template=r1.template)
    r2 = det_tiled.detect(img)
    # 分块拼接响应图与整图计算一致 ⇒ 计数一致（边界目标无重复）
    assert np.allclose(r1.response, r2.response, atol=1e-4)
    assert r1.count == r2.count


def test_margin_exclusion(synth):
    img, _, _ = synth
    crop = img[:, 100:420]            # 裁剪后左右边界会截断部分气孔
    on = ClassicalStomataDetector(DetectorConfig(expected_major_px=35, bootstrap_seeds=12)).detect(crop)
    off = ClassicalStomataDetector(DetectorConfig(expected_major_px=35, bootstrap_seeds=12, margin_exclusion=False),
                                   template=on.template).detect(crop)
    assert off.count >= on.count
    assert all(not d.edge for d in on.detections)


def test_counter_csv(tmp_path, synth, result):
    img, _, _ = synth
    import cv2
    p = tmp_path / "synth.png"
    cv2.imwrite(str(p), img)
    counter = StomataCounter(DetectorConfig(expected_major_px=35, bootstrap_seeds=15, um_per_px=0.5),
                             template=result.template)
    reps = counter.run(str(tmp_path), verbose=False)
    detail, summary = counter.export_csv(reps, str(tmp_path / "out.csv"))
    assert os.path.exists(detail) and os.path.exists(summary)
    lines = open(detail, encoding="utf-8-sig").read().strip().splitlines()
    assert lines[0].startswith("image,id,x,y,major_axis_px")
    assert len(lines) - 1 == reps[0].count
    assert reps[0].summary()["density_per_mm2"] != ""
