import numpy as np

from stomata.evaluation import evaluate_against_annotation, match_points, precision_recall_f1


def test_prf_basic():
    p, r, f = precision_recall_f1(8, 2, 2)
    assert abs(p - 0.8) < 1e-9 and abs(r - 0.8) < 1e-9 and abs(f - 0.8) < 1e-9
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_one_to_one_matching_counts_duplicates_as_fp():
    gt = [(10, 10), (50, 50)]
    pred = [(11, 10), (12, 11), (49, 52), (200, 200)]   # 第二个点是第一个 GT 的重复检测
    m = match_points(pred, gt, tol=5)
    assert (m.tp, m.fp, m.fn) == (2, 2, 0)


def test_ignore_points_are_neutral():
    m = match_points([(10, 10), (100, 100)], [(10, 10)], tol=5, ignore=[(101, 99)])
    assert (m.tp, m.fp, m.fn) == (1, 0, 0)


def test_region_annotations():
    ann = {"regions": [{"box": [0, 0, 100, 100], "points": [[20, 20], [60, 60]], "ignore": []},
                       {"box": [200, 200, 300, 300], "points": [[250, 250]]}]}
    pred = np.array([[21, 19], [250, 252], [500, 500]])   # (500,500) 在所有区域之外，不计入
    m = evaluate_against_annotation(pred, ann, tol=5)
    assert (m.tp, m.fp, m.fn) == (2, 0, 1)
