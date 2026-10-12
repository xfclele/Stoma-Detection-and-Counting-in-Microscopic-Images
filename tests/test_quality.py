import cv2
import numpy as np

from stomata.quality import CHECK, OK, RETAKE, assess, image_quality

from .synthetic import make_synthetic


def _image(blur=0.0, gain=1.0):
    img = make_synthetic(1280, 960, n=60, seed=1)[0].astype(np.float32) / 255
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = 0.5 + (img - 0.5) * gain
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def test_blur_lowers_sharpness_but_not_by_contrast():
    sharp, blurred, flat = image_quality(_image()), image_quality(_image(blur=2.5)), image_quality(_image(gain=0.4))
    assert blurred["sharpness"] < 0.7 * sharp["sharpness"]
    assert abs(flat["sharpness"] - sharp["sharpness"]) < 0.1 * sharp["sharpness"]   # 清晰度与对比度无关 / sharpness is contrast-independent
    assert flat["contrast"] < 0.5 * sharp["contrast"]


def test_half_blurred_image_shows_uneven_focus():
    img = _image()
    img[:, :640] = cv2.GaussianBlur(img, (0, 0), 3)[:, :640]
    assert image_quality(img)["focus_uniformity"] < 0.8 * image_quality(_image())["focus_uniformity"]


def _q(**kw):
    q = dict(sharpness=0.24, sharpness_low=0.22, focus_uniformity=0.92, contrast=44.0, clipped=0.10)
    q.update(kw)
    return q


def test_assess_verdicts():
    assert assess(_q()) == (OK, [])
    assert assess(_q(sharpness_low=0.12))[0] == RETAKE
    assert assess(_q(contrast=15.0))[0] == RETAKE
    assert assess(_q(contrast=30.0))[0] == CHECK
    assert assess(_q(focus_uniformity=0.6))[0] == CHECK
    assert assess(_q(n_stomata=0))[0] == RETAKE
    assert assess(_q(n_stomata=200, size_px=35.0))[0] == RETAKE        # 倍率太低 / magnification too low
    assert assess(_q(n_stomata=200, size_px=60.0))[0] == OK
    assert assess(_q(n_stomata=5, size_px=35.0))[0] == OK              # 检出太少不判尺寸 / too few detections to judge size
