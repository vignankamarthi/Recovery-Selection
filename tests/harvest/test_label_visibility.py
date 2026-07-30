"""Synthetic-image tests for the overhead label-visibility read.

These use hand-built arrays, no MuJoCo, so they run anywhere and pin the contract the hardware
SceneOracle relies on: a coverage count, a legibility flag at the same 70px bar the sim uses, and a
centroid/bbox. The sim-ground-truth cross-check lives in scripts/sim/, not here (it needs a render).
"""
import numpy as np
import pytest

from harvest.vision.label_visibility import (
    DEFAULT_MIN_COVERAGE_PX,
    LabelSpec,
    LabelVisibility,
    label_mask,
    label_visibility,
    rgb_to_hsv,
)


def _canvas(h=200, w=200, bg=(0.1, 0.1, 0.1)):
    img = np.empty((h, w, 3), dtype=np.float64)
    img[:] = np.asarray(bg)
    return img


def _paint(img, y0, y1, x0, x1, color):
    img[y0:y1, x0:x1] = np.asarray(color)
    return img


# --- rgb-mode (the default, matches the sim's near-white label) ---------------------------------

def test_white_label_patch_counted_and_visible():
    img = _paint(_canvas(), 90, 110, 80, 120, (0.95, 0.95, 0.98))   # 20x40 = 800 px
    r = label_visibility(img)                                        # default spec = sim white label
    assert r.coverage_px == 800
    assert r.visible is True
    assert r.coverage_frac == pytest.approx(800 / (200 * 200))


def test_below_threshold_patch_not_visible_but_counted():
    img = _paint(_canvas(), 0, 5, 0, 10, (0.95, 0.95, 0.98))        # 50 px < 70
    r = label_visibility(img)
    assert r.coverage_px == 50
    assert r.visible is False                                       # counted, but below the bar


def test_exactly_at_threshold_is_visible():
    img = _paint(_canvas(), 0, 7, 0, 10, (0.95, 0.95, 0.98))        # 70 px == bar
    r = label_visibility(img)
    assert r.coverage_px == DEFAULT_MIN_COVERAGE_PX
    assert r.visible is True                                        # >= is inclusive


def test_no_label_returns_empty_read():
    r = label_visibility(_canvas())                                # dark scene, no label
    assert r.coverage_px == 0
    assert r.visible is False
    assert r.centroid_xy is None and r.bbox is None


def test_centroid_and_bbox_are_correct():
    img = _paint(_canvas(), 50, 70, 100, 140, (0.95, 0.95, 0.98))   # rows 50..69, cols 100..139
    r = label_visibility(img)
    cx, cy = r.centroid_xy
    assert cx == pytest.approx((100 + 139) / 2)
    assert cy == pytest.approx((50 + 69) / 2)
    assert r.bbox == (100, 50, 140, 70)                            # x0,y0,x1_excl,y1_excl


def test_uint8_and_float_inputs_agree():
    f = _paint(_canvas(), 90, 110, 80, 120, (0.95, 0.95, 0.98))
    u = (f * 255).round().astype(np.uint8)
    assert label_visibility(f).coverage_px == label_visibility(u).coverage_px


def test_rgba_frame_ignores_alpha():
    rgb = _paint(_canvas(), 90, 110, 80, 120, (0.95, 0.95, 0.98))
    rgba = np.concatenate([rgb, np.ones((200, 200, 1))], axis=-1)
    assert label_visibility(rgba).coverage_px == 800


def test_tolerance_excludes_far_colors():
    img = _paint(_canvas(), 90, 110, 80, 120, (0.2, 0.6, 0.2))      # green patch, not near white
    assert label_visibility(img).coverage_px == 0


# --- roi windowing (the spatial gate that removes a same-coloured confound) ----------------------

def test_roi_excludes_a_same_colored_confound():
    img = _canvas()
    _paint(img, 10, 30, 10, 30, (0.95, 0.95, 0.98))                # confound (a white "arm"), 400 px
    _paint(img, 150, 160, 150, 160, (0.95, 0.95, 0.98))            # the label, 100 px, elsewhere
    whole = label_visibility(img)
    assert whole.coverage_px == 500                                # colour alone catches both
    windowed = label_visibility(img, roi=(140, 140, 175, 175))     # window on the label only
    assert windowed.coverage_px == 100
    assert windowed.visible is True


def test_roi_reports_coordinates_in_full_image_frame():
    img = _paint(_canvas(), 150, 160, 150, 160, (0.95, 0.95, 0.98))
    r = label_visibility(img, roi=(140, 140, 175, 175))
    assert r.bbox == (150, 150, 160, 160)                          # full-image coords, not roi-relative
    assert r.centroid_xy == pytest.approx((154.5, 154.5))


def test_roi_off_the_label_reads_empty():
    img = _paint(_canvas(), 150, 160, 150, 160, (0.95, 0.95, 0.98))
    r = label_visibility(img, roi=(0, 0, 40, 40))
    assert r.coverage_px == 0 and r.visible is False


# --- hsv-mode (for a saturated real label) ------------------------------------------------------

def test_hsv_mode_detects_a_red_label_with_hue_wrap():
    img = _paint(_canvas(), 90, 110, 80, 120, (0.85, 0.05, 0.05))   # strong red -> hue ~0
    spec = LabelSpec(mode="hsv", hue=(0.95, 0.05), sat=(0.5, 1.0), val=(0.3, 1.0))  # wrap-around band
    r = label_visibility(img, spec)
    assert r.coverage_px == 800 and r.visible is True


def test_hsv_mode_rejects_out_of_band_color():
    img = _paint(_canvas(), 90, 110, 80, 120, (0.05, 0.05, 0.85))   # blue, hue ~0.66
    spec = LabelSpec(mode="hsv", hue=(0.95, 0.05), sat=(0.5, 1.0), val=(0.3, 1.0))
    assert label_visibility(img, spec).coverage_px == 0


def test_rgb_to_hsv_matches_known_values():
    px = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])
    hsv = rgb_to_hsv(px)
    assert hsv[0, 0] == pytest.approx([0.0, 1.0, 1.0])              # red
    assert hsv[0, 1] == pytest.approx([1 / 3, 1.0, 1.0])           # green
    assert hsv[0, 2] == pytest.approx([2 / 3, 1.0, 1.0])           # blue
    assert hsv[0, 3, 1] == pytest.approx(0.0)                      # white -> zero saturation


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        label_mask(_canvas(), LabelSpec(mode="bogus"))


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        label_visibility(np.zeros((10, 10)))                       # not HxWx3
