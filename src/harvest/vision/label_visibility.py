"""Overhead label-visibility read from an RGB frame (Part 1, hardware SceneOracle's label read).

Padir's criterion is RGB pixel coverage of the nutrition label in the fixed overhead view. In sim this
is read from ground-truth segmentation; here it is read from the image itself, so the same signal works
on the real camera. The label is recognized by a swappable `LabelSpec` (an RGB-distance match by default,
which handles a white or a coloured label, or an HSV band for a saturated colour). Swap the spec for the
real label's appearance, or a learned mask, behind the same call. Pure numpy, no MuJoCo / torch / ROS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Matches the sim's `LABEL_VISIBLE_PX` (Padir's legibility bar), kept here so the hardware read applies
# the same threshold the sim labelling used, at the sim's 200x200 overhead framing.
DEFAULT_MIN_COVERAGE_PX = 70


@dataclass(frozen=True)
class LabelSpec:
    """How to recognize the label in an overhead RGB frame. Two modes.

    - ``rgb`` (default), match pixels within ``rgb_tol`` (L2 distance in [0,1] RGB) of ``target_rgb``.
      Works for any label colour including WHITE (where hue is undefined), which is the sim label.
    - ``hsv``, match an HSV band (``hue`` / ``sat`` / ``val``, each a (lo, hi) in [0,1], ``hue`` is
      wrap-aware). More lighting-robust for a saturated colour.

    The default matches the sim's near-white label so the read validates against sim ground truth out of
    the box. On hardware, set the spec from the real label's measured appearance (or a coloured sticker)."""

    mode: str = "rgb"
    target_rgb: Sequence[float] = (0.95, 0.95, 0.98)
    rgb_tol: float = 0.14
    hue: tuple[float, float] = (0.0, 1.0)
    sat: tuple[float, float] = (0.0, 1.0)
    val: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class LabelVisibility:
    """The overhead label read. ``coverage_px`` is the raw label-pixel count (the direct analog of the
    sim's segmentation pixel count), ``visible`` clears the legibility bar, and the centroid / bbox are
    for downstream framing and debugging."""

    coverage_px: int
    coverage_frac: float
    visible: bool
    centroid_xy: Optional[tuple[float, float]]
    bbox: Optional[tuple[int, int, int, int]]


def _as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("expected an HxWx3 (or 4) RGB image")
    if arr.max(initial=0.0) > 1.0:            # uint8 or 0-255 float -> [0,1]
        arr = arr / 255.0
    return arr[..., :3]


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB->HSV, all channels in [0,1], hue wraps at 1.0. Numpy only."""
    arr = _as_float_rgb(rgb)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    h = np.zeros_like(mx)
    nz = df > 1e-12
    rm, gm, bm = nz & (mx == r), nz & (mx == g), nz & (mx == b)
    h[rm] = ((g[rm] - b[rm]) / df[rm]) % 6.0
    h[gm] = ((b[gm] - r[gm]) / df[gm]) + 2.0
    h[bm] = ((r[bm] - g[bm]) / df[bm]) + 4.0
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-12, df / np.where(mx > 1e-12, mx, 1.0), 0.0)
    return np.stack([h, s, mx], axis=-1)


def label_mask(rgb: np.ndarray, spec: LabelSpec) -> np.ndarray:
    """Boolean HxW mask of the label pixels per the spec."""
    if spec.mode == "rgb":
        arr = _as_float_rgb(rgb)
        t = np.asarray(spec.target_rgb, dtype=np.float64)[:3]
        dist = np.sqrt(((arr - t) ** 2).sum(axis=-1))
        return dist <= spec.rgb_tol
    if spec.mode == "hsv":
        hsv = rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = spec.hue
        hue_ok = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
        return hue_ok & (s >= spec.sat[0]) & (s <= spec.sat[1]) & (v >= spec.val[0]) & (v <= spec.val[1])
    raise ValueError(f"unknown LabelSpec.mode {spec.mode!r} (expected 'rgb' or 'hsv')")


def label_visibility(
    rgb: np.ndarray,
    spec: Optional[LabelSpec] = None,
    min_coverage_px: int = DEFAULT_MIN_COVERAGE_PX,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> LabelVisibility:
    """Read label-visibility from an overhead RGB frame. `rgb` is HxWx3 (or 4), uint8 or float. Returns
    the pixel coverage, the visible/legible flag (coverage >= `min_coverage_px`), and the label centroid
    and bounding box (pixel coordinates), or None for those two when no label pixels are found.

    `roi` (x0, y0, x1, y1, x1/y1 exclusive) windows the read to where the presented can is expected.
    A colour match alone cannot tell the white label from other same-coloured things in frame (a light
    robot arm), so gate it with an independent cue: the gripper location is always known from forward
    kinematics, so projecting it into the fixed overhead view gives the window to look in. Counts,
    centroid, and bbox are reported in full-image coordinates. Omit `roi` to read the whole frame."""
    spec = spec if spec is not None else LabelSpec()
    mask = label_mask(rgb, spec)
    if roi is not None:
        x0, y0, x1, y1 = roi
        gate = np.zeros(mask.shape, dtype=bool)
        gate[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        mask = mask & gate
    n = int(mask.sum())
    frac = float(n) / float(mask.size)
    if n == 0:
        return LabelVisibility(0, 0.0, False, None, None)
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return LabelVisibility(n, frac, bool(n >= min_coverage_px), centroid, bbox)
