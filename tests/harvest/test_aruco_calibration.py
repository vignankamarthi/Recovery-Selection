"""Synthetic-correspondence tests for the planar eye-to-hand calibration.

No cv2 and no hardware: a known homography is chosen, synthetic (world -> pixel) correspondences are
generated from it, `PlanarCalibration.fit` recovers it, and the pixel<->world maps are checked to
round-trip to tight tolerance. Covers identity-ish and a non-identity (rotation + translation + scale)
transform, plus the >=4-correspondence guard and the cv2-optional ArUco seam.
"""
import numpy as np
import pytest

from harvest.vision.aruco_calibration import (
    MIN_CORRESPONDENCES,
    PlanarCalibration,
    detect_aruco_markers,
)


def _project(H: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Map world points to pixels through a ground-truth world->pixel homography."""
    hom = np.concatenate([world, np.ones((world.shape[0], 1))], axis=1)
    out = hom @ H.T
    return out[:, :2] / out[:, 2:3]


def _grid_world(n: int = 5) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, n), np.linspace(-0.15, 0.15, n))
    return np.column_stack([xs.ravel(), ys.ravel()])


def _similarity_world_to_pixel(theta, scale, tx, ty) -> np.ndarray:
    """A world->pixel homography that is a rotation + uniform scale + translation (affine, last row 0,0,1)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [scale * c, -scale * s, tx],
        [scale * s,  scale * c, ty],
        [0.0,        0.0,       1.0],
    ])


# --- round-trip on a non-identity transform -------------------------------------------------------

def test_fit_recovers_similarity_and_round_trips():
    world = _grid_world(5)
    H_w2p = _similarity_world_to_pixel(theta=0.6, scale=1800.0, tx=640.0, ty=360.0)
    pixels = _project(H_w2p, world)

    calib = PlanarCalibration.fit(image_points=pixels, world_points=world)

    # pixel -> world recovers the true table coordinates
    np.testing.assert_allclose(calib.pixel_to_world(pixels), world, atol=1e-6)
    # world -> pixel recovers the true pixels
    np.testing.assert_allclose(calib.world_to_pixel(world), pixels, atol=1e-4)


def test_round_trip_is_inverse_consistent():
    world = _grid_world(4)
    H_w2p = _similarity_world_to_pixel(theta=-0.3, scale=1200.0, tx=500.0, ty=500.0)
    pixels = _project(H_w2p, world)
    calib = PlanarCalibration.fit(pixels, world)

    probe = np.array([123.4, 456.7])
    back = calib.world_to_pixel(calib.pixel_to_world(probe))
    np.testing.assert_allclose(back, probe, atol=1e-6)


# --- a genuine projective (non-affine) homography, not just a similarity ---------------------------

def test_fit_recovers_projective_homography():
    world = _grid_world(6)
    H_w2p = np.array([
        [1500.0, 30.0, 620.0],
        [-40.0, 1450.0, 350.0],
        [0.0008, -0.0011, 1.0],          # non-zero last row = true perspective
    ])
    pixels = _project(H_w2p, world)
    calib = PlanarCalibration.fit(pixels, world)
    np.testing.assert_allclose(calib.pixel_to_world(pixels), world, atol=1e-6)


# --- single-point and batch shapes ----------------------------------------------------------------

def test_single_point_returns_shape_2():
    world = _grid_world(4)
    H_w2p = _similarity_world_to_pixel(0.2, 1000.0, 300.0, 300.0)
    pixels = _project(H_w2p, world)
    calib = PlanarCalibration.fit(pixels, world)

    out = calib.pixel_to_world(pixels[0])
    assert out.shape == (2,)


def test_label_centroid_to_world_returns_tuple():
    world = _grid_world(4)
    H_w2p = _similarity_world_to_pixel(0.1, 900.0, 400.0, 250.0)
    pixels = _project(H_w2p, world)
    calib = PlanarCalibration.fit(pixels, world)

    x, y = calib.label_centroid_to_world(tuple(pixels[3]))
    assert isinstance(x, float) and isinstance(y, float)
    np.testing.assert_allclose([x, y], world[3], atol=1e-6)


# --- guards ---------------------------------------------------------------------------------------

def test_too_few_correspondences_raises():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])       # only 3
    assert pts.shape[0] < MIN_CORRESPONDENCES
    with pytest.raises(ValueError):
        PlanarCalibration.fit(pts, pts)


def test_mismatched_lengths_raise():
    a = np.zeros((5, 2))
    b = np.zeros((4, 2))
    with pytest.raises(ValueError):
        PlanarCalibration.fit(a, b)


# --- ArUco helper is optional (cv2 not installed here) --------------------------------------------

def test_detect_aruco_markers_without_cv2_raises_clear_error():
    try:
        import cv2  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="OpenCV"):
            detect_aruco_markers(np.zeros((8, 8, 3), dtype=np.uint8))
    else:
        pytest.skip("cv2 is installed; the no-cv2 error path is not exercisable here")
