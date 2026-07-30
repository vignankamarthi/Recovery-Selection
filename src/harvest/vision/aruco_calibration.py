"""Eye-to-hand calibration for the fixed overhead camera (Part 1, hardware registration).

The overhead camera looks straight down at a planar workspace (the table), so the pixel <-> table map
is a planar HOMOGRAPHY, no full camera intrinsics needed. `PlanarCalibration` fits that homography from
>=4 pixel/world correspondences by DLT in pure numpy, then maps overhead pixels to table (robot-frame)
coordinates and back. This is what turns a label centroid read from the overhead RGB frame into a table
position the arm can act on.

The correspondences come from ArUco markers at known table positions. Detecting those markers needs
OpenCV, so `detect_aruco_markers` lazy-imports `cv2.aruco` and is optional; the calibration MATH does
not depend on it (tests feed synthetic correspondences). No torch, no MuJoCo, no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CORRESPONDENCES = 4          # a homography has 8 DOF, so 4 point pairs (8 equations) is the minimum


def _as_points(pts: np.ndarray) -> np.ndarray:
    """Coerce input to an (N, 2) float array (accepts a single (2,) point too)."""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("expected points shaped (N, 2) or a single (2,) point")
    return arr


def _normalizing_transform(pts: np.ndarray) -> np.ndarray:
    """Hartley normalization: 3x3 similarity that centres `pts` and scales mean distance to sqrt(2).

    Conditioning the DLT this way is what keeps the homography accurate on real pixel coordinates
    (hundreds of pixels), where the raw normal equations are badly scaled."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map points through a 3x3 homography. Returns (N, 2), or (2,) for a single input point."""
    p = _as_points(pts)
    hom = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)      # (N, 3)
    out = hom @ H.T
    w = out[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("homography maps a point to infinity (degenerate correspondence)")
    xy = out[:, :2] / w
    return xy[0] if np.asarray(pts).ndim == 1 else xy


def _solve_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve the 3x3 homography mapping `src` -> `dst` by the normalized DLT (SVD null-space)."""
    Ts, Td = _normalizing_transform(src), _normalizing_transform(dst)
    s = _apply_homography(Ts, src)
    d = _apply_homography(Td, dst)
    rows = []
    for (xs, ys), (xd, yd) in zip(s, d):
        rows.append([-xs, -ys, -1.0, 0.0, 0.0, 0.0, xd * xs, xd * ys, xd])
        rows.append([0.0, 0.0, 0.0, -xs, -ys, -1.0, yd * xs, yd * ys, yd])
    A = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    Hn = vt[-1].reshape(3, 3)                                        # smallest singular vector
    H = np.linalg.inv(Td) @ Hn @ Ts                                 # denormalize
    return H / H[2, 2]


@dataclass(frozen=True)
class PlanarCalibration:
    """A fitted pixel <-> table homography for the fixed overhead camera. `H` maps overhead pixel
    coordinates (u, v) to table/robot-frame coordinates (x, y)."""

    H: np.ndarray                    # 3x3 pixel -> world homography

    @classmethod
    def fit(cls, image_points: np.ndarray, world_points: np.ndarray) -> "PlanarCalibration":
        """Fit the homography from >=4 correspondences. `image_points` are (N, 2) overhead pixels,
        `world_points` the matching (N, 2) table coordinates (same order)."""
        img = _as_points(image_points)
        wld = _as_points(world_points)
        if img.shape[0] != wld.shape[0]:
            raise ValueError("image_points and world_points must have the same length")
        if img.shape[0] < MIN_CORRESPONDENCES:
            raise ValueError(f"need at least {MIN_CORRESPONDENCES} correspondences, got {img.shape[0]}")
        return cls(H=_solve_homography_dlt(img, wld))

    def pixel_to_world(self, pt: np.ndarray) -> np.ndarray:
        """Map overhead pixel(s) to table coordinates. Accepts (2,) or (N, 2), returns the same shape."""
        return _apply_homography(self.H, pt)

    def world_to_pixel(self, pt: np.ndarray) -> np.ndarray:
        """Map table coordinate(s) back to overhead pixels. Accepts (2,) or (N, 2)."""
        return _apply_homography(np.linalg.inv(self.H), pt)

    def label_centroid_to_world(self, centroid_xy: tuple[float, float]) -> tuple[float, float]:
        """Convenience: map a label centroid (the pixel (x, y) from `label_visibility`) to table (x, y)."""
        x, y = self.pixel_to_world(np.asarray(centroid_xy, dtype=np.float64))
        return float(x), float(y)


def detect_aruco_markers(image: np.ndarray, dictionary_name: str = "DICT_4X4_50"):
    """Detect ArUco markers in an overhead frame (OPTIONAL helper, lazy `cv2.aruco`).

    Returns `(corners, ids)` exactly as OpenCV does: `corners` a list of (1, 4, 2) float arrays and
    `ids` an (M, 1) int array (or None when nothing is found). Raises a clear error if OpenCV is not
    installed. The calibration math above does not use this, so tests run without cv2."""
    try:
        import cv2                                        # lazy: calibration math must not need OpenCV
    except ImportError as e:
        raise RuntimeError(
            "detect_aruco_markers needs OpenCV (cv2) with the aruco module. Install opencv-contrib-python "
            "on the bench box. The PlanarCalibration math itself needs no cv2."
        ) from e
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    return corners, ids
