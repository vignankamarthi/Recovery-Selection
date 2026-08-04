"""Overhead / wrist RGB(-D) cameras as HARVEST SensorSources (hardware, Part 1).

The overhead camera is the fixed top-down view Padir's label-visibility criterion reads from; the
wrist camera is the Gen3's RGB-D module. This module wraps a frame source behind the `SensorSource`
seam so the recorder captures camera streams exactly like `MockSource` / `SimSource` / `TSF85Source`.
It emits an `RGB_OVERHEAD` Sample by default (configurable), whose `data` is an HxWx3 uint8 RGB frame
(or HxW for depth).

numpy lives here, never in schema. The frame source is injected as a `FrameGrabber`, so the whole
SensorSource path is testable with synthetic frames and no hardware, no cv2, and no pyrealsense2. The
confirmed overhead camera (2026-08-04) is an Intel RealSense D435i, so `RealSenseGrabber` (lazy
`pyrealsense2`, RGB + aligned depth off one device) is the real path. `OpenCVGrabber` still opens the
D435i's color stream as a plain USB webcam for a quick RGB-only bring-up. The `AzureKinectGrabber` seam
is kept but superseded (the lab camera is the D435i, not the Kinect).
"""
from __future__ import annotations

import time
from typing import Callable, Iterator, Optional, Protocol, runtime_checkable

import numpy as np

from schema.streams import Modality, Sample

DEFAULT_CAMERA_INDEX = 0


@runtime_checkable
class FrameGrabber(Protocol):
    """The minimal frame source CameraSource needs. `OpenCVGrabber` is the real one; tests inject a fake.

    `read()` returns one frame as a numpy array: HxWx3 uint8 RGB (colour), or HxW for depth. `close()`
    releases the underlying device."""

    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


class OpenCVGrabber:
    """Real USB-webcam grabber (lazy `cv2`, so importing this module needs no OpenCV dep).

    Wraps `cv2.VideoCapture(index)` and returns RGB frames (OpenCV delivers BGR, so channels are
    reversed on the way out). This is the overhead prototype path before the Azure Kinect DK arrives.
    Validate against the live camera at the bench before trusting frame timing / colour."""

    def __init__(self, index: int = DEFAULT_CAMERA_INDEX):
        import cv2                                        # lazy: only when actually opening hardware
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open camera index {index} (cv2.VideoCapture). "
                "Check the USB webcam is connected and not held by another process."
            )

    def read(self) -> np.ndarray:
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError("camera frame read failed (cv2.VideoCapture.read returned no frame)")
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._cap.release()


class AzureKinectGrabber:
    """Placeholder for the Azure Kinect DK depth-camera path (the top-down RGB-D per the RIVeR setup).

    Not implemented on purpose. The Kinect depth engine needs the Azure Kinect SDK + pyk4a and a GPU on
    a physical terminal (it will not run over SSH or in a VM), so it is wired at the bench, not here. The
    seam is left so the real grabber drops in behind `FrameGrabber` with no change to `CameraSource`.
    Use `modality=Modality.DEPTH_OVERHEAD` and a real pyk4a-backed grabber once Austin's setup is confirmed."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AzureKinectGrabber is a bench-only seam: install pyk4a / the Azure Kinect SDK and wire "
            "the depth engine on the physical Linux box (needs a GPU, not SSH). Use OpenCVGrabber for "
            "the USB-webcam prototype until then."
        )

    def read(self) -> np.ndarray:                          # pragma: no cover - seam, never reached
        raise NotImplementedError

    def close(self) -> None:                               # pragma: no cover - seam, never reached
        raise NotImplementedError


class RealSenseSource:
    """One Intel RealSense (D435i) device pipeline, color + depth with depth ALIGNED to the color frame.

    Lazy `pyrealsense2`, so importing this module needs no RealSense dep. Both overhead streams
    (`rgb_overhead`, `depth_overhead`) come off this one device, so a single source backs two grabbers.
    `poll()` grabs one aligned frameset and caches color (converted BGR->RGB) + depth (uint16, mm). The
    BGR->RGB conversion is load-bearing, the label read's `CAMPBELL_RED_SPEC` targets red, so a missed
    swap would match blue instead. `start()` / `stop()` are idempotent so two grabbers can share one
    source. Frames are untested until bench-validated against the live D435i (timing + color/depth align)."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, _rs=None):
        self._width, self._height, self._fps = width, height, fps
        self._rs = _rs                         # inject a fake for tests; else lazy pyrealsense2
        self._pipeline = None
        self._align = None
        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None

    def start(self) -> None:
        if self._pipeline is not None:         # idempotent (shared by color + depth grabbers)
            return
        if self._rs is None:
            import pyrealsense2 as rs           # lazy: only when opening real hardware
            self._rs = rs
        rs = self._rs
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        cfg.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        self._pipeline = rs.pipeline()
        self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)

    def poll(self) -> None:
        """Grab one aligned frameset, cache color as RGB and depth as uint16."""
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color_bgr = np.asanyarray(aligned.get_color_frame().get_data())
        self._color = np.ascontiguousarray(color_bgr[..., ::-1])    # BGR -> RGB
        self._depth = np.asanyarray(aligned.get_depth_frame().get_data())

    def color(self) -> np.ndarray:
        if self._color is None:
            self.poll()
        return self._color

    def depth(self) -> np.ndarray:
        if self._depth is None:
            self.poll()
        return self._depth

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


class RealSenseGrabber:
    """A `FrameGrabber` view over a `RealSenseSource` for one stream, 'color' (RGB) or 'depth' (uint16 mm).

    The confirmed overhead camera (2026-08-04) is an Intel RealSense D435i, so this replaces the
    Azure-Kinect seam. `read()` polls the device for a fresh frame and returns the selected stream. A
    single D435i backs BOTH overhead streams, pass the same `source` to an `rgb_overhead` grabber and a
    `depth_overhead` grabber (each polls, so their framesets sit ~1/fps apart, fine for per-tick
    recording); pass no source for the common single-stream case and it opens its own."""

    def __init__(self, stream: str = "color", *, width: int = 640, height: int = 480, fps: int = 30,
                 source: Optional[RealSenseSource] = None):
        if stream not in ("color", "depth"):
            raise ValueError(f"stream must be 'color' or 'depth', got {stream!r}")
        self._stream = stream
        self._source = source if source is not None else RealSenseSource(width, height, fps)

    def read(self) -> np.ndarray:
        self._source.start()                   # idempotent
        self._source.poll()
        return self._source.color() if self._stream == "color" else self._source.depth()

    def close(self) -> None:
        self._source.stop()                    # idempotent (safe when the source is shared)


class CameraSource:
    """A `SensorSource` over an RGB(-D) camera. Reads frames through an injected `FrameGrabber` and emits
    the latest frame per `read()`, matching the recorder's latest-sample model. Default modality is the
    fixed overhead RGB view; set `modality` for the wrist camera or a depth stream."""

    def __init__(
        self,
        grabber: Optional[FrameGrabber] = None,
        *,
        modality: Modality = Modality.RGB_OVERHEAD,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        notes: str = "camera",
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.modality = modality
        self._grabber = grabber
        self._camera_index = camera_index
        self._notes = notes
        self._clock = clock
        self._stopped = False

    def start(self) -> None:
        """Open the grabber (if not injected) so frames are ready to read."""
        if self._grabber is None:
            self._grabber = OpenCVGrabber(self._camera_index)
        self._stopped = False

    def read(self) -> Sample:
        """Return the latest camera frame as a Sample of this source's modality."""
        if self._grabber is None:
            raise RuntimeError("CameraSource.start() not called")
        frame = self._grabber.read()
        return Sample(modality=self.modality, timestamp_ns=int(self._clock()), data=frame, notes=self._notes)

    def stream(self) -> Iterator[Sample]:
        while not self._stopped:
            yield self.read()

    def stop(self) -> None:
        self._stopped = True
        if self._grabber is not None:
            try:
                self._grabber.close()
            except Exception:
                pass
