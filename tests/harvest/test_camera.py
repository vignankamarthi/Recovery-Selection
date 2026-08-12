"""Synthetic-frame tests for the camera SensorSource.

No hardware and no cv2: frames come from an injected FakeGrabber, so the whole SensorSource path
(modality, latest-frame read, injected-clock timestamps, stream/stop lifecycle) is exercised end to
end. The real OpenCV / Azure Kinect paths are bench-only seams and are not touched here.
"""
from typing import List

import numpy as np
import pytest

from harvest.sensors.base import SensorSource
from harvest.sensors.camera import (
    AzureKinectGrabber,
    CameraSource,
    FrameGrabber,
    RealSenseGrabber,
    RealSenseSource,
)
from schema.streams import Modality, Sample


class FakeGrabber:
    """Serves queued frames to the source; records whether it was closed."""

    def __init__(self, frames: List[np.ndarray]):
        self._frames = list(frames)
        self.closed = False

    def read(self) -> np.ndarray:
        return self._frames.pop(0) if self._frames else self._last()

    def _last(self) -> np.ndarray:
        raise RuntimeError("no more frames queued")

    def close(self) -> None:
        self.closed = True


def _rgb_frame(fill: int) -> np.ndarray:
    return np.full((4, 5, 3), fill, dtype=np.uint8)


def _source(frames, **kw) -> CameraSource:
    src = CameraSource(FakeGrabber(frames), clock=lambda: 0, **kw)
    src.start()
    return src


# --- SensorSource conformance + modality ----------------------------------------------------------

def test_is_a_sensorsource():
    assert isinstance(CameraSource(FakeGrabber([])), SensorSource)


def test_fakegrabber_is_a_framegrabber():
    assert isinstance(FakeGrabber([]), FrameGrabber)


def test_default_modality_is_overhead_rgb():
    assert CameraSource(FakeGrabber([])).modality is Modality.RGB_OVERHEAD


def test_modality_is_configurable():
    src = CameraSource(FakeGrabber([]), modality=Modality.RGB_WRIST)
    assert src.modality is Modality.RGB_WRIST


# --- read returns the frame as a Sample -----------------------------------------------------------

def test_read_returns_frame_as_sample():
    frame = _rgb_frame(120)
    s = _source([frame]).read()
    assert isinstance(s, Sample)
    assert s.modality is Modality.RGB_OVERHEAD
    assert s.notes == "camera"
    np.testing.assert_array_equal(s.data, frame)


def test_read_passes_through_successive_frames():
    src = _source([_rgb_frame(1), _rgb_frame(2)])
    assert src.read().data[0, 0, 0] == 1
    assert src.read().data[0, 0, 0] == 2


def test_read_before_start_raises():
    # No grabber injected and start() not called, so there is nothing to read from.
    src = CameraSource(clock=lambda: 0)
    with pytest.raises(RuntimeError):
        src.read()


# --- injected clock -------------------------------------------------------------------------------

def test_injected_clock_sets_timestamp():
    ticks = iter([10, 20, 30])
    src = CameraSource(FakeGrabber([_rgb_frame(1), _rgb_frame(2)]), clock=lambda: next(ticks))
    src.start()
    assert src.read().timestamp_ns == 10
    assert src.read().timestamp_ns == 20


# --- stream / stop lifecycle ----------------------------------------------------------------------

def test_stream_stops_and_closes_grabber():
    grab = FakeGrabber([_rgb_frame(1), _rgb_frame(2), _rgb_frame(3)])
    src = CameraSource(grab, clock=lambda: 0)
    src.start()
    it = src.stream()
    next(it)
    src.stop()
    assert list(it) == []
    assert grab.closed is True


# --- Azure Kinect seam is a clear NotImplemented --------------------------------------------------

def test_azure_kinect_grabber_is_a_seam():
    with pytest.raises(NotImplementedError):
        AzureKinectGrabber()


# --- RealSense D435i (the confirmed overhead camera) ----------------------------------------------

class _FakeRSSource:
    """Stands in for a RealSenseSource: serves fixed color + depth arrays, counts poll/start/stop."""

    def __init__(self, color: np.ndarray, depth: np.ndarray):
        self._c, self._d = color, depth
        self.started = 0
        self.polls = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def poll(self) -> None:
        self.polls += 1

    def color(self) -> np.ndarray:
        return self._c

    def depth(self) -> np.ndarray:
        return self._d

    def stop(self) -> None:
        self.stopped += 1


def test_realsense_grabber_color_returns_rgb():
    fake = _FakeRSSource(_rgb_frame(90), np.zeros((4, 5), np.uint16))
    g = RealSenseGrabber("color", source=fake)
    np.testing.assert_array_equal(g.read(), _rgb_frame(90))
    assert fake.started == 1 and fake.polls == 1        # read() starts (idempotent) + polls


def test_realsense_grabber_depth_returns_depth():
    depth = np.full((4, 5), 1234, np.uint16)
    g = RealSenseGrabber("depth", source=_FakeRSSource(_rgb_frame(0), depth))
    np.testing.assert_array_equal(g.read(), depth)


def test_realsense_one_device_backs_both_streams():
    fake = _FakeRSSource(_rgb_frame(7), np.full((4, 5), 55, np.uint16))
    color_g, depth_g = RealSenseGrabber("color", source=fake), RealSenseGrabber("depth", source=fake)
    assert color_g.read()[0, 0, 0] == 7 and depth_g.read()[0, 0] == 55   # shared source, both work


def test_realsense_grabber_bad_stream_raises():
    with pytest.raises(ValueError):
        RealSenseGrabber("infrared", source=_FakeRSSource(_rgb_frame(0), np.zeros((1, 1), np.uint16)))


def test_realsense_grabber_is_a_framegrabber():
    assert isinstance(RealSenseGrabber("color", source=_FakeRSSource(_rgb_frame(0), np.zeros((1, 1), np.uint16))), FrameGrabber)


def test_camera_source_over_realsense_depth():
    depth = np.full((4, 5), 900, np.uint16)
    src = CameraSource(RealSenseGrabber("depth", source=_FakeRSSource(_rgb_frame(0), depth)),
                       modality=Modality.DEPTH_OVERHEAD, clock=lambda: 0)
    src.start()
    s = src.read()
    assert isinstance(src, SensorSource) and s.modality is Modality.DEPTH_OVERHEAD
    np.testing.assert_array_equal(s.data, depth)


# --- RealSenseSource: the one bit of device-API logic (BGR->RGB) tested via a fake pyrealsense2 ----

class _FakeAligned:
    def __init__(self, c, d):
        self._c, self._d = c, d

    def get_color_frame(self):
        return type("F", (), {"get_data": lambda s: self._c})()

    def get_depth_frame(self):
        return type("F", (), {"get_data": lambda s: self._d})()


class _FakeRS:
    """Minimal pyrealsense2 stand-in exercising RealSenseSource.start()/poll()."""

    class stream:
        color = 1
        depth = 2

    class format:
        bgr8 = 1
        z16 = 2

    def __init__(self, color_bgr, depth):
        self._c, self._d = color_bgr, depth

    def config(self):
        return type("Cfg", (), {"enable_stream": lambda *a, **k: None})()

    def pipeline(self):
        rs = self
        return type("Pipe", (), {"start": lambda s, c: None,
                                 "wait_for_frames": lambda s: _FakeAligned(rs._c, rs._d),
                                 "stop": lambda s: None})()

    def align(self, to):
        return type("Align", (), {"process": lambda s, f: f})()


def test_realsense_source_polls_and_converts_bgr_to_rgb():
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 2] = 200                                   # RED in BGR is channel 2
    depth = np.full((4, 4), 4321, np.uint16)
    src = RealSenseSource(_rs=_FakeRS(bgr, depth))
    src.start()
    src.poll()
    rgb = src.color()
    assert rgb[..., 0].mean() == 200 and rgb[..., 2].mean() == 0   # red moved to channel 0 (RGB)
    assert src.depth()[0, 0] == 4321


def test_realsense_source_start_is_idempotent():
    src = RealSenseSource(_rs=_FakeRS(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint16)))
    src.start()
    src.start()                                          # second start is a no-op, must not raise
    src.stop()
    src.stop()                                           # second stop is a no-op too


def test_realsense_grabber_poll_on_read_false_does_not_poll():
    # A follower grabber (poll_on_read=False) serves the cached frameset a sibling already polled,
    # so ONE device poll per tick backs BOTH overhead streams (half the USB reads, same frameset).
    fake = _FakeRSSource(_rgb_frame(3), np.full((4, 5), 9, np.uint16))
    leader = RealSenseGrabber("color", source=fake)                 # polls
    follower = RealSenseGrabber("depth", source=fake, poll_on_read=False)  # reads cache, no poll
    leader.read()
    follower.read()
    assert fake.polls == 1                                          # only the leader polled
    np.testing.assert_array_equal(follower.read(), np.full((4, 5), 9, np.uint16))
    assert fake.polls == 1                                          # follower still never polls


def test_realsense_source_align_false_skips_align_block():
    # The align block is the step that fails under USB-2/CPU contention. align=False must never build
    # or call it; poll() then serves raw (unaligned) color + depth, which is correct for raw capture.
    class _BoomAlignRS(_FakeRS):
        def align(self, to):
            raise AssertionError("align must not be used when align=False")

    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 2] = 150
    src = RealSenseSource(align=False, _rs=_BoomAlignRS(bgr, np.full((2, 2), 7, np.uint16)))
    src.start()
    src.poll()                                          # would raise if align were used
    assert src.color()[..., 0].mean() == 150            # still BGR->RGB converted
    assert src.depth()[0, 0] == 7
