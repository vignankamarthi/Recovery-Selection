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
