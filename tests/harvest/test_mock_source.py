"""Tests for the MockSource sensor backend (Phase 1.2). Synthetic data only."""

import itertools

import numpy as np

from harvest.sensors.base import SensorSource
from harvest.sensors.mock import MockSource
from schema.streams import Modality, Sample


def test_mock_source_satisfies_the_protocol():
    src = MockSource(Modality.TACTILE)
    assert isinstance(src, SensorSource)
    assert src.modality is Modality.TACTILE


def test_read_returns_sample_of_the_right_modality():
    src = MockSource(Modality.FORCE_TORQUE)
    src.start()
    s = src.read()
    assert isinstance(s, Sample)
    assert s.modality is Modality.FORCE_TORQUE


def test_timestamps_are_monotonic_and_rate_spaced():
    src = MockSource(Modality.PROPRIOCEPTION, rate_hz=100.0, start_ns=0)
    src.start()
    ts = [src.read().timestamp_ns for _ in range(5)]
    assert ts == sorted(ts)
    assert len(set(ts)) == 5
    # 100 Hz -> 10 ms = 10_000_000 ns between samples
    assert ts[1] - ts[0] == 10_000_000


def test_same_seed_is_deterministic():
    a = MockSource(Modality.TACTILE, seed=7)
    b = MockSource(Modality.TACTILE, seed=7)
    a.start()
    b.start()
    for _ in range(3):
        np.testing.assert_array_equal(a.read().data, b.read().data)


def test_different_seed_differs():
    a = MockSource(Modality.TACTILE, seed=1)
    b = MockSource(Modality.TACTILE, seed=2)
    a.start()
    b.start()
    assert not np.array_equal(a.read().data, b.read().data)


def test_per_modality_shapes():
    expected = {
        Modality.TACTILE: (4, 7),          # 28-taxel TSF-85 grid
        Modality.FORCE_TORQUE: (6,),       # Fx,Fy,Fz,Tx,Ty,Tz
        Modality.PROPRIOCEPTION: (7,),     # Gen3 has 7 joints
        Modality.RGB_OVERHEAD: (16, 16, 3),
        Modality.DEPTH_WRIST: (16, 16),
    }
    for modality, shape in expected.items():
        src = MockSource(modality)
        src.start()
        assert np.asarray(src.read().data).shape == shape


def test_stream_yields_samples():
    src = MockSource(Modality.RGB_WRIST)
    src.start()
    got = list(itertools.islice(src.stream(), 3))
    assert len(got) == 3
    assert all(isinstance(s, Sample) for s in got)
