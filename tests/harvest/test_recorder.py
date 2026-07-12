"""Tests for the synchronized recorder (Phase 1.4). Synthetic data only."""

import pytest

from harvest.recorder.recorder import (
    TimestampConsistencyError,
    check_timestamp_consistency,
    record_episode,
)
from harvest.sensors.mock import MockSource
from schema.episode import ConditionClass, Episode, RecordedEpisode
from schema.streams import Modality, Sample


def _episode() -> Episode:
    return Episode(episode_id="e1", can_id="c1", condition=ConditionClass.NOMINAL)


def _aligned_sources():
    return {
        "tactile": MockSource(Modality.TACTILE, rate_hz=100.0, start_ns=0),
        "force_torque": MockSource(Modality.FORCE_TORQUE, rate_hz=100.0, start_ns=0),
    }


def test_record_produces_recorded_episode_with_streams():
    rec = record_episode(_episode(), _aligned_sources(), n_samples=4)
    assert isinstance(rec, RecordedEpisode)
    assert set(rec.streams) == {"tactile", "force_torque"}
    assert all(len(v) == 4 for v in rec.streams.values())
    # stream_keys stamped on the episode, original episode untouched
    assert set(rec.episode.stream_keys) == {"tactile", "force_torque"}


def test_aligned_streams_pass_consistency():
    rec = record_episode(_episode(), _aligned_sources(), n_samples=4)
    # no raise
    check_timestamp_consistency(rec.streams, skew_tolerance_ns=1_000_000)


def test_injected_clock_skew_is_detected():
    sources = {
        "tactile": MockSource(Modality.TACTILE, rate_hz=100.0, start_ns=0),
        # 1 second head start -> gross skew
        "force_torque": MockSource(Modality.FORCE_TORQUE, rate_hz=100.0, start_ns=1_000_000_000),
    }
    with pytest.raises(TimestampConsistencyError):
        record_episode(_episode(), sources, n_samples=4, skew_tolerance_ns=10_000_000)


def test_non_monotonic_stream_is_rejected():
    bad = {
        "tactile": [
            Sample(Modality.TACTILE, 2000, [[0.0]], "x"),
            Sample(Modality.TACTILE, 1000, [[0.0]], "x"),  # goes backwards
        ]
    }
    with pytest.raises(TimestampConsistencyError):
        check_timestamp_consistency(bad, skew_tolerance_ns=1_000_000)


def test_verify_false_skips_the_check():
    sources = {
        "a": MockSource(Modality.TACTILE, start_ns=0),
        "b": MockSource(Modality.FORCE_TORQUE, start_ns=9_999_999_999),
    }
    rec = record_episode(_episode(), sources, n_samples=2, verify=False)
    assert set(rec.streams) == {"a", "b"}
