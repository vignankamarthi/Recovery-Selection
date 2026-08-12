"""Tests for the synchronized recorder (Phase 1.4). Synthetic data only."""

import pytest

from harvest.recorder.recorder import (
    TimestampConsistencyError,
    check_timestamp_consistency,
    record_episode,
    record_ticks,
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


# -- record_ticks: synchronized per-tick capture (one sample from every source per tick), the
# -- stop-controlled recorder for live hardware sessions (NOT the sequential-block record_episode).

def test_record_ticks_stops_on_predicate_and_is_tick_aligned():
    calls = {"n": 0}

    def should_stop() -> bool:
        # False, False, False, True -> exactly 3 ticks recorded
        stop = calls["n"] >= 3
        calls["n"] += 1
        return stop

    rec = record_ticks(
        _episode(), _aligned_sources(), should_stop,
        rate_hz=1000.0, clock=lambda: 0, sleep=lambda _dt: None,
    )
    assert set(rec.streams) == {"tactile", "force_torque"}
    # every stream advanced exactly one sample per tick -> equal, tick-aligned lengths
    assert all(len(v) == 3 for v in rec.streams.values())
    assert set(rec.episode.stream_keys) == {"tactile", "force_torque"}


def test_record_ticks_respects_max_ticks():
    rec = record_ticks(
        _episode(), _aligned_sources(), should_stop=lambda: False,
        max_ticks=5, rate_hz=1000.0, clock=lambda: 0, sleep=lambda _dt: None,
    )
    assert all(len(v) == 5 for v in rec.streams.values())


def test_record_ticks_reads_sources_round_robin_each_tick():
    # interleave order proves per-tick round-robin: tactile then force_torque, each tick.
    order: list[str] = []

    class Spy:
        def __init__(self, name, modality):
            self.name, self.modality = name, modality
            self._t = 0

        def start(self):
            pass

        def read(self):
            order.append(self.name)
            self._t += 1
            return Sample(self.modality, self._t, [[0.0]], self.name)

        def stream(self):
            while True:
                yield self.read()

        def stop(self):
            pass

    sources = {"tac": Spy("tac", Modality.TACTILE), "ft": Spy("ft", Modality.FORCE_TORQUE)}
    n = {"i": 0}

    def should_stop():
        stop = n["i"] >= 2
        n["i"] += 1
        return stop

    record_ticks(_episode(), sources, should_stop, rate_hz=1000.0, clock=lambda: 0, sleep=lambda _dt: None)
    assert order == ["tac", "ft", "tac", "ft"]  # two ticks, round-robin within each


def test_record_ticks_drops_a_failing_tick_atomically_when_tolerant():
    # A camera-like source that raises on its 2nd read (a transient USB hiccup). With tolerate_errors,
    # that whole tick is dropped, so NO stream gets a partial sample and all streams stay equal length.
    dropped = {"n": 0}

    class FlakySource:
        def __init__(self, modality, fail_on):
            self.modality, self._fail_on, self._t = modality, fail_on, 0

        def start(self):
            pass

        def read(self):
            self._t += 1
            if self._t == self._fail_on:
                raise RuntimeError("transient frame error")
            return Sample(self.modality, self._t * 1000, [[0.0]], "flaky")

        def stream(self):
            while True:
                yield self.read()

        def stop(self):
            pass

    good = MockSource(Modality.PROPRIOCEPTION, rate_hz=1000.0, start_ns=0)
    flaky = FlakySource(Modality.RGB_OVERHEAD, fail_on=2)
    n = {"i": 0}

    def should_stop():
        stop = n["i"] >= 3   # attempt 3 ticks; the 2nd read of flaky fails -> 2 committed
        n["i"] += 1
        return stop

    rec = record_ticks(
        _episode(), {"prop": good, "cam": flaky}, should_stop,
        rate_hz=1000.0, clock=lambda: 0, sleep=lambda _dt: None,
        tolerate_errors=True, on_drop=lambda _e: dropped.__setitem__("n", dropped["n"] + 1),
    )
    # 3 attempts, tick #2 dropped -> 2 committed samples in EVERY stream (atomic, aligned)
    assert len(rec.streams["prop"]) == len(rec.streams["cam"]) == 2
    assert dropped["n"] == 1


def test_record_ticks_reraises_by_default():
    class Boom:
        modality = Modality.RGB_OVERHEAD

        def start(self):
            pass

        def read(self):
            raise RuntimeError("boom")

        def stream(self):
            yield from ()

        def stop(self):
            pass

    with pytest.raises(RuntimeError):
        record_ticks(
            _episode(), {"cam": Boom()}, should_stop=lambda: False,
            max_ticks=3, rate_hz=1000.0, clock=lambda: 0, sleep=lambda _dt: None,
        )
