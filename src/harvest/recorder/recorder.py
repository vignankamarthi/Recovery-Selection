"""Synchronized multi-stream recorder (Phase 1.4).

Pulls samples from several `SensorSource`s, checks their timestamps are consistent
(the failure mode the HARVEST draft calls out), and emits a `RecordedEpisode`. Works
on any `SensorSource`, so mock now and the real driver later, unchanged.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Optional

from harvest.sensors.base import SensorSource
from schema.episode import Episode, RecordedEpisode
from schema.streams import Sample

# Default alignment tolerance: streams should start within 10 ms of each other.
DEFAULT_SKEW_TOLERANCE_NS = 10_000_000


class TimestampConsistencyError(ValueError):
    """Raised when recorded streams are not time-consistent (skew or non-monotonic)."""


def check_timestamp_consistency(
    streams: dict[str, list[Sample]],
    skew_tolerance_ns: int = DEFAULT_SKEW_TOLERANCE_NS,
) -> None:
    """Verify per-stream monotonicity and cross-stream start alignment. Raises on failure."""
    starts: list[int] = []
    for key, samples in streams.items():
        ts = [s.timestamp_ns for s in samples]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise TimestampConsistencyError(
                f"stream {key!r} timestamps are not strictly increasing: {ts}"
            )
        if ts:
            starts.append(ts[0])
    if starts:
        skew = max(starts) - min(starts)
        if skew > skew_tolerance_ns:
            raise TimestampConsistencyError(
                f"stream start skew {skew} ns exceeds tolerance {skew_tolerance_ns} ns"
            )


def record_episode(
    episode: Episode,
    sources: dict[str, SensorSource],
    n_samples: int,
    skew_tolerance_ns: int = DEFAULT_SKEW_TOLERANCE_NS,
    verify: bool = True,
) -> RecordedEpisode:
    """Record `n_samples` from each source into a RecordedEpisode.

    The passed `episode` is not mutated; a copy is returned with `stream_keys` stamped.
    """
    streams: dict[str, list[Sample]] = {}
    for key, source in sources.items():
        source.start()
        streams[key] = [source.read() for _ in range(n_samples)]

    if verify:
        check_timestamp_consistency(streams, skew_tolerance_ns)

    stamped = replace(episode, stream_keys=tuple(sources.keys()))
    return RecordedEpisode(episode=stamped, streams=streams)


def record_ticks(
    episode: Episode,
    sources: dict[str, SensorSource],
    should_stop: Callable[[], bool],
    *,
    max_ticks: Optional[int] = None,
    rate_hz: float = 20.0,
    tolerate_errors: bool = False,
    on_drop: Optional[Callable[[Exception], None]] = None,
    clock: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> RecordedEpisode:
    """Synchronized, stop-controlled capture for LIVE hardware sessions.

    Unlike `record_episode` (which reads all N samples of one source, then the next, so real-time
    streams end up a full second apart), this reads ONE sample from every source per tick, round-robin.
    All streams therefore advance together and stay tick-aligned (to within one tick's read latency).
    It runs until `should_stop()` returns True (the on/off button) or `max_ticks` ticks elapse, pacing
    each tick to `rate_hz`. `clock`/`sleep` are injectable so the loop is testable with no real time.

    Each tick is ATOMIC: every source is read into a batch first, and only a complete batch is committed.
    If a read raises and `tolerate_errors` is True, the WHOLE tick is dropped (no stream gets a partial
    sample, so streams never misalign), `on_drop(exc)` is called, and capture continues. This keeps a
    long teleop session alive through a transient USB frame hiccup instead of losing all of it. With
    `tolerate_errors` False (the default) the error propagates.

    Every source is started once up front (so the camera pipeline and the ROS bridge are warm before
    the first tick). Stopping is the caller's job (this does not call `source.stop()`), so an interactive
    wrapper can report and clean up after the loop returns.
    """
    for source in sources.values():
        source.start()

    streams: dict[str, list[Sample]] = {key: [] for key in sources}
    period_ns = int(1e9 / rate_hz) if rate_hz > 0 else 0
    next_tick_ns = clock()
    tick = 0
    while not should_stop() and (max_ticks is None or tick < max_ticks):
        try:
            batch = {key: source.read() for key, source in sources.items()}
        except Exception as exc:  # transient sensor hiccup -> drop the whole tick, keep the session
            if not tolerate_errors:
                raise
            if on_drop is not None:
                on_drop(exc)
        else:
            for key, sample in batch.items():
                streams[key].append(sample)
            tick += 1
        next_tick_ns += period_ns
        dt_s = (next_tick_ns - clock()) / 1e9
        if dt_s > 0:
            sleep(dt_s)

    stamped = replace(episode, stream_keys=tuple(sources.keys()))
    return RecordedEpisode(episode=stamped, streams=streams)
