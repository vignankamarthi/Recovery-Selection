"""Synchronized multi-stream recorder (Phase 1.4).

Pulls samples from several `SensorSource`s, checks their timestamps are consistent
(the failure mode the HARVEST draft calls out), and emits a `RecordedEpisode`. Works
on any `SensorSource`, so mock now and the real driver later, unchanged.
"""

from __future__ import annotations

from dataclasses import replace

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
