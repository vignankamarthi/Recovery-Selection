"""Sensor-stream abstraction (Part 1).

One clean interface per modality so MockSource (tests), SimSource (physics sim), and
RosSource (real Kinova, on the lab Linux box) are interchangeable behind the same
protocol. Swapping SimSource for RosSource is the only change needed to go from
synthetic to real data. Imports schema only.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from schema.streams import Modality, Sample


@runtime_checkable
class SensorSource(Protocol):
    """A source of timestamped samples for one modality."""

    modality: Modality

    def start(self) -> None:
        """Begin producing samples."""
        ...

    def read(self) -> Sample:
        """Return the next sample (blocking or latest, backend-defined)."""
        ...

    def stream(self) -> Iterator[Sample]:
        """Yield samples until stopped."""
        ...

    def stop(self) -> None:
        """Stop and release resources."""
        ...
