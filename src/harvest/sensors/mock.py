"""MockSource: a deterministic synthetic sensor backend (Phase 1.2).

Produces plausible-shaped, seed-deterministic samples for any modality with no robot
and no simulator. It satisfies the `SensorSource` protocol, so the recorder, io, and
protocol layers can be built and tested end to end before the physics sim (1.6) or the
real driver (Phase 3) exist. numpy lives here, never in schema.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from schema.streams import Modality, Sample

# Per-modality synthetic array shapes.
_SHAPES: dict[Modality, tuple[int, ...]] = {
    Modality.TACTILE: (4, 7),          # 28-taxel Robotiq TSF-85 grid
    Modality.FORCE_TORQUE: (6,),       # Fx, Fy, Fz, Tx, Ty, Tz
    Modality.PROPRIOCEPTION: (7,),     # Kinova Gen3 has 7 joints
    Modality.RGB_OVERHEAD: (16, 16, 3),
    Modality.RGB_WRIST: (16, 16, 3),
    Modality.DEPTH_OVERHEAD: (16, 16),
    Modality.DEPTH_WRIST: (16, 16),
}


class MockSource:
    """A `SensorSource` that emits deterministic synthetic samples at a fixed rate."""

    def __init__(
        self,
        modality: Modality,
        rate_hz: float = 30.0,
        seed: int = 0,
        start_ns: int = 0,
    ) -> None:
        self.modality = modality
        self._rate_hz = rate_hz
        self._seed = seed
        self._start_ns = start_ns
        self._dt_ns = int(round(1e9 / rate_hz))
        self._rng = np.random.default_rng(seed)
        self._i = 0
        self._stopped = False

    def start(self) -> None:
        """Reset the sequence so runs are reproducible from a clean state."""
        self._rng = np.random.default_rng(self._seed)
        self._i = 0
        self._stopped = False

    def _make_data(self) -> np.ndarray:
        shape = _SHAPES[self.modality]
        if self.modality in (Modality.RGB_OVERHEAD, Modality.RGB_WRIST):
            return self._rng.integers(0, 256, size=shape, dtype=np.uint8)
        return self._rng.random(size=shape)

    def read(self) -> Sample:
        """Return the next synthetic sample and advance the clock and rng."""
        ts = self._start_ns + self._i * self._dt_ns
        data = self._make_data()
        self._i += 1
        return Sample(modality=self.modality, timestamp_ns=ts, data=data, notes="mock")

    def stream(self) -> Iterator[Sample]:
        """Yield samples until `stop()` is called."""
        while not self._stopped:
            yield self.read()

    def stop(self) -> None:
        self._stopped = True
