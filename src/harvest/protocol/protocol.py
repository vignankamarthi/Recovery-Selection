"""Episode-protocol state machine (Phase 1.5).

Encodes the HARVEST episode structure as an explicit FSM:
place -> grasp -> reorient -> overhead-verify -> done. The verify step scores label
visibility from the OVERHEAD camera (vision, independent of the tactile stream, per F7)
and sets the episode outcome. Driven by any `SensorSource`s, so mock now, real later.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Callable

import numpy as np

from harvest.recorder.recorder import DEFAULT_SKEW_TOLERANCE_NS, record_episode
from harvest.sensors.base import SensorSource
from schema.episode import Episode, Label, LabelProvenance, Outcome, RecordedEpisode

# The label must occupy >= this fraction of the overhead frame to count as visible.
DEFAULT_VISIBILITY_THRESHOLD = 0.2


class Phase(str, Enum):
    PLACE = "place"
    GRASP = "grasp"
    REORIENT = "reorient"
    VERIFY = "verify"
    DONE = "done"


_ORDER = (Phase.PLACE, Phase.GRASP, Phase.REORIENT, Phase.VERIFY, Phase.DONE)


class EpisodeProtocol:
    """The episode FSM. `advance()` steps to the next phase and saturates at DONE."""

    def __init__(self) -> None:
        self.phase = Phase.PLACE

    def is_done(self) -> bool:
        return self.phase is Phase.DONE

    def advance(self) -> Phase:
        if not self.is_done():
            self.phase = _ORDER[_ORDER.index(self.phase) + 1]
        return self.phase


def label_visibility_score(recorded: RecordedEpisode, overhead_key: str = "rgb_overhead") -> float:
    """Placeholder label-visibility score in [0, 1] from the last overhead frame.

    A deterministic mock proxy (normalized mean brightness) standing in for real
    label detection. Replaced by a sim-ground-truth scorer in 1.7 and by real overhead
    perception in Phase 3. Vision-only by construction, never tactile.
    """
    samples = recorded.streams.get(overhead_key)
    if not samples:
        raise ValueError(f"no {overhead_key!r} stream to score label visibility")
    frame = np.asarray(samples[-1].data, dtype=float)
    peak = 255.0 if frame.max() > 1.0 else 1.0
    return float(frame.mean() / peak)


Scorer = Callable[[RecordedEpisode], float]


def run_episode(
    episode: Episode,
    sources: dict[str, SensorSource],
    *,
    scorer: Scorer = label_visibility_score,
    n_samples: int = 4,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    verify_timestamps: bool = True,
    skew_tolerance_ns: int = DEFAULT_SKEW_TOLERANCE_NS,
) -> RecordedEpisode:
    """Run one episode through the FSM and return the RecordedEpisode with its outcome.

    Records the multimodal streams during grasp/reorient, scores label visibility from
    the overhead camera at verify, and stamps the outcome plus two AUTO_VISION labels.
    """
    proto = EpisodeProtocol()
    recorded: RecordedEpisode | None = None
    outcome: Outcome | None = None
    verify_labels: list[Label] = []

    while not proto.is_done():
        if proto.phase is Phase.GRASP:
            recorded = record_episode(
                episode, sources, n_samples, skew_tolerance_ns, verify_timestamps
            )
        elif proto.phase is Phase.VERIFY:
            if recorded is None:  # GRASP always precedes VERIFY; guard the invariant explicitly
                raise RuntimeError("VERIFY reached before GRASP recorded the episode")
            score = float(scorer(recorded))
            success = score >= threshold
            outcome = Outcome.SUCCESS if success else Outcome.FAILURE
            verify_labels = [
                Label("label_visibility_score", score, LabelProvenance.AUTO_VISION),
                Label("label_visible", success, LabelProvenance.AUTO_VISION),
            ]
        proto.advance()

    if recorded is None:  # the FSM always passes through GRASP before DONE
        raise RuntimeError("episode protocol finished without recording (GRASP never ran)")
    stamped = replace(
        recorded.episode,
        outcome=outcome,
        labels=list(recorded.episode.labels) + verify_labels,
    )
    return RecordedEpisode(episode=stamped, streams=recorded.streams)
