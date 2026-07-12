"""The contract: a format-agnostic episode/dataset representation.

This module is the ONLY surface shared by the two tracks (harvest produces these,
recovery consumes them). Keep it dependency-light, stdlib only. No rosbag2, no sim,
no torch here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from schema.streams import Sample


class ConditionClass(str, Enum):
    """The five can-condition classes (grounded in the corrected HARVEST taxonomy)."""

    NOMINAL = "nominal"
    BODY_DENT = "body_dent"
    SEAM_DENT = "seam_dent"
    BULGE = "bulge"
    RUST = "rust"


class Outcome(str, Enum):
    """Episode outcome. FAILURE episodes are the substrate the recovery layer consumes."""

    SUCCESS = "success"
    FAILURE = "failure"


class LabelProvenance(str, Enum):
    """How a label was produced. Tactile-derived labels are flagged so the eventual
    tactile ablation cannot silently confound (HARVEST flag F7)."""

    MANUAL = "manual"
    AUTO_VISION = "auto_vision"
    AUTO_TACTILE = "auto_tactile"  # confound-flagged
    SIMULATOR = "simulator"


@dataclass(frozen=True)
class Label:
    """A single annotation with explicit provenance."""

    name: str
    value: object
    provenance: LabelProvenance


@dataclass
class Episode:
    """One time-aligned multimodal manipulation episode.

    `can_id` is mandatory and load-bearing: it is the unit of independence for
    splits and effective-sample-size accounting (HARVEST flags F5, F6). Streams are
    referenced by modality key; the actual samples live behind the I/O layer.
    """

    episode_id: str
    can_id: str
    condition: ConditionClass
    outcome: Outcome | None = None
    stream_keys: tuple[str, ...] = ()
    labels: list[Label] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def is_tactile_label_confounded(self) -> bool:
        """True if any label was derived from the tactile stream (F7 guard).

        A tactile-derived label must never be used as the evaluation target of a
        tactile ablation, that would be circular (target leakage). This flags it.
        """
        return any(
            label.provenance is LabelProvenance.AUTO_TACTILE for label in self.labels
        )


@dataclass
class RecordedEpisode:
    """A full episode: its metadata plus the actual per-stream samples.

    The `Episode` above is metadata only. A `RecordedEpisode` is what the recorder
    emits and the io layer persists, the metadata bundled with the time-ordered
    `Sample`s for each stream key (e.g. the modality value). Still stdlib-only: the
    sample `data` is opaque here, so no numpy/rosbag2 leaks into the schema.
    """

    episode: Episode
    streams: dict[str, list[Sample]] = field(default_factory=dict)
