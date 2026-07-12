"""Shared (de)serialization of Episode metadata to/from a plain dict (Phase 1.3).

Used by both io adapters (rosbag2 and lerobot) so episode metadata is encoded one way.
Enums go to their `.value`; reconstruction restores them. stdlib only.
"""

from __future__ import annotations

from schema.episode import (
    ConditionClass,
    Episode,
    Label,
    LabelProvenance,
    Outcome,
)


def episode_to_dict(ep: Episode) -> dict:
    return {
        "episode_id": ep.episode_id,
        "can_id": ep.can_id,
        "condition": ep.condition.value,
        "outcome": ep.outcome.value if ep.outcome is not None else None,
        "stream_keys": list(ep.stream_keys),
        "labels": [
            {"name": l.name, "value": l.value, "provenance": l.provenance.value}
            for l in ep.labels
        ],
        "metadata": ep.metadata,
    }


def episode_from_dict(d: dict) -> Episode:
    return Episode(
        episode_id=d["episode_id"],
        can_id=d["can_id"],
        condition=ConditionClass(d["condition"]),
        outcome=Outcome(d["outcome"]) if d["outcome"] is not None else None,
        stream_keys=tuple(d["stream_keys"]),
        labels=[
            Label(x["name"], x["value"], LabelProvenance(x["provenance"]))
            for x in d["labels"]
        ],
        metadata=dict(d["metadata"]),
    )
