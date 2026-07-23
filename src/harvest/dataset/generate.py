"""End-to-end synthetic dataset generation (Phase 1.9).

Runs the MuJoCo sim over a grid of distinct physical cans (condition x can-index x pose)
and returns the recorded episodes. A `can_id` names one physical can, so its geometry is
fixed across its poses (seeded by `can_id` in the scene), while the can position varies per
episode. The organic, condition-correlated grasp failures come from 1.7. Downstream, these
episodes are split by can (F6), exported, and carded by the `dataset` package.
"""

from __future__ import annotations

import random
from typing import Sequence

from harvest.sim.episode import DEFAULT_MODALITIES, record_episode
from schema.episode import ConditionClass, Episode, RecordedEpisode
from schema.streams import Modality

# Reachable can workspace (metres) around the nominal grasp pose used in 1.6/1.7.
_X_RANGE = (0.46, 0.54)
_Y_RANGE = (-0.06, 0.06)
# Cans spawn LYING (record_episode's default), so drop them from a small height above the table
# and let them settle on their side to an unknown resting yaw. This matches the proven pipeline.
_SPAWN_Z = 0.11


def sample_can_pose(rng: random.Random) -> tuple[float, float, float]:
    """A reachable table spawn pose for a lying can (jittered around the nominal grasp position)."""
    return (rng.uniform(*_X_RANGE), rng.uniform(*_Y_RANGE), _SPAWN_Z)


def generate_dataset(
    conditions: Sequence[ConditionClass] = tuple(ConditionClass),
    cans_per_condition: int = 4,
    poses_per_can: int = 2,
    modalities: Sequence[Modality] = DEFAULT_MODALITIES,
    seed: int = 0,
) -> list[RecordedEpisode]:
    """Generate the full synthetic dataset as a list of RecordedEpisodes.

    Each (condition, can-index) pair is one physical can; `poses_per_can` episodes vary its
    table pose. Geometry is fixed per can via its `can_id`. Deterministic under `seed`.
    """
    rng = random.Random(seed)
    recorded: list[RecordedEpisode] = []
    for c in conditions:
        for i in range(cans_per_condition):
            can_id = f"{c.value}-{i:03d}"  # one physical can, fixed geometry
            for p in range(poses_per_can):
                pos = sample_can_pose(rng)
                episode = Episode(f"{can_id}-p{p}", can_id, c)
                recorded.append(record_episode(episode, can_pos=pos, modalities=modalities))
    return recorded
