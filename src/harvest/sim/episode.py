"""Sim episode runner (Part 1).

Runs one lying-can pick-and-reorient demonstration (`sim.reorient.demonstrate`) on a `SimWorld`
backend while recording the multimodal streams into a `RecordedEpisode`. The demonstration is a
3-stage graded pipeline with three labels: `upright_success` (the reorient brought the label up
to face the overhead camera, right-side-up, in one move with no stand-upright detour) and
`label_visible` (the overhead camera reads enough label coverage) are REAL sim signals;
`grasp_stable` is a SIMULATOR default (a real tactile/physics signal on hardware, flagged by the
dataset card).
Also exposes `SimSource`, a `SensorSource` reading one modality from a shared, externally-stepped
world, for interface parity with the mock and (later) the real driver.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterator, Sequence

from harvest.labels import GRASP_STABLE, LABEL_UP_COS, LABEL_VISIBLE, UPRIGHT_SUCCESS
from harvest.sim.reorient import demonstrate, slip_severity
from harvest.sim.scene import can_seed_from_id
from harvest.sim.world import SimWorld
from schema.episode import Episode, Label, LabelProvenance, Outcome, RecordedEpisode
from schema.streams import Modality, Sample

# The canonical seven-stream suite (the F4 review split depth into two justified viewpoints). Depth
# is rendered only when streams are materialized (on the cluster); the local metadata run records
# just proprioception, so this cost is not paid on the Mac.
DEFAULT_MODALITIES: tuple[Modality, ...] = (
    Modality.PROPRIOCEPTION,
    Modality.FORCE_TORQUE,
    Modality.TACTILE,
    Modality.RGB_OVERHEAD,
    Modality.RGB_WRIST,
    Modality.DEPTH_OVERHEAD,
    Modality.DEPTH_WRIST,
)

# All cans spawn LYING (the task is to right them). This 90-degree tip lays the cylinder on its
# side; it settles under gravity to an unknown resting yaw, so the grasp is orientation-aware.
LYING_QUAT = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)


def record_sim_demo(
    episode: Episode,
    can_pos: tuple[float, float, float] = (0.5, 0.0, 0.11),
    modalities: Sequence[Modality] = DEFAULT_MODALITIES,
    record_every: int = 15,
    can_quat: tuple[float, float, float, float] = LYING_QUAT,
) -> RecordedEpisode:
    """Run one lying-can pick-and-reorient episode and return a RecordedEpisode.

    The can's physical variant is `episode.condition`, its per-can geometry is seeded from
    `episode.can_id` (so every can_id is a fixed distinct physical unit, F5/F6), and `can_quat`
    is its placement orientation (default LYING; the can settles under gravity to an unknown
    resting yaw). Streams are recorded through the smooth demonstration glide only, never through
    the hidden reachability search.
    """
    seed = can_seed_from_id(episode.can_id)
    world = SimWorld(
        can_pos,
        condition=episode.condition,
        can_seed=seed,
        can_quat=can_quat,
    )
    streams: dict[str, list[Sample]] = {m.value: [] for m in modalities}
    n = [0]

    def tick() -> None:
        n[0] += 1
        if n[0] % record_every == 0:
            ts = world.time_ns
            for m in modalities:
                streams[m.value].append(world.sample(m, ts))

    # The reorient is robust; failures come from a condition-correlated in-hand SLIP (damaged cans
    # slip, the label rolls off, the read fails), which is where tactile helps on real hardware.
    res = demonstrate(world, on_step=tick, slip=slip_severity(episode.condition, seed))

    # Three graded stage labels. `upright_success` and `label_visible` are real sim signals;
    # `grasp_stable` is a SIMULATOR default here (the weld carries the can, so the welded grasp
    # is not an honest hold signal), a real tactile/physics signal only on hardware. F7: no label
    # is tactile-derived. The task outcome requires all three stages to pass.
    grasp_stable = True
    labels = list(episode.labels) + [
        Label(UPRIGHT_SUCCESS, bool(res.upright_success), LabelProvenance.SIMULATOR),
        Label(GRASP_STABLE, grasp_stable, LabelProvenance.SIMULATOR),
        Label(LABEL_VISIBLE, bool(res.label_visible), LabelProvenance.AUTO_VISION),
        Label(LABEL_UP_COS, float(res.label_nz), LabelProvenance.AUTO_VISION),
    ]
    passed = bool(res.upright_success and grasp_stable and res.label_visible)
    stamped = replace(
        episode,
        outcome=Outcome.SUCCESS if passed else Outcome.FAILURE,
        stream_keys=tuple(streams.keys()),
        labels=labels,
    )
    return RecordedEpisode(episode=stamped, streams=streams)


class SimSource:
    """A `SensorSource` reading one modality from a shared, externally-stepped SimWorld."""

    def __init__(self, world: SimWorld, modality: Modality) -> None:
        self.world = world
        self.modality = modality

    def start(self) -> None:
        pass

    def read(self) -> Sample:
        return self.world.sample(self.modality, self.world.time_ns)

    def stream(self) -> Iterator[Sample]:
        while True:
            yield self.read()

    def stop(self) -> None:
        pass
