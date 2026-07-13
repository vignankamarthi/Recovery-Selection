"""Sim episode generation (Phase 1.6d).

Runs a scripted approach -> grasp -> lift on the physical Gen3 + gripper scene and
records the multimodal streams into a `RecordedEpisode`. The grasp-stability label comes
from SIMULATOR ground truth (F7), never the tactile stream. Also exposes `SimSource`, a
`SensorSource` reading one modality from a shared world, for interface parity with the
mock and (later) real driver.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Sequence

import mujoco
import numpy as np

from harvest.sim.scene import CAN_HALF_HEIGHT, can_seed_from_id
from harvest.sim.world import SimWorld
from schema.episode import Episode, Label, LabelProvenance, Outcome, RecordedEpisode
from schema.streams import Modality, Sample

DEFAULT_MODALITIES: tuple[Modality, ...] = (
    Modality.PROPRIOCEPTION,
    Modality.FORCE_TORQUE,
    Modality.TACTILE,
    Modality.RGB_OVERHEAD,
    Modality.RGB_WRIST,
)


def _ik_move(world: SimWorld, target, max_steps=150, tol=0.008, damping=0.1, gain=0.6, on_step=None) -> None:
    """Damped-least-squares IK: drive the gripper_pinch site to `target` by stepping."""
    model, data = world.model, world.data
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    target = np.asarray(target, dtype=float)
    for _ in range(max_steps):
        err = target - data.site_xpos[world._pinch_sid]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, world._pinch_sid)
        j = jacp[:, :7]
        dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(3), err)
        world.set_arm(data.qpos[:7] + gain * dq)
        world.step(5)
        if on_step is not None:
            on_step()


def sim_episode(
    episode: Episode,
    can_pos: tuple[float, float, float] = (0.5, 0.0, CAN_HALF_HEIGHT),
    modalities: Sequence[Modality] = DEFAULT_MODALITIES,
    record_every: int = 15,
) -> RecordedEpisode:
    """Run one scripted grasp episode and return a RecordedEpisode with a sim label.

    The can's physical variant is `episode.condition`, and its per-can geometry is seeded
    from `episode.can_id`, so every can_id is a fixed, distinct physical unit (F5/F6)."""
    world = SimWorld(
        can_pos,
        condition=episode.condition,
        can_seed=can_seed_from_id(episode.can_id),
    )
    streams: dict[str, list[Sample]] = {m.value: [] for m in modalities}
    n = [0]

    def tick() -> None:
        n[0] += 1
        if n[0] % record_every == 0:
            ts = world.time_ns
            for m in modalities:
                streams[m.value].append(world.sample(m, ts))

    can = np.asarray(can_pos, dtype=float)
    world.set_gripper(0.0)                                   # open
    _ik_move(world, can + [0, 0, 0.12], on_step=tick)       # approach above
    _ik_move(world, can + [0, 0, 0.02], on_step=tick)       # descend to grasp height
    world.set_gripper(1.0)                                   # close
    for _ in range(40):
        world.step(5)
        tick()                                              # settle the grasp
    _ik_move(world, can + [0, 0, 0.18], on_step=tick)       # lift
    for _ in range(20):
        world.step(5)
        tick()                                              # settle at top

    success = world.grasp_success()
    labels = list(episode.labels) + [
        Label("grasp_stable", success, LabelProvenance.SIMULATOR),
    ]
    stamped = replace(
        episode,
        outcome=Outcome.SUCCESS if success else Outcome.FAILURE,
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
