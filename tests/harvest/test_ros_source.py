"""Tests for RosSource -- synthetic, a fake bridge (no ROS, no arm)."""
from __future__ import annotations

import pytest

from harvest.sensors.ros_source import (
    GEN3_ARM_JOINTS,
    force_torque_source,
    ordered_by_name,
    proprioception_source,
)
from schema.streams import Modality


class FakeJointState:
    def __init__(self, name, position=None, velocity=None, effort=None):
        self.name = list(name)
        self.position = list(position) if position is not None else [0.0] * len(name)
        self.velocity = list(velocity) if velocity is not None else [0.0] * len(name)
        self.effort = list(effort) if effort is not None else [0.0] * len(name)


class FakeBridge:
    def __init__(self, js=None):
        self._js = js
        self.started = False

    def start(self):
        self.started = True

    def latest_joint_state(self):
        return self._js

    def stop(self):
        self.started = False

    def set(self, js):
        self._js = js


# Realistic /joint_states as the real Gen3 publishes it: gripper knuckle interleaved, arm joints
# out of numeric order (this is exactly what we saw on the bench).
SCRAMBLED = FakeJointState(
    name=["joint_1", "robotiq_85_left_knuckle_joint", "joint_2", "joint_4",
          "joint_5", "joint_3", "joint_6", "joint_7"],
    position=[0.10, 0.03, -0.50, -2.68, 0.18, 2.89, -0.46, 1.52],
    effort=[0.12, 9.9, -0.58, 2.64, -0.27, 1.01, -1.20, 0.53],
)
# joint_1..joint_7 in order = positions [0.10, -0.50, 2.89, -2.68, 0.18, -0.46, 1.52]


def test_ordered_by_name_reorders_and_drops_gripper():
    assert ordered_by_name(SCRAMBLED, GEN3_ARM_JOINTS, "position") == [
        0.10, -0.50, 2.89, -2.68, 0.18, -0.46, 1.52
    ]


def test_ordered_by_name_missing_joint_raises():
    with pytest.raises(KeyError):
        ordered_by_name(FakeJointState(name=["joint_1"]), GEN3_ARM_JOINTS, "position")


def test_proprioception_source_reads_arm_positions():
    b = FakeBridge(SCRAMBLED)
    s = proprioception_source(b)
    s.start()
    assert b.started
    sample = s.read()
    assert sample.modality == Modality.PROPRIOCEPTION
    assert sample.data == [0.10, -0.50, 2.89, -2.68, 0.18, -0.46, 1.52]
    assert sample.timestamp_ns > 0
    assert "proprioception" in sample.notes.lower() or "positions" in sample.notes.lower()


def test_force_torque_source_reads_arm_efforts():
    s = force_torque_source(FakeBridge(SCRAMBLED))
    s.start()
    sample = s.read()
    assert sample.modality == Modality.FORCE_TORQUE
    assert sample.data == [0.12, -0.58, 1.01, 2.64, -0.27, -1.20, 0.53]


def test_stream_yields_monotonic_timestamps():
    s = proprioception_source(FakeBridge(SCRAMBLED))
    s.start()
    it = s.stream()
    a, b = next(it), next(it)
    assert b.timestamp_ns >= a.timestamp_ns


def test_timeout_when_no_joint_state():
    s = proprioception_source(FakeBridge(None))
    s.start()
    s._poll_timeout_s = 0.05
    with pytest.raises(TimeoutError):
        s.read()
