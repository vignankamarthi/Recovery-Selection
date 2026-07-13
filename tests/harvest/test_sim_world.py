"""Tests for the MuJoCo SimWorld (Phase 1.6b). Physical sim, structural assertions."""

import numpy as np
import pytest

# Skip the whole module cleanly if the sim stack isn't installed (keeps CI portable).
mujoco = pytest.importorskip("mujoco")
from harvest.sim.world import SimWorld  # noqa: E402


def test_world_has_gen3_arm_and_gripper():
    w = SimWorld()
    # 7 arm actuators + 1 gripper actuator
    assert w.model.nu == 8
    assert w.proprioception().shape == (7,)


def test_step_advances_sim_time():
    w = SimWorld()
    t0 = w.time_ns
    w.step(20)
    assert w.time_ns > t0


def test_render_rgb_and_depth_shapes():
    w = SimWorld()
    rgb = w.render("overhead", height=64, width=64)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    depth = w.render("overhead", depth=True, height=64, width=64)
    assert depth.shape == (64, 64) and depth.dtype == np.float32
    # the wrist camera also renders
    assert w.render("wrist", height=48, width=48).shape == (48, 48, 3)


def test_can_starts_resting_and_unlifted():
    w = SimWorld()
    assert w.can_position().shape == (3,)
    assert w.grasp_success() is False


def test_arm_and_gripper_control_accepted():
    w = SimWorld()
    w.set_arm(np.zeros(7))
    w.set_gripper(1.0)
    w.step(5)  # must not raise
    assert w.data.ctrl[7] == pytest.approx(255.0)


def test_snapshot_restore_gives_deterministic_replay():
    # The recovery-regret oracle (Part 2) needs to replay arms from an identical state.
    w = SimWorld()
    w.step(20)
    snap = w.snapshot()
    w.set_arm(np.zeros(7))
    w.step(30)
    a = w.proprioception().copy()
    # restore + same commands -> bit-identical
    w.restore(snap)
    w.set_arm(np.zeros(7))
    w.step(30)
    np.testing.assert_allclose(a, w.proprioception(), atol=1e-9)
    # restore + different command -> diverges
    w.restore(snap)
    w.set_arm(np.full(7, 0.2))
    w.step(30)
    assert not np.allclose(a, w.proprioception())


def test_force_torque_reads_near_zero_without_contact():
    # Gravity-compensated F/T (C5): a settled no-contact hold carries ~no wrench.
    w = SimWorld()
    w.step(300)
    assert float(np.linalg.norm(w.force_torque())) < 0.1


def test_sim_modality_reads_have_expected_shapes():
    from schema.streams import Modality

    w = SimWorld()
    w.step(5)
    assert w.force_torque().shape == (6,)
    assert w.tactile().shape == (4, 7)
    for mod, shape in [
        (Modality.PROPRIOCEPTION, (7,)),
        (Modality.FORCE_TORQUE, (6,)),
        (Modality.TACTILE, (4, 7)),
    ]:
        s = w.sample(mod, 1234)
        assert s.modality is mod and s.timestamp_ns == 1234 and s.notes == "sim"
        assert np.asarray(s.data).shape == shape
    assert np.asarray(w.sample(Modality.RGB_OVERHEAD, 1).data).ndim == 3
    assert np.asarray(w.sample(Modality.DEPTH_WRIST, 1).data).ndim == 2
