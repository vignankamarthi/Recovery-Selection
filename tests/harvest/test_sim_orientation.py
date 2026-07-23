"""Unknown-orientation cans + nutrition label + orientation-aware grasp (Phase 1.7b).

Covers the proposal's "can of unknown orientation": cans settle upright or lying, carry a
nutrition-label patch, and the scripted grasp aligns the fingers across a lying can's short
axis. The grasp stays a frozen, imperfect base policy (rich in orientation/condition-correlated
failures, the substrate Part 2 consumes), so lying and deformed cans fail far more often.
"""

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from harvest.control.policy import ScriptedGraspPolicy  # noqa: E402
from harvest.sim.scene import build_scene, can_seed_from_id  # noqa: E402
from harvest.sim.world import SimWorld  # noqa: E402
from schema.episode import ConditionClass  # noqa: E402

_UPRIGHT = (1.0, 0.0, 0.0, 0.0)
_LYING = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # 90 deg about x


def _grasp_lifts(can_id: str, can_quat, condition=ConditionClass.NOMINAL) -> bool:
    """Run the grasp policy alone (no reorient) and return whether it lifted the can.

    These tests probe the orientation-aware grasp's reach/lift, so they drive the grasp
    directly rather than the full pick-and-reorient episode (whose grasp label reflects the
    grasp surviving the reorient, a stricter, different property)."""
    w = SimWorld(can_pos=(0.5, 0.0, 0.11), condition=condition,
                 can_seed=can_seed_from_id(can_id), can_quat=can_quat)
    return ScriptedGraspPolicy().run(w)


def test_can_settles_upright_or_lying_from_orientation():
    up = SimWorld(can_pos=(0.5, 0.0, 0.11), can_quat=_UPRIGHT)
    lying = SimWorld(can_pos=(0.5, 0.0, 0.11), can_quat=_LYING)
    assert up.can_is_upright() is True
    assert lying.can_is_upright() is False
    # a lying can rests lower (on its side, ~radius) than an upright one (~half height)
    assert lying.can_position()[2] < up.can_position()[2]


def test_nutrition_label_geom_present():
    model = build_scene(condition=ConditionClass.NOMINAL, can_seed=5)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "can_label") != -1


def test_label_pose_is_on_the_wall_and_rotates_with_the_can():
    up = SimWorld(can_pos=(0.5, 0.0, 0.11), can_seed=3, can_quat=_UPRIGHT)
    lying = SimWorld(can_pos=(0.5, 0.0, 0.11), can_seed=3, can_quat=_LYING)
    _, n_up = up.can_label_pose()
    _, n_lying = lying.can_label_pose()
    assert abs(np.linalg.norm(n_up) - 1.0) < 1e-6
    # the label normal points differently once the can is laid on its side
    assert not np.allclose(n_up, n_lying, atol=1e-3)


@pytest.mark.slow
def test_upright_nominal_grasp_succeeds():
    assert _grasp_lifts("can-u", _UPRIGHT) is True


@pytest.mark.slow
def test_orientation_aware_grasp_lifts_some_lying_cans():
    # The wrist-alignment must make lying cans graspable at all (not 0%), while staying an
    # imperfect base policy (not all succeed). This is the orientation-aware grasp working.
    grasped = [_grasp_lifts(f"can-lying-{i}", _LYING) for i in range(8)]
    n_succ = sum(grasped)
    assert n_succ >= 2  # the alignment works on lying cans
    assert n_succ < 8   # but a lying grasp is genuinely harder (failure-rich)


@pytest.mark.slow
def test_grasp_is_deterministic_under_can_id_and_orientation():
    assert _grasp_lifts("can-det", _LYING) is _grasp_lifts("can-det", _LYING)
