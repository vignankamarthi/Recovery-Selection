"""Tests for per-condition can geometry (Phase 1.7). Physical sim; slower than unit tests.

Covers C1 (rigid condition geometry as distinct variants), per-can determinism (each
can_id is a fixed distinct physical unit, load-bearing for the F6 by-can splits), and C3
(deformed cans produce condition-correlated organic grasp failures under the fixed script).
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from harvest.control.policy import ScriptedGraspPolicy  # noqa: E402
from harvest.sim.scene import build_scene, can_seed_from_id  # noqa: E402
from harvest.sim.world import SimWorld  # noqa: E402
from schema.episode import ConditionClass  # noqa: E402


def _grasp_lifts(can_id: str, condition: ConditionClass) -> bool:
    """Whether the grasp policy alone lifts this can. C3 is a grasp-reach claim, so the test
    drives the grasp directly (not the full episode, whose outcome also needs the reorient)."""
    w = SimWorld(can_pos=(0.5, 0.0, 0.11), condition=condition,
                 can_seed=can_seed_from_id(can_id))
    return ScriptedGraspPolicy().run(w)


def _geom(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _can_geoms(model):
    """geom ids belonging to the can body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can")
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def test_every_condition_compiles():
    for c in ConditionClass:
        model = build_scene(condition=c, can_seed=can_seed_from_id(f"{c.value}"))
        assert model.nq == 22  # 7 arm + 8 gripper + 7 can freejoint
        assert _geom(model, "can_geom") != -1


def test_conditions_have_distinct_geometry():
    nominal = build_scene(condition=ConditionClass.NOMINAL, can_seed=1)
    ng = _geom(nominal, "can_geom")
    assert nominal.geom_type[ng] == mujoco.mjtGeom.mjGEOM_CYLINDER

    # body_dent is an out-of-round ellipsoid, not a cylinder.
    body = build_scene(condition=ConditionClass.BODY_DENT, can_seed=1)
    bg = _geom(body, "can_geom")
    assert body.geom_type[bg] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
    assert body.geom_size[bg][0] != body.geom_size[bg][1]  # x != y radius

    # seam_dent and bulge each add a second, distinct feature geom.
    seam = build_scene(condition=ConditionClass.SEAM_DENT, can_seed=1)
    assert _geom(seam, "can_seam") != -1
    bulge = build_scene(condition=ConditionClass.BULGE, can_seed=1)
    assert _geom(bulge, "can_bulge") != -1

    # rust is recolored and lower-friction than a nominal can.
    rust = build_scene(condition=ConditionClass.RUST, can_seed=1)
    rg = _geom(rust, "can_geom")
    assert not np.allclose(rust.geom_rgba[rg], nominal.geom_rgba[ng])
    assert rust.geom_friction[rg][0] < nominal.geom_friction[ng][0]


def test_same_can_id_is_deterministic_geometry():
    a = build_scene(condition=ConditionClass.BODY_DENT, can_seed=can_seed_from_id("canA"))
    b = build_scene(condition=ConditionClass.BODY_DENT, can_seed=can_seed_from_id("canA"))
    ga = _can_geoms(a)
    assert np.allclose(a.geom_size[ga], b.geom_size[_can_geoms(b)])
    assert np.allclose(a.geom_pos[ga], b.geom_pos[_can_geoms(b)])


def test_different_can_ids_differ():
    a = build_scene(condition=ConditionClass.BODY_DENT, can_seed=can_seed_from_id("canA"))
    c = build_scene(condition=ConditionClass.BODY_DENT, can_seed=can_seed_from_id("canC"))
    ga, gc = _can_geoms(a), _can_geoms(c)
    same_size = np.allclose(a.geom_size[ga], c.geom_size[gc])
    same_pos = np.allclose(a.geom_pos[ga], c.geom_pos[gc])
    assert not (same_size and same_pos)


@pytest.mark.slow
def test_deformed_cans_fail_more_than_nominal():
    """C3: the fixed scripted grasp fails more often on deformed cans (organic,
    condition-correlated failure), while nominal cans reliably succeed."""
    def grasp_rate(condition, n):
        grasped = [
            _grasp_lifts(f"can-{condition.value}-{i}", condition) for i in range(n)
        ]
        return sum(grasped) / n

    nominal_rate = grasp_rate(ConditionClass.NOMINAL, 8)
    body_rate = grasp_rate(ConditionClass.BODY_DENT, 8)

    assert nominal_rate >= 0.75      # nominal is reliably graspable
    assert body_rate < nominal_rate  # deformation induces condition-correlated grasp failures
