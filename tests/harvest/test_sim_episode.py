"""Tests for the sim episode runner + the control split (Phase 1.7c). Physical sim; slow."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from harvest.control.backend import RobotBackend  # noqa: E402
from harvest.control.policy import ScriptedGraspPolicy  # noqa: E402
from harvest.sensors.base import SensorSource  # noqa: E402
from harvest.sim.episode import SimSource, record_episode  # noqa: E402
from harvest.sim.world import SimWorld  # noqa: E402
from schema.episode import (  # noqa: E402
    ConditionClass,
    Episode,
    LabelProvenance,
    Outcome,
    RecordedEpisode,
)
from schema.streams import Modality  # noqa: E402

_FAST = (Modality.PROPRIOCEPTION, Modality.FORCE_TORQUE, Modality.TACTILE)


def test_simworld_satisfies_the_robot_backend_interface():
    # The extensible split: the sim is a RobotBackend, so a policy drives it backend-agnostically.
    assert isinstance(SimWorld(), RobotBackend)


def test_scripted_grasp_policy_runs_on_the_backend_and_returns_success():
    w = SimWorld(can_pos=(0.5, 0.0, 0.05), can_seed=1)
    ok = ScriptedGraspPolicy().run(w)
    assert isinstance(ok, bool)
    assert ok is w.grasp_success()


def test_record_episode_produces_labeled_recorded_episode_with_sim_ground_truth():
    ep = Episode("e1", "can1", ConditionClass.NOMINAL)
    rec = record_episode(ep, modalities=_FAST)

    assert isinstance(rec, RecordedEpisode)
    assert set(rec.streams) == {m.value for m in _FAST}
    assert all(len(v) > 0 for v in rec.streams.values())
    assert rec.episode.outcome in (Outcome.SUCCESS, Outcome.FAILURE)

    # F7: the grasp label is simulator ground truth, never the tactile stream.
    grasp = {l.name: l for l in rec.episode.labels}["grasp_stable"]
    assert grasp.provenance is LabelProvenance.SIMULATOR
    assert rec.episode.is_tactile_label_confounded() is False


def test_record_episode_emits_the_three_graded_stage_labels():
    # 1.7d: the lying-can pipeline is a 3-stage graded task. `upright_success` (right the lying
    # can) and `label_visible` (present the label to the overhead camera) are real sim signals;
    # `grasp_stable` is a SIMULATOR default (real on hardware). The outcome requires all three.
    ep = Episode("v1", "can-v", ConditionClass.NOMINAL)
    rec = record_episode(ep, modalities=_FAST, record_every=999)
    labels = {l.name: l for l in rec.episode.labels}
    assert {"upright_success", "grasp_stable", "label_visible", "label_up_cos"} <= set(labels)

    up, grasp, vis, cos = (
        labels["upright_success"], labels["grasp_stable"], labels["label_visible"], labels["label_up_cos"],
    )
    assert up.provenance is LabelProvenance.SIMULATOR      # a physics/search read, never tactile
    assert grasp.provenance is LabelProvenance.SIMULATOR   # sim default here, real on hardware
    assert vis.provenance is LabelProvenance.AUTO_VISION   # an overhead-camera read, never tactile
    assert cos.provenance is LabelProvenance.AUTO_VISION
    assert all(isinstance(l.value, bool) for l in (up, grasp, vis))
    assert -1.0 <= cos.value <= 1.0

    passed = up.value and grasp.value and vis.value
    expected = Outcome.SUCCESS if passed else Outcome.FAILURE
    assert rec.episode.outcome is expected


def test_record_episode_records_image_modality():
    ep = Episode("e2", "can2", ConditionClass.NOMINAL)
    rec = record_episode(ep, modalities=(Modality.RGB_OVERHEAD,), record_every=30)
    frames = rec.streams["rgb_overhead"]
    assert len(frames) > 0
    assert np.asarray(frames[0].data).ndim == 3


def test_tactile_is_spatially_structured_during_grasp():
    # C4: while the fingers actually grip the can, the tactile pressure map varies spatially (not
    # a degenerate 2-value proxy). We probe the proxy at the grasp contact directly rather than
    # via episode sampling: contact is brief in wall-clock, and during the WELDED reorient the
    # kinematically-placed can gives a near-uniform (2-value) map (a documented sim-tactile
    # limitation, which is why the sim tactile ablation is a smoke test, not the reported result).
    import math

    from harvest.sim import reorient

    lying = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    w = SimWorld(can_pos=(0.5, 0.0, 0.11), can_seed=7, can_quat=lying)
    frames = []
    reorient._grasp_head(w, on_step=lambda: frames.append(np.asarray(w.tactile()).copy()))
    distinct = max((len(np.unique(t[t > 0])) for t in frames if t.max() > 0), default=0)
    assert distinct > 2


def test_simsource_satisfies_the_sensor_protocol():
    w = SimWorld()
    src = SimSource(w, Modality.PROPRIOCEPTION)
    assert isinstance(src, SensorSource)
    src.start()
    s = src.read()
    assert s.modality is Modality.PROPRIOCEPTION
    assert np.asarray(s.data).shape == (7,)
