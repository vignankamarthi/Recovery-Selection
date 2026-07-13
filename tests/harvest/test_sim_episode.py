"""Tests for sim episode generation (Phase 1.6d). Physical sim; slower than unit tests."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from harvest.sensors.base import SensorSource  # noqa: E402
from harvest.sim.episode import SimSource, sim_episode  # noqa: E402
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


def test_sim_episode_produces_labeled_recorded_episode_with_sim_ground_truth():
    ep = Episode("e1", "can1", ConditionClass.NOMINAL)
    rec = sim_episode(ep, can_pos=(0.5, 0.0, 0.05), modalities=_FAST)

    assert isinstance(rec, RecordedEpisode)
    assert set(rec.streams) == {m.value for m in _FAST}
    assert all(len(v) > 0 for v in rec.streams.values())
    assert rec.episode.outcome in (Outcome.SUCCESS, Outcome.FAILURE)

    # F7: the grasp label is simulator ground truth, never the tactile stream.
    grasp = {l.name: l for l in rec.episode.labels}["grasp_stable"]
    assert grasp.provenance is LabelProvenance.SIMULATOR
    assert rec.episode.is_tactile_label_confounded() is False


def test_sim_episode_records_image_modality():
    ep = Episode("e2", "can2", ConditionClass.NOMINAL)
    rec = sim_episode(ep, modalities=(Modality.RGB_OVERHEAD,), record_every=30)
    frames = rec.streams["rgb_overhead"]
    assert len(frames) > 0
    assert np.asarray(frames[0].data).ndim == 3


def test_tactile_is_spatially_structured_during_grasp():
    # C4: not the old degenerate 2-value proxy; the pressure map varies spatially.
    ep = Episode("et", "cant", ConditionClass.NOMINAL)
    rec = sim_episode(ep, modalities=(Modality.TACTILE,))
    tac = [np.asarray(s.data) for s in rec.streams["tactile"]]
    distinct = max((len(np.unique(t[t > 0])) for t in tac if t.max() > 0), default=0)
    assert distinct > 2


def test_simsource_satisfies_the_sensor_protocol():
    w = SimWorld()
    src = SimSource(w, Modality.PROPRIOCEPTION)
    assert isinstance(src, SensorSource)
    src.start()
    s = src.read()
    assert s.modality is Modality.PROPRIOCEPTION
    assert np.asarray(s.data).shape == (7,)
