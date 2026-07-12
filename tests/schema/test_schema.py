"""Tests for the schema contract (Phase 1.1). Synthetic data only."""

import pytest

from schema.episode import (
    ConditionClass,
    Episode,
    Label,
    LabelProvenance,
    Outcome,
)
from schema.streams import Modality, Sample, StreamSpec


def test_condition_taxonomy_is_the_five_usda_classes():
    assert {c.value for c in ConditionClass} == {
        "nominal",
        "body_dent",
        "seam_dent",
        "bulge",
        "rust",
    }


def test_outcome_values():
    assert Outcome.SUCCESS.value == "success"
    assert Outcome.FAILURE.value == "failure"


def test_label_carries_provenance():
    lab = Label(name="grasp_stable", value=True, provenance=LabelProvenance.SIMULATOR)
    assert lab.provenance is LabelProvenance.SIMULATOR


def test_episode_requires_can_id():
    ep = Episode(episode_id="e1", can_id="can-007", condition=ConditionClass.BODY_DENT)
    assert ep.can_id == "can-007"
    with pytest.raises(TypeError):
        Episode(episode_id="e1", condition=ConditionClass.NOMINAL)  # type: ignore[call-arg]


def test_episode_defaults():
    ep = Episode(episode_id="e1", can_id="c1", condition=ConditionClass.NOMINAL)
    assert ep.outcome is None
    assert ep.stream_keys == ()
    assert ep.labels == []
    assert ep.metadata == {}


def test_is_tactile_label_confounded_true_when_a_label_is_tactile_derived():
    ep = Episode(
        episode_id="e1",
        can_id="c1",
        condition=ConditionClass.SEAM_DENT,
        labels=[
            Label("grasp_stable", True, LabelProvenance.AUTO_TACTILE),
            Label("label_visible", True, LabelProvenance.AUTO_VISION),
        ],
    )
    assert ep.is_tactile_label_confounded() is True


def test_is_tactile_label_confounded_false_for_independent_labels():
    ep = Episode(
        episode_id="e1",
        can_id="c1",
        condition=ConditionClass.NOMINAL,
        labels=[
            Label("grasp_stable", True, LabelProvenance.SIMULATOR),
            Label("label_visible", True, LabelProvenance.AUTO_VISION),
        ],
    )
    assert ep.is_tactile_label_confounded() is False


def test_is_tactile_label_confounded_false_when_no_labels():
    ep = Episode(episode_id="e1", can_id="c1", condition=ConditionClass.NOMINAL)
    assert ep.is_tactile_label_confounded() is False


def test_modalities_include_both_depth_viewpoints():
    # F4 reconciliation: wrist depth (Gen3 RGB-D module) and overhead depth are distinct.
    vals = {m.value for m in Modality}
    assert "depth_wrist" in vals
    assert "depth_overhead" in vals
    assert {"tactile", "force_torque", "proprioception"} <= vals


def test_sample_construct():
    s = Sample(modality=Modality.TACTILE, timestamp_ns=1000, data=[[0.1, 0.2]], notes="mock")
    assert s.modality is Modality.TACTILE
    assert s.timestamp_ns == 1000


def test_streamspec_required_by_default():
    spec = StreamSpec(modality=Modality.RGB_OVERHEAD, nominal_hz=30.0)
    assert spec.required is True
