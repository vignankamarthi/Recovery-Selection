"""Tests for the episode-protocol state machine (Phase 1.5). Synthetic data only."""

from harvest.protocol.protocol import (
    EpisodeProtocol,
    Phase,
    run_episode,
)
from harvest.sensors.mock import MockSource
from schema.episode import ConditionClass, Episode, LabelProvenance, Outcome
from schema.streams import Modality


def _episode() -> Episode:
    return Episode(episode_id="e1", can_id="c1", condition=ConditionClass.BODY_DENT)


def _sources():
    return {
        "tactile": MockSource(Modality.TACTILE, start_ns=0),
        "rgb_overhead": MockSource(Modality.RGB_OVERHEAD, start_ns=0),
    }


def test_phase_sequence_is_place_grasp_reorient_verify_done():
    p = EpisodeProtocol()
    seq = [p.phase]
    while not p.is_done():
        seq.append(p.advance())
    assert seq == [Phase.PLACE, Phase.GRASP, Phase.REORIENT, Phase.VERIFY, Phase.DONE]


def test_advance_saturates_at_done():
    p = EpisodeProtocol()
    for _ in range(20):
        p.advance()
    assert p.phase is Phase.DONE


def test_run_episode_success_when_score_meets_threshold():
    rec = run_episode(_episode(), _sources(), scorer=lambda r: 0.6, n_samples=3, threshold=0.2)
    assert rec.episode.outcome is Outcome.SUCCESS
    by_name = {l.name: l for l in rec.episode.labels}
    assert by_name["label_visible"].value is True
    assert by_name["label_visibility_score"].value == 0.6
    # streams captured across the episode
    assert set(rec.streams) == {"tactile", "rgb_overhead"}
    assert all(len(v) == 3 for v in rec.streams.values())


def test_run_episode_failure_below_threshold():
    rec = run_episode(_episode(), _sources(), scorer=lambda r: 0.05, threshold=0.2)
    assert rec.episode.outcome is Outcome.FAILURE
    by_name = {l.name: l for l in rec.episode.labels}
    assert by_name["label_visible"].value is False


def test_visibility_labels_are_vision_provenance_not_tactile():
    # F7: outcome labels come from the overhead camera, never the tactile stream.
    rec = run_episode(_episode(), _sources(), scorer=lambda r: 0.9)
    assert all(l.provenance is LabelProvenance.AUTO_VISION for l in rec.episode.labels)
    assert rec.episode.is_tactile_label_confounded() is False


def test_default_scorer_returns_unit_interval():
    rec = run_episode(_episode(), _sources(), n_samples=2)
    score = {l.name: l.value for l in rec.episode.labels}["label_visibility_score"]
    assert 0.0 <= score <= 1.0
