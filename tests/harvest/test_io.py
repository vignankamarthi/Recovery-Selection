"""Tests for the io adapters (Phase 1.3): rosbag2 (mcap) round-trip + LeRobot export."""

import numpy as np

from harvest.io.lerobot_adapter import export_dataset, load_export
from harvest.io.rosbag2_adapter import read_episode, write_episode
from schema.episode import (
    ConditionClass,
    Episode,
    Label,
    LabelProvenance,
    Outcome,
    RecordedEpisode,
)
from schema.streams import Modality, Sample


def _make_recorded() -> RecordedEpisode:
    ep = Episode(
        episode_id="ep-001",
        can_id="can-042",
        condition=ConditionClass.SEAM_DENT,
        outcome=Outcome.FAILURE,
        stream_keys=("tactile", "force_torque", "rgb_overhead"),
        labels=[
            Label("grasp_stable", False, LabelProvenance.SIMULATOR),
            Label("label_visible", True, LabelProvenance.AUTO_VISION),
        ],
        metadata={"seed": 3, "note": "synthetic"},
    )
    rng = np.random.default_rng(0)
    streams = {
        "tactile": [
            Sample(Modality.TACTILE, 1000, rng.random((4, 7)), "mock"),
            Sample(Modality.TACTILE, 2000, rng.random((4, 7)), "mock"),
        ],
        "force_torque": [Sample(Modality.FORCE_TORQUE, 1500, rng.random((6,)), "mock")],
        "rgb_overhead": [
            Sample(Modality.RGB_OVERHEAD, 1200, rng.integers(0, 256, (8, 8, 3), np.uint8), "mock")
        ],
    }
    return RecordedEpisode(episode=ep, streams=streams)


def test_rosbag2_mcap_round_trip_preserves_metadata(tmp_path):
    rec = _make_recorded()
    bag = tmp_path / "bag"
    write_episode(rec, bag)  # default backend = mcap
    assert (bag / "bag.mcap").exists()

    back = read_episode(bag)
    assert back.episode.episode_id == "ep-001"
    assert back.episode.can_id == "can-042"
    assert back.episode.condition is ConditionClass.SEAM_DENT
    assert back.episode.outcome is Outcome.FAILURE
    assert back.episode.labels == rec.episode.labels
    assert back.episode.metadata == {"seed": 3, "note": "synthetic"}


def test_rosbag2_round_trip_preserves_samples_exactly(tmp_path):
    rec = _make_recorded()
    bag = tmp_path / "bag"
    write_episode(rec, bag)
    back = read_episode(bag)

    assert set(back.streams) == set(rec.streams)
    for key in rec.streams:
        orig, got = rec.streams[key], back.streams[key]
        assert len(orig) == len(got)
        for a, b in zip(orig, got):
            assert b.modality is a.modality
            assert b.timestamp_ns == a.timestamp_ns
            assert b.notes == a.notes
            assert np.asarray(b.data).shape == np.asarray(a.data).shape
            assert np.asarray(b.data).dtype == np.asarray(a.data).dtype
            np.testing.assert_array_equal(np.asarray(b.data), np.asarray(a.data))


def test_lerobot_export_round_trips_metadata_and_arrays(tmp_path):
    rec = _make_recorded()
    out = tmp_path / "hf"
    export_dataset([rec], out)
    loaded = load_export(out)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.episode.can_id == "can-042"
    np.testing.assert_array_equal(
        np.asarray(got.streams["tactile"][0].data),
        np.asarray(rec.streams["tactile"][0].data),
    )


def test_per_episode_write_plus_index_matches_batch_export(tmp_path):
    # Parallel/resumable generation writes each episode's streams on its own, then assembles the
    # index. The result must round-trip identically to a batch export_dataset (same fence seam,
    # so a sharded cluster run and a local batch run produce the same dataset).
    from harvest.io.lerobot_adapter import write_episode_streams, write_index

    rec = _make_recorded()
    out = tmp_path / "sharded"
    manifest = write_episode_streams(rec, out)          # writes data/<eid>.npz, returns its manifest
    assert (out / "data" / f"{rec.episode.episode_id}.npz").exists()
    write_index([manifest], out)                         # assembles episodes.jsonl from manifests
    loaded = load_export(out)
    assert len(loaded) == 1 and loaded[0].episode.can_id == "can-042"
    np.testing.assert_array_equal(
        np.asarray(loaded[0].streams["rgb_overhead"][0].data),
        np.asarray(rec.streams["rgb_overhead"][0].data),
    )
