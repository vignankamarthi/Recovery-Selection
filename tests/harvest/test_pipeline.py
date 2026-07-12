"""End-to-end mock pipeline (Phase 1.1-1.5): protocol -> record -> mcap write -> read."""

import numpy as np

from harvest.io.rosbag2_adapter import read_episode, write_episode
from harvest.protocol.protocol import run_episode
from harvest.sensors.mock import MockSource
from schema.episode import ConditionClass, Episode, Outcome
from schema.streams import Modality


def test_full_mock_pipeline_roundtrips_through_mcap(tmp_path):
    episode = Episode(episode_id="pipe-1", can_id="can-9", condition=ConditionClass.BULGE)
    sources = {
        "tactile": MockSource(Modality.TACTILE, start_ns=0),
        "force_torque": MockSource(Modality.FORCE_TORQUE, start_ns=0),
        "rgb_overhead": MockSource(Modality.RGB_OVERHEAD, start_ns=0),
    }

    recorded = run_episode(episode, sources, scorer=lambda r: 0.5, n_samples=3)
    assert recorded.episode.outcome is Outcome.SUCCESS

    bag = tmp_path / "bag"
    write_episode(recorded, bag)  # mcap
    assert (bag / "bag.mcap").exists()

    back = read_episode(bag)
    assert back.episode.episode_id == "pipe-1"
    assert back.episode.outcome is Outcome.SUCCESS
    assert set(back.streams) == set(recorded.streams)
    for key in recorded.streams:
        for a, b in zip(recorded.streams[key], back.streams[key]):
            assert a.timestamp_ns == b.timestamp_ns
            np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))
