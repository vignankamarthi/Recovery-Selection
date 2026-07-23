"""End-to-end synthetic dataset generation (Phase 1.9). Physics sim; slow."""

import json

import pytest

mujoco = pytest.importorskip("mujoco")
from harvest.dataset.card import dataset_card  # noqa: E402
from harvest.dataset.export import export_hf  # noqa: E402
from harvest.dataset.generate import generate_dataset  # noqa: E402
from harvest.dataset.splits import split_by_can  # noqa: E402
from schema.episode import ConditionClass, Outcome, RecordedEpisode  # noqa: E402
from schema.streams import Modality  # noqa: E402

_FAST = (Modality.PROPRIOCEPTION, Modality.FORCE_TORQUE, Modality.TACTILE)


@pytest.mark.slow
def test_generate_dataset_is_recorded_split_and_exportable(tmp_path):
    conds = (ConditionClass.NOMINAL, ConditionClass.BODY_DENT)
    recorded = generate_dataset(
        conditions=conds, cans_per_condition=2, poses_per_can=2, modalities=_FAST, seed=0
    )
    # 2 conditions x 2 cans x 2 poses = 8 episodes over 4 distinct cans
    assert len(recorded) == 8
    assert all(isinstance(r, RecordedEpisode) for r in recorded)
    episodes = [r.episode for r in recorded]
    assert len({e.can_id for e in episodes}) == 4
    assert all(e.outcome in (Outcome.SUCCESS, Outcome.FAILURE) for e in episodes)
    # every episode actually recorded its streams
    assert all(len(r.streams[Modality.PROPRIOCEPTION.value]) > 0 for r in recorded)

    # by-can split is leak-free and covers every generated can
    split = split_by_can(episodes, seed=0)
    assert split.is_leak_free()
    assert split.train_can_ids | split.val_can_ids | split.test_can_ids == {e.can_id for e in episodes}

    # packages to disk with a card
    export_hf(episodes, split, tmp_path)
    assert (tmp_path / "README.md").read_text().startswith("#")
    assert len((tmp_path / "metadata.jsonl").read_text().splitlines()) == 8
    assert "caveat" in dataset_card(episodes, split).lower()


@pytest.mark.slow
def test_generation_is_deterministic_under_seed():
    a = generate_dataset(
        conditions=(ConditionClass.BODY_DENT,), cans_per_condition=2, poses_per_can=1,
        modalities=_FAST, seed=3,
    )
    b = generate_dataset(
        conditions=(ConditionClass.BODY_DENT,), cans_per_condition=2, poses_per_can=1,
        modalities=_FAST, seed=3,
    )
    assert [r.episode.outcome for r in a] == [r.episode.outcome for r in b]
