"""Tests for the dataset card, split viz, and HF export (Phase 1.8). Pure logic, no sim."""

import json

from harvest.dataset.card import dataset_card, dataset_stats, render_split_table
from harvest.dataset.export import export_hf, to_hf_rows
from harvest.dataset.splits import assigned_split, split_by_can
from schema.episode import ConditionClass, Episode, Outcome

_CONDS = list(ConditionClass)


def _episodes(cans_per_condition=8, episodes_per_can=2):
    eps = []
    for c in _CONDS:
        for i in range(cans_per_condition):
            can_id = f"can-{c.value}-{i:03d}"
            for k in range(episodes_per_can):
                # make deformed cans fail sometimes so success rates vary by condition
                fail = c is not ConditionClass.NOMINAL and (i % 3 == 0)
                out = Outcome.FAILURE if fail else Outcome.SUCCESS
                eps.append(Episode(f"{can_id}-ep{k}", can_id, c, outcome=out))
    return eps


def test_dataset_stats_counts_cans_episodes_and_outcomes():
    eps = _episodes()
    split = split_by_can(eps, seed=0)
    stats = dataset_stats(eps, split)
    assert stats["total_episodes"] == len(eps)
    assert stats["total_cans"] == len({e.can_id for e in eps})
    # per-condition can counts sum to the total cans
    assert sum(stats["cans_per_condition"].values()) == stats["total_cans"]
    # per-split can counts sum to total cans and are leak-free
    assert sum(stats["cans_per_split"].values()) == stats["total_cans"]
    assert stats["leak_free"] is True


def test_dataset_card_is_markdown_with_caveats_and_key_facts():
    eps = _episodes()
    split = split_by_can(eps, seed=0)
    card = dataset_card(eps, split)
    assert card.startswith("#")
    low = card.lower()
    # the load-bearing honesty: leak-free by-can splits + sim caveats
    assert "leak-free" in low or "leak free" in low
    assert "caveat" in low
    assert "sim" in low
    # every condition class named
    for c in _CONDS:
        assert c.value in card


def test_render_split_table_has_conditions_and_split_columns():
    eps = _episodes()
    split = split_by_can(eps, seed=0)
    table = render_split_table(eps, split)
    assert "train" in table and "val" in table and "test" in table
    for c in _CONDS:
        assert c.value in table


def test_to_hf_rows_tags_each_episode_with_its_split():
    eps = _episodes()
    split = split_by_can(eps, seed=0)
    rows = to_hf_rows(eps, split)
    assert len(rows) == len(eps)
    by_id = {e.episode_id: e for e in eps}
    for row in rows:
        assert row["split"] == assigned_split(split, by_id[row["episode_id"]])
        assert row["condition"] in {c.value for c in _CONDS}
        assert row["outcome"] in {"success", "failure"}


def test_export_hf_writes_card_and_split_tagged_metadata(tmp_path):
    eps = _episodes()
    split = split_by_can(eps, seed=0)
    export_hf(eps, split, tmp_path)

    readme = (tmp_path / "README.md").read_text()
    assert readme.startswith("#")
    lines = (tmp_path / "metadata.jsonl").read_text().splitlines()
    assert len(lines) == len(eps)
    rows = [json.loads(ln) for ln in lines]
    by_id = {e.episode_id: e for e in eps}
    for row in rows:
        assert row["split"] == assigned_split(split, by_id[row["episode_id"]])
