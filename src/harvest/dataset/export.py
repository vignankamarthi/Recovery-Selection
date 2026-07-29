"""HF export (Phase 1.8).

Writes the HuggingFace-loadable METADATA table, one row per episode tagged with its by-can
split, plus the human-facing README card. The raw sensor arrays are large binary and are
exported through the io/flat_npz adapter (the on-disk format boundary), referenced here by
`episode_id`. This keeps the dataset layer free of any on-disk format knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from harvest.dataset.card import dataset_card
from harvest.dataset.splits import Split, assigned_split
from harvest.io._serde import episode_to_dict
from schema.episode import Episode


def _row(episode: Episode, split: Split) -> dict:
    # Reuse the shared episode -> dict encoding (id, can_id, condition, outcome, stream_keys, labels,
    # metadata -- the metadata carries the 1.10 competence_tier tag), adding only the by-can split tag.
    return {**episode_to_dict(episode), "split": assigned_split(split, episode)}


def to_hf_rows(episodes: Iterable[Episode], split: Split) -> list[dict]:
    """One JSON-serializable row per episode, tagged with its split. Ready for
    `datasets.Dataset.from_list(...)` when the optional `datasets` package is present."""
    return [_row(ep, split) for ep in episodes]


def export_hf(episodes: Iterable[Episode], split: Split, out_dir: Path | str) -> None:
    """Write `metadata.jsonl` (split-tagged rows) and `README.md` (the card) to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(episodes)
    (out_dir / "README.md").write_text(dataset_card(episodes, split))
    with (out_dir / "metadata.jsonl").open("w") as f:
        for ep in episodes:
            f.write(json.dumps(_row(ep, split)) + "\n")
