"""By-can train/val/test splitting (Part 1) -- the load-bearing HARVEST fix (F6).

The single most important correction from the red team: episodes from the same
physical can are correlated, so splitting by episode leaks the same can across
train/test and inflates success rates. Here, a `can_id` appears in EXACTLY ONE split.
This module enforces that invariant.

Design choices to settle when we build (earmarked for Vignan's input):
  - stratify by ConditionClass so each split keeps the five-class balance;
  - target ratios and how to round with only ~50 cans/class (small-N);
  - whether failure-heavy cans are balanced across splits (matters for the recovery
    layer downstream, which feeds on failures).
Stub only, no logic yet (TDD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from schema.episode import Episode


@dataclass(frozen=True)
class Split:
    train_can_ids: frozenset[str]
    val_can_ids: frozenset[str]
    test_can_ids: frozenset[str]

    def is_leak_free(self) -> bool:
        """True iff no can_id appears in more than one split. The invariant. Stub."""
        raise NotImplementedError


def split_by_can(
    episodes: Iterable[Episode],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    stratify_by_condition: bool = True,
    seed: int = 0,
) -> Split:
    """Partition cans (not episodes) into train/val/test. A can lands in one split.
    Stub: implementation follows a failing test asserting `is_leak_free()`."""
    raise NotImplementedError


def assigned_split(split: Split, episode: Episode) -> str:
    """Return which split an episode belongs to, by its can_id. Stub."""
    raise NotImplementedError
