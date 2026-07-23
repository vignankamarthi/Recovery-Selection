"""By-can train/val/test splitting (Part 1) -- the load-bearing HARVEST fix (F6).

The single most important correction from the red team: episodes from the same
physical can are correlated, so splitting by episode leaks the same can across
train/test and inflates success rates. Here, a `can_id` appears in EXACTLY ONE split.
This module enforces that invariant.

Proposed default policy (ratified with Vignan at the Part 1 CHECKPOINT):
  - stratify by ConditionClass, so each split keeps the five-class balance;
  - ratios 0.7 / 0.15 / 0.15, floor-rounded within each class (small-N safe);
  - deterministic under `seed` (a fixed can_id ordering, then a seeded shuffle);
  - failure-heavy balancing across splits is NOT applied yet. It is the one open
    sub-decision, deferred to the CHECKPOINT, because it couples the split to episode
    outcomes and matters mainly for the downstream recovery layer.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from schema.episode import Episode


@dataclass(frozen=True)
class Split:
    train_can_ids: frozenset[str]
    val_can_ids: frozenset[str]
    test_can_ids: frozenset[str]

    def is_leak_free(self) -> bool:
        """True iff no can_id appears in more than one split. The invariant."""
        t, v, s = self.train_can_ids, self.val_can_ids, self.test_can_ids
        return not (t & v) and not (t & s) and not (v & s)


def _condition_by_can(episodes: Iterable[Episode]) -> dict[str, object]:
    """Map each can_id to its (single) condition, rejecting inconsistent labeling.

    A physical can is one condition class. If two episodes disagree on a can's
    condition the dataset is malformed, so fail loudly rather than split silently.
    """
    cond: dict[str, object] = {}
    for ep in episodes:
        prev = cond.setdefault(ep.can_id, ep.condition)
        if prev != ep.condition:
            raise ValueError(
                f"can_id {ep.can_id!r} has inconsistent conditions ({prev} vs {ep.condition})"
            )
    return cond


def split_by_can(
    episodes: Iterable[Episode],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    stratify_by_condition: bool = True,
    seed: int = 0,
) -> Split:
    """Partition cans (not episodes) into train/val/test. A can lands in exactly one split.

    Within each stratum the can_ids are sorted (a stable base order) then shuffled with a
    seeded RNG, so the result is deterministic and leak-free by construction.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    cond = _condition_by_can(episodes)
    rng = random.Random(seed)

    if stratify_by_condition:
        groups: dict[object, list[str]] = defaultdict(list)
        for can_id, c in cond.items():
            groups[c].append(can_id)
        buckets = [groups[c] for c in sorted(groups, key=lambda c: c.value)]
    else:
        buckets = [list(cond.keys())]

    train: set[str] = set()
    val: set[str] = set()
    test: set[str] = set()
    for cans in buckets:
        cans = sorted(cans)  # stable base order before the seeded shuffle
        rng.shuffle(cans)
        n = len(cans)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.update(cans[:n_train])
        val.update(cans[n_train : n_train + n_val])
        test.update(cans[n_train + n_val :])  # remainder to test

    return Split(frozenset(train), frozenset(val), frozenset(test))


def assigned_split(split: Split, episode: Episode) -> str:
    """Return which split an episode belongs to, by its can_id."""
    cid = episode.can_id
    if cid in split.train_can_ids:
        return "train"
    if cid in split.val_can_ids:
        return "val"
    if cid in split.test_can_ids:
        return "test"
    raise KeyError(f"can_id {cid!r} is not assigned to any split")
