"""Tests for by-can leak-free splits (Phase 1.8, HARVEST flag F6).

The load-bearing correction: episodes from the same physical can are correlated, so a
per-episode split leaks a can across train/test and inflates scores. Every can_id must
land in EXACTLY ONE split. These are pure-logic tests, no sim.
"""

from harvest.dataset.splits import Split, assigned_split, split_by_can
from schema.episode import ConditionClass, Episode, Outcome

_CONDS = list(ConditionClass)


def _episodes(cans_per_condition=10, episodes_per_can=3):
    """A synthetic episode set: distinct cans per condition, several episodes per can."""
    eps = []
    for c in _CONDS:
        for i in range(cans_per_condition):
            can_id = f"can-{c.value}-{i:03d}"
            for k in range(episodes_per_can):
                eps.append(Episode(f"{can_id}-ep{k}", can_id, c, outcome=Outcome.SUCCESS))
    return eps


def _all_can_ids(eps):
    return {e.can_id for e in eps}


def test_is_leak_free_detects_overlap():
    clean = Split(frozenset({"a", "b"}), frozenset({"c"}), frozenset({"d"}))
    assert clean.is_leak_free() is True
    leaky = Split(frozenset({"a", "b"}), frozenset({"b"}), frozenset({"d"}))
    assert leaky.is_leak_free() is False


def test_split_is_leak_free_and_covers_every_can():
    eps = _episodes()
    split = split_by_can(eps, seed=0)
    assert split.is_leak_free()
    union = split.train_can_ids | split.val_can_ids | split.test_can_ids
    assert union == _all_can_ids(eps)  # every can assigned, none invented


def test_no_episode_leaks_across_splits():
    # The whole point of F6: no can_id appears in two splits, checked at episode level.
    eps = _episodes()
    split = split_by_can(eps, seed=1)
    placements = {e.can_id: assigned_split(split, e) for e in eps}
    # every episode of a given can maps to the same single split
    for e in eps:
        assert placements[e.can_id] == assigned_split(split, e)


def test_stratified_keeps_every_condition_in_train():
    eps = _episodes(cans_per_condition=10)
    split = split_by_can(eps, stratify_by_condition=True, seed=0)
    cond_of = {e.can_id: e.condition for e in eps}
    train_conditions = {cond_of[cid] for cid in split.train_can_ids}
    assert train_conditions == set(_CONDS)  # all five classes present in train


def test_ratios_are_approximately_respected():
    eps = _episodes(cans_per_condition=20)  # 100 cans total
    split = split_by_can(eps, ratios=(0.7, 0.15, 0.15), seed=0)
    n = len(_all_can_ids(eps))
    assert abs(len(split.train_can_ids) - 0.70 * n) <= 5
    assert abs(len(split.val_can_ids) - 0.15 * n) <= 5
    assert abs(len(split.test_can_ids) - 0.15 * n) <= 5


def test_deterministic_under_seed():
    eps = _episodes()
    a = split_by_can(eps, seed=42)
    b = split_by_can(eps, seed=42)
    assert a == b
    c = split_by_can(eps, seed=7)
    assert (a.train_can_ids, a.test_can_ids) != (c.train_can_ids, c.test_can_ids)


def test_assigned_split_raises_on_unknown_can():
    eps = _episodes(cans_per_condition=3)
    split = split_by_can(eps, seed=0)
    stray = Episode("x", "can-does-not-exist", ConditionClass.NOMINAL)
    try:
        assigned_split(split, stray)
        assert False, "expected a lookup error for an unassigned can"
    except KeyError:
        pass
