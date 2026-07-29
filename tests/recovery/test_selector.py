"""Cost-sensitive selector with a Lagrangian budget on the ask-human arm (2.7). Trains on the grid,
picks the arm minimizing predicted cost plus a Lagrangian penalty on ask-human, and calibrates the
multiplier to hold the human-intervention budget. Fence: imports `recovery` + `schema` only."""

import pytest

from recovery.arms import ALL_ARMS
from recovery.backend import ScriptedRecoveryBackend
from recovery.failures.injection import failure_from_episode, generate_failures, mode_for_tier
from recovery.grid.counterfactual import CounterfactualGrid
from recovery.metric.recovery_regret import DEFAULT_COST_WEIGHTS, RecoveryArm
from recovery.selector.selector import CostSensitiveSelector, SelectorFeatures
from schema.episode import CompetenceTier, ConditionClass, Episode


def _samples():
    """Build (features, grid-row) training pairs across all four tiers (many failures per tier)."""
    grid = CounterfactualGrid(DEFAULT_COST_WEIGHTS)
    episodes, feats = [], []
    tiers = list(CompetenceTier)
    for i in range(80):
        tier = tiers[i % len(tiers)]
        ep = Episode(f"e{i}", f"c{i}", ConditionClass.NOMINAL)
        ep.metadata["competence_tier"] = tier.value
        episodes.append(ep)
    failures = [failure_from_episode(ep, seed=i) for i, ep in enumerate(episodes)]
    rows = grid.evaluate(lambda f: ScriptedRecoveryBackend(f), failures)
    samples = [
        (SelectorFeatures(competence_tier=CompetenceTier(ep.metadata["competence_tier"]), severity=f.severity), row)
        for ep, f, row in zip(episodes, failures, rows)
    ]
    return samples


def test_selector_learns_to_beat_the_best_fixed_arm():
    samples = _samples()
    grid = CounterfactualGrid(DEFAULT_COST_WEIGHTS)
    rows = [r for _, r in samples]
    sel = CostSensitiveSelector(DEFAULT_COST_WEIGHTS, human_budget=1.0).fit(samples)
    report = sel.evaluate(samples)
    _, best_fixed_mean = grid.best_fixed_arm(rows)
    # A learned per-failure selector should cost less than the single best fixed arm.
    assert report.mean_cost < best_fixed_mean
    assert report.mean_regret >= 0.0


def test_unconstrained_selector_routes_unsafe_states_to_ask_human():
    samples = _samples()
    sel = CostSensitiveSelector(DEFAULT_COST_WEIGHTS, human_budget=1.0).fit(samples)
    # An outside-risky (unsafe) failure has no safe autonomous recovery, so the selector asks a human.
    risky = SelectorFeatures(competence_tier=CompetenceTier.OUTSIDE_RISKY, severity=0.95)
    assert sel.select(risky) is RecoveryArm.ASK_HUMAN


def test_lagrangian_penalty_monotonically_reduces_ask_human_usage():
    samples = _samples()
    sel = CostSensitiveSelector(DEFAULT_COST_WEIGHTS).fit(samples)
    frac_no_penalty = sel.ask_human_fraction(samples, lam=0.0)
    frac_big_penalty = sel.ask_human_fraction(samples, lam=5000.0)
    assert frac_big_penalty <= frac_no_penalty
    assert frac_big_penalty == 0.0                 # a large enough penalty drives ask-human to zero


def test_calibration_holds_the_human_budget():
    samples = _samples()
    budget = 0.10
    sel = CostSensitiveSelector(DEFAULT_COST_WEIGHTS, human_budget=budget).fit(samples)
    lam = sel.calibrate(samples)
    assert lam >= 0.0
    assert sel.ask_human_fraction(samples) <= budget + 1e-9
    # The calibrated multiplier is what `select` uses by default.
    assert sel.lambda_ == pytest.approx(lam)
