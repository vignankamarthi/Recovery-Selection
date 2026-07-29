"""The counterfactual reset-and-replay grid + per-failure oracle + recovery-regret aggregates (2.5).
Fence: imports `recovery` + `schema` only."""

import pytest

from recovery.arms import ALL_ARMS
from recovery.backend import ScriptedRecoveryBackend
from recovery.failures.injection import FailureMode, InjectedFailure, generate_failures
from recovery.grid.counterfactual import CounterfactualGrid
from recovery.metric.recovery_regret import DEFAULT_COST_WEIGHTS, RecoveryArm
from schema.episode import ConditionClass


def _factory(f):
    return ScriptedRecoveryBackend(f)


def _one(mode):
    return [InjectedFailure("f", mode, ConditionClass.RUST, 0.5, 0)]


def test_grid_replays_every_arm_on_every_failure():
    grid = CounterfactualGrid(DEFAULT_COST_WEIGHTS)
    rows = grid.evaluate(_factory, _one(FailureMode.TRANSIENT_SLIP))
    assert len(rows) == 1
    assert set(rows[0].outcomes) == set(RecoveryArm)


def test_oracle_picks_the_cheapest_recovering_arm_per_failure():
    grid = CounterfactualGrid(DEFAULT_COST_WEIGHTS)
    # A transient slip: retry recovers it and is the cheapest, so it is the oracle.
    row = grid.evaluate(_factory, _one(FailureMode.TRANSIENT_SLIP))[0]
    assert row.oracle_arm is RecoveryArm.RETRY
    # An unsafe state: only the human recovers without a safety violation, so ask-human is the oracle.
    row_unsafe = grid.evaluate(_factory, _one(FailureMode.UNSAFE_STATE))[0]
    assert row_unsafe.oracle_arm is RecoveryArm.ASK_HUMAN
    # A plan failure: replan is the cheapest recovering arm.
    row_plan = grid.evaluate(_factory, _one(FailureMode.PLAN_FAILURE))[0]
    assert row_plan.oracle_arm is RecoveryArm.REPLAN


def test_oracle_has_zero_regret_and_a_wrong_fixed_arm_has_positive_regret():
    grid = CounterfactualGrid(DEFAULT_COST_WEIGHTS)
    row = grid.evaluate(_factory, _one(FailureMode.LABEL_OCCLUSION))[0]
    assert row.regret_of(row.oracle_arm) == 0.0
    # Always-retry cannot clear an occlusion, so it carries the unrecovered penalty -> positive regret.
    assert row.regret_of(RecoveryArm.RETRY) > 0.0


def test_oracle_beats_the_best_fixed_arm_across_the_injected_catalog():
    grid = CounterfactualGrid(DEFAULT_COST_WEIGHTS)
    rows = grid.evaluate(_factory, generate_failures(n=120, seed=0))
    oracle_mean = grid.mean_oracle_cost(rows)
    best_arm, best_fixed_mean = grid.best_fixed_arm(rows)
    assert best_arm in set(RecoveryArm)
    # Because different modes need different arms, no single fixed arm matches the per-failure oracle.
    assert oracle_mean < best_fixed_mean
    improvement = grid.oracle_improvement(rows)
    assert 0.0 < improvement <= 1.0
