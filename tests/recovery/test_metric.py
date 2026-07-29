"""Recovery-regret metric (2.5). Cost scalarization, the unrecovered penalty, and regret >= 0.

Fence: this test imports `recovery` only (never `harvest`), so it also exercises the fence.
"""

import pytest

from recovery.metric.recovery_regret import (
    ARM_COST_PROFILE,
    DEFAULT_COST_WEIGHTS,
    ArmCost,
    CostWeights,
    RecoveryArm,
    recovery_regret,
    total_cost,
)


def test_scalarize_is_the_weighted_sum():
    cost = ArmCost(time_s=10.0, human_effort=1.0, safety_violations=2)
    w = CostWeights(time=1.0, human_effort=30.0, safety=1000.0, unrecovered_penalty=100.0)
    assert cost.scalarize(w) == pytest.approx(10.0 * 1.0 + 1.0 * 30.0 + 2 * 1000.0)


def test_total_cost_adds_penalty_only_when_unrecovered():
    cost = ArmCost(time_s=8.0, human_effort=0.0, safety_violations=0)
    w = DEFAULT_COST_WEIGHTS
    recovered = total_cost(cost, recovered=True, weights=w)
    unrecovered = total_cost(cost, recovered=False, weights=w)
    assert recovered == pytest.approx(cost.scalarize(w))
    assert unrecovered == pytest.approx(cost.scalarize(w) + w.unrecovered_penalty)


def test_recovery_regret_is_realized_minus_oracle_floored_at_zero():
    # A cheap arm that recovered (oracle) vs an expensive realized choice.
    assert recovery_regret(realized_cost=35.0, oracle_cost=8.0) == pytest.approx(27.0)
    # The oracle can never be beaten, so regret is floored at 0 (never negative).
    assert recovery_regret(realized_cost=8.0, oracle_cost=8.0) == 0.0
    assert recovery_regret(realized_cost=5.0, oracle_cost=8.0) == 0.0


def test_default_cost_weights_encode_the_safety_ordering():
    # Safety-critical food inspection: a safety violation must dominate a human hand-off,
    # which must dominate raw robot time. This ordering is the FLAGGED research default.
    w = DEFAULT_COST_WEIGHTS
    assert w.safety > w.human_effort > w.time > 0
    assert w.unrecovered_penalty > 0


def test_arm_cost_profile_covers_all_four_arms_and_escalates_in_time():
    assert set(ARM_COST_PROFILE) == set(RecoveryArm)
    # The autonomous ladder gets more expensive in time as it escalates.
    assert (
        ARM_COST_PROFILE[RecoveryArm.RETRY].time_s
        < ARM_COST_PROFILE[RecoveryArm.REWIND].time_s
        < ARM_COST_PROFILE[RecoveryArm.REPLAN].time_s
    )
    # ask-human is the only arm that spends human effort (the budgeted resource).
    assert ARM_COST_PROFILE[RecoveryArm.ASK_HUMAN].human_effort > 0
    for arm in (RecoveryArm.RETRY, RecoveryArm.REWIND, RecoveryArm.REPLAN):
        assert ARM_COST_PROFILE[arm].human_effort == 0
