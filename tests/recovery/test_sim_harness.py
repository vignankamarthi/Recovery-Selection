"""Integration test: the counterfactual grid runs on the REAL sim across the fence.

This is the literal check the architecture asks for -- the arms (schema-only, Protocol-typed) drive a
real `SimWorld` through the `SimRecoveryBackend` adapter, and the grid replays them with snapshot/
restore. It asserts MECHANICAL properties only (the wiring runs, costs are produced, all four arms are
replayed), NOT a recovery-success rate: the sim is a finished smoke test and the arms are not tuned for
a favorable number here. Marked `slow` (physics)."""

import numpy as np
import pytest

from harvest.sim.world import SimWorld
from recovery.arms import ALL_ARMS, AskHumanArm, RetryArm
from recovery.arms.base import ArmOutcome
from recovery.backend import RecoveryBackend
from recovery.failures.injection import FailureMode, InjectedFailure
from recovery.grid.counterfactual import CounterfactualGrid
from recovery.metric.recovery_regret import DEFAULT_COST_WEIGHTS, RecoveryArm
from schema.episode import ConditionClass

from tests.recovery.sim_harness import SimRecoveryBackend, make_sim_backend_factory

LYING = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)


def _backend(mode=FailureMode.TRANSIENT_SLIP, severity=0.3):
    world = SimWorld(condition=ConditionClass.NOMINAL, can_seed=3, can_quat=LYING)
    failure = InjectedFailure("sim-f", mode, ConditionClass.NOMINAL, severity, 0)
    return SimRecoveryBackend(world, failure)


@pytest.mark.slow
def test_sim_backend_satisfies_the_recovery_backend_protocol():
    b = _backend()
    assert isinstance(b, RecoveryBackend)


@pytest.mark.slow
def test_snapshot_restore_round_trips_and_resets_elapsed_time():
    b = _backend()
    snap = b.snapshot()
    b.retreat(0.1)
    b.step(5)
    assert b.elapsed_s() >= 0.0
    b.restore(snap)
    assert b.elapsed_s() == 0.0            # a fresh attempt from the same failure state


@pytest.mark.slow
def test_arms_execute_on_the_real_sim_and_return_costed_outcomes():
    b = _backend()
    snap = b.snapshot()
    for arm in (RetryArm(), AskHumanArm()):
        b.restore(snap)
        out = arm.execute(b)
        assert isinstance(out, ArmOutcome)
        assert out.cost.time_s >= 0.0
        assert out.cost.safety_violations >= 0
    # ask-human always resolves the task and spends human effort.
    b.restore(snap)
    human = AskHumanArm().execute(b)
    assert human.recovered is True
    assert human.cost.human_effort > 0


@pytest.mark.slow
def test_the_grid_replays_all_four_arms_on_the_real_sim():
    grid = CounterfactualGrid(DEFAULT_COST_WEIGHTS)
    factory = make_sim_backend_factory()
    failures = [InjectedFailure("sim-0", FailureMode.TRANSIENT_SLIP, ConditionClass.NOMINAL, 0.3, 0)]
    rows = grid.evaluate(factory, failures)
    assert len(rows) == 1
    assert set(rows[0].outcomes) == set(RecoveryArm)
    # Every arm produced a finite total cost (the grid + metric ran end to end on the sim).
    for arm in RecoveryArm:
        assert np.isfinite(rows[0].arm_total(arm))
