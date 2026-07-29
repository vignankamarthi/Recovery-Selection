"""Counterfactual reset-and-replay grid + per-failure oracle (Part 2, step 2.5).

For each injected failure, snapshot the failed state, then replay ALL FOUR arms from that identical
state (restore -> execute -> record cost + whether it recovered). The per-failure oracle is the arm
with the least total cost (attempt cost, plus the unrecovered penalty if it failed). recovery-regret
of any arm is its total cost minus the oracle's, floored at 0.

This is the ORCHESTRATION logic only. It depends on the abstract `RecoveryBackend` (never a sim), and
a `backend_factory` builds one backend per failure. The reference model runs it with no sim; the sim
harness (outside the fence) runs the SAME grid on a real `SimWorld`.

FENCE: imports `recovery` siblings + `schema`-nothing, never `harvest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from recovery.arms import ALL_ARMS
from recovery.arms.base import Arm, ArmOutcome
from recovery.backend import RecoveryBackend
from recovery.failures.injection import InjectedFailure
from recovery.metric.recovery_regret import (
    CostWeights,
    RecoveryArm,
    recovery_regret,
    total_cost,
)

BackendFactory = Callable[[InjectedFailure], RecoveryBackend]


@dataclass
class GridRow:
    """One failure replayed across all arms: the per-arm outcomes, the oracle arm, and its total cost."""

    failure: InjectedFailure
    outcomes: dict[RecoveryArm, ArmOutcome]
    weights: CostWeights

    def arm_total(self, arm: RecoveryArm) -> float:
        out = self.outcomes[arm]
        return total_cost(out.cost, out.recovered, self.weights)

    @property
    def oracle_arm(self) -> RecoveryArm:
        return min(self.outcomes, key=self.arm_total)

    @property
    def oracle_cost(self) -> float:
        return self.arm_total(self.oracle_arm)

    def regret_of(self, arm: RecoveryArm) -> float:
        return recovery_regret(self.arm_total(arm), self.oracle_cost)


class CounterfactualGrid:
    """Runs the reset-and-replay grid and reports the recovery-regret aggregates."""

    def __init__(self, weights: CostWeights, arms: Sequence[Arm] = tuple(ALL_ARMS)) -> None:
        self.weights = weights
        self.arms = list(arms)

    def evaluate(self, backend_factory: BackendFactory, failures: Sequence[InjectedFailure]) -> list[GridRow]:
        rows: list[GridRow] = []
        for failure in failures:
            backend = backend_factory(failure)
            snap = backend.snapshot()
            outcomes: dict[RecoveryArm, ArmOutcome] = {}
            for arm in self.arms:
                backend.restore(snap)                 # replay every arm from the SAME failure state
                outcomes[arm.arm] = arm.execute(backend)
            rows.append(GridRow(failure=failure, outcomes=outcomes, weights=self.weights))
        return rows

    # --- aggregates (the 2.6 dry-run reads these) ---
    def mean_oracle_cost(self, rows: Sequence[GridRow]) -> float:
        return _mean(row.oracle_cost for row in rows)

    def fixed_arm_mean_cost(self, rows: Sequence[GridRow], arm: RecoveryArm) -> float:
        """The mean total cost of ALWAYS using one fixed arm (the single-mechanism baseline)."""
        return _mean(row.arm_total(arm) for row in rows)

    def best_fixed_arm(self, rows: Sequence[GridRow]) -> tuple[RecoveryArm, float]:
        """The single fixed arm with the lowest mean total cost, and that cost."""
        means = {arm: self.fixed_arm_mean_cost(rows, arm) for arm in RecoveryArm}
        best = min(means, key=means.get)
        return best, means[best]

    def oracle_improvement(self, rows: Sequence[GridRow]) -> float:
        """Fractional cost reduction of the per-failure oracle over the best fixed arm, in [0, 1].
        The proposal's make-or-break asks this to clear ~15% (on real failures, at GATE 4)."""
        _, best_fixed_mean = self.best_fixed_arm(rows)
        if best_fixed_mean <= 0:
            return 0.0
        return (best_fixed_mean - self.mean_oracle_cost(rows)) / best_fixed_mean

    def mean_fixed_arm_regret(self, rows: Sequence[GridRow], arm: RecoveryArm) -> float:
        return _mean(row.regret_of(arm) for row in rows)


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
