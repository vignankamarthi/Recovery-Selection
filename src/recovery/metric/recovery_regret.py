"""Recovery-regret metric (Part 2, our contribution -- step 2.5).

Replaces binary recovery-success. For each injected failure, the counterfactual grid
(`recovery.grid`) replays all four recovery arms and records each arm's cost. The per-failure
oracle is the min-cost arm. recovery-regret = realized (selected) cost - oracle cost, floored at 0.

Cost is multi-objective (time, human-intervention effort, safety violations), collapsed to a scalar
under explicit `CostWeights`. An arm that fails to recover the task leaves it failed, which costs the
`unrecovered_penalty` on top of what the attempt itself spent.

FENCE: this module imports `schema`-nothing (stdlib only), never `harvest`. Safe for both tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RecoveryArm(str, Enum):
    # Four arms, an escalation ladder. ask-human is the terminal safety fallback.
    # A control-invariant safe set is a hard floor beneath the arms (a low-level
    # safety halt), not a selectable arm, so there is no separate "abort" arm.
    RETRY = "retry"
    REWIND = "rewind"          # retreat and re-approach
    REPLAN = "replan"
    ASK_HUMAN = "ask_human"    # terminal safety fallback


@dataclass(frozen=True)
class ArmCost:
    """Multi-objective cost of running one arm on one failure.

    `time_s` is robot/wall time the attempt consumed, `human_effort` is normalized human-intervention
    effort (1.0 = one full hand-off), `safety_violations` counts control-invariant-safe-set breaches
    observed during the attempt.
    """

    time_s: float
    human_effort: float
    safety_violations: int

    def scalarize(self, weights: "CostWeights") -> float:
        """Collapse to a scalar cost under explicit weights (the attempt's own cost, no penalty)."""
        return (
            self.time_s * weights.time
            + self.human_effort * weights.human_effort
            + self.safety_violations * weights.safety
        )


@dataclass(frozen=True)
class CostWeights:
    """Weights collapsing the multi-objective cost to a scalar, plus the penalty for leaving the task
    unrecovered. FLAGGED research default lives in `DEFAULT_COST_WEIGHTS`; ratify at GATE 4."""

    time: float
    human_effort: float
    safety: float
    unrecovered_penalty: float


# ---------------------------------------------------------------------------
# FLAGGED research defaults (documented; ratify with the human at GATE 4).
#
# Cost weights. Units: time in seconds, human_effort in [0, 1] (one hand-off = 1.0),
# safety_violations an integer count. The ordering is what matters and is deliberate for a
# SAFETY-CRITICAL food-inspection task: one safety violation (e.g. mishandling a botulism-risk
# bulged can) must dominate a human hand-off, which must dominate raw robot time. A left-unrecovered
# task costs a fixed penalty above whatever the failed attempt already spent.
# ---------------------------------------------------------------------------
DEFAULT_COST_WEIGHTS = CostWeights(
    time=1.0,             # 1 unit of cost per second of robot time
    human_effort=30.0,    # a human hand-off is worth ~30 s of robot time (operator is the scarce resource)
    safety=1000.0,        # a safety violation is catastrophic; it dwarfs any time/human cost
    unrecovered_penalty=100.0,  # leaving the task failed costs far more than any single recovery attempt
)

# Nominal per-arm cost profile (the escalation ladder). time_s is the FLAGGED default duration of one
# attempt; a live backend refines it (and safety_violations) at execution. Only ask-human spends human
# effort, which is the budgeted resource the Lagrangian selector rations.
ARM_COST_PROFILE: dict[RecoveryArm, ArmCost] = {
    RecoveryArm.RETRY: ArmCost(time_s=8.0, human_effort=0.0, safety_violations=0),
    RecoveryArm.REWIND: ArmCost(time_s=15.0, human_effort=0.0, safety_violations=0),
    RecoveryArm.REPLAN: ArmCost(time_s=25.0, human_effort=0.0, safety_violations=0),
    RecoveryArm.ASK_HUMAN: ArmCost(time_s=5.0, human_effort=1.0, safety_violations=0),
}


def total_cost(cost: ArmCost, recovered: bool, weights: CostWeights) -> float:
    """The scalar cost the oracle and the selector compare on: the attempt's own scalarized cost,
    plus the unrecovered penalty when the arm failed to recover the task."""
    scalar = cost.scalarize(weights)
    return scalar if recovered else scalar + weights.unrecovered_penalty


def recovery_regret(realized_cost: float, oracle_cost: float) -> float:
    """Realized (selected-arm) scalar cost minus the per-failure oracle's scalar cost, floored at 0.

    Both inputs are already-scalarized totals (see `total_cost`). Regret is 0 exactly when the
    selector chose an arm as good as the oracle, and never negative (the oracle is a lower bound)."""
    return max(0.0, realized_cost - oracle_cost)
