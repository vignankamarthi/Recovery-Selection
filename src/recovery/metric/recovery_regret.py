"""Recovery-regret metric (Part 2, our contribution).

Replaces binary recovery-success. For each injected failure, the counterfactual grid
replays all five recovery arms and records each arm's cost; the per-failure oracle is
the min-cost arm. recovery-regret = realized cost - oracle cost.

FENCE: this module imports `schema` only, never `harvest` internals. Build only after
Part 1 is complete (ANTIPATTERN 1). Stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RecoveryArm(str, Enum):
    # Four arms, escalation ladder. ask-human is the terminal safety fallback.
    # A control-invariant safe set is a hard floor beneath the arms (a low-level
    # safety halt), not a selectable arm, so there is no separate "abort" arm.
    RETRY = "retry"
    REWIND = "rewind"          # retreat and re-approach
    REPLAN = "replan"
    ASK_HUMAN = "ask_human"    # terminal safety fallback


@dataclass(frozen=True)
class ArmCost:
    """Multi-objective cost of running one arm on one failure."""

    time_s: float
    human_effort: float
    safety_violations: int

    def scalarize(self, weights: "CostWeights") -> float:
        """Collapse to a scalar cost under explicit weights. Stub."""
        raise NotImplementedError


@dataclass(frozen=True)
class CostWeights:
    time: float
    human_effort: float
    safety: float


def recovery_regret(realized: ArmCost, oracle: ArmCost, weights: CostWeights) -> float:
    """realized scalar cost minus oracle scalar cost (>= 0). Stub."""
    raise NotImplementedError
