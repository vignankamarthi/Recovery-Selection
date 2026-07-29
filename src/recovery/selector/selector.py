"""Cost-sensitive recovery selector with a Lagrangian budget on ask-human (Part 2, step 2.7).

Given a failure's features (its competence tier and severity), the selector picks one of the four arms.
It is a learned, cost-aware chooser, not a fixed tier->arm lookup: it estimates each arm's expected
total cost from the training grid and picks the cheapest, with an explicit Lagrangian penalty
`lambda` added to the ask-human arm so its use stays under an intervention budget. The competence tier
still sets the DEFAULT arm (via the tier->arm map); the selector deviates from it under cost and budget.

FLAGGED research decisions:
  - Architecture: a transparent tabular cost model. For each (competence tier, severity bin) it stores
    the mean realized total cost of each arm on the training grid, then selects argmin over arms of
    predicted_cost + lambda*[arm is ask-human]. Chosen over a torch model because (a) the fence forbids
    torch in `recovery`, (b) it is interpretable and calibratable, and (c) with ~400-600 labeled
    instances (the proposal's target) a tabular model over a handful of feature cells is well-posed.
    A richer model (e.g. a small logistic/GBM cost regressor on the ACT competence signals) is a drop-in
    replacement behind the same `fit`/`select` interface once real signals exist.
  - Features: (proxy competence tier, coarse severity bin). These are what is actually observable at
    decision time (the ACT competence signals + task context), NOT the hidden ground-truth failure mode.
  - Human budget (default 0.30): ask-human is rationed to <=30% of failures. Ratify at GATE 4.

FENCE: imports `recovery` siblings + `schema`, never `harvest`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from recovery.grid.counterfactual import GridRow
from recovery.metric.recovery_regret import (
    ARM_COST_PROFILE,
    CostWeights,
    RecoveryArm,
    recovery_regret,
    total_cost,
)
from schema.episode import CompetenceTier


@dataclass(frozen=True)
class SelectorFeatures:
    """The decision-time features: the proxy competence tier and the failure severity."""

    competence_tier: CompetenceTier
    severity: float

    def key(self) -> tuple:
        # Coarse severity bin so each feature cell has enough training instances.
        sev_bin = "hi" if self.severity >= 0.5 else "lo"
        return (self.competence_tier, sev_bin)


@dataclass(frozen=True)
class SelectorReport:
    """Held-out summary of the calibrated selector over a set of (features, grid-row) samples."""

    mean_cost: float
    mean_regret: float
    ask_human_fraction: float
    lambda_: float


class CostSensitiveSelector:
    def __init__(self, weights: CostWeights, human_budget: float = 0.30) -> None:
        self.weights = weights
        self.human_budget = human_budget
        self._cost: dict[tuple, dict[RecoveryArm, float]] = {}
        self._global: dict[RecoveryArm, float] = {}
        self._lambda: float = 0.0

    # --- training ---
    def fit(self, samples: Sequence[tuple[SelectorFeatures, GridRow]]) -> "CostSensitiveSelector":
        """Learn E[total cost | feature cell, arm] from the grid, and a global fallback per arm."""
        sums: dict[tuple, dict[RecoveryArm, list]] = defaultdict(lambda: {a: [0.0, 0] for a in RecoveryArm})
        gsum: dict[RecoveryArm, list] = {a: [0.0, 0] for a in RecoveryArm}
        for feats, row in samples:
            k = feats.key()
            for arm in RecoveryArm:
                c = row.arm_total(arm)
                sums[k][arm][0] += c
                sums[k][arm][1] += 1
                gsum[arm][0] += c
                gsum[arm][1] += 1
        self._cost = {
            k: {a: (v[a][0] / v[a][1] if v[a][1] else _profile_cost(a, self.weights)) for a in RecoveryArm}
            for k, v in sums.items()
        }
        self._global = {
            a: (gsum[a][0] / gsum[a][1] if gsum[a][1] else _profile_cost(a, self.weights)) for a in RecoveryArm
        }
        return self

    def _predicted(self, features: SelectorFeatures) -> dict[RecoveryArm, float]:
        return self._cost.get(features.key(), self._global)

    # --- selection ---
    def select(self, features: SelectorFeatures, lam: Optional[float] = None) -> RecoveryArm:
        """Pick argmin over arms of predicted cost + lambda on the ask-human arm."""
        penalty = self._lambda if lam is None else lam
        pred = self._predicted(features)
        adjusted = {a: pred[a] + (penalty if a is RecoveryArm.ASK_HUMAN else 0.0) for a in RecoveryArm}
        # Deterministic tie-break along the escalation ladder (cheapest-first ordering).
        order = {a: i for i, a in enumerate(RecoveryArm)}
        return min(RecoveryArm, key=lambda a: (adjusted[a], order[a]))

    # --- Lagrangian calibration ---
    def ask_human_fraction(self, samples: Sequence[tuple[SelectorFeatures, GridRow]], lam: Optional[float] = None) -> float:
        if not samples:
            return 0.0
        n = sum(1 for feats, _ in samples if self.select(feats, lam) is RecoveryArm.ASK_HUMAN)
        return n / len(samples)

    def calibrate(self, samples: Sequence[tuple[SelectorFeatures, GridRow]]) -> float:
        """Find the smallest lambda whose ask-human fraction meets the budget (bisection; ask-human use
        is monotone non-increasing in lambda). Sets and returns `self.lambda_`."""
        if self.ask_human_fraction(samples, 0.0) <= self.human_budget:
            self._lambda = 0.0
            return 0.0
        lo, hi = 0.0, 1.0
        while self.ask_human_fraction(samples, hi) > self.human_budget:
            hi *= 2.0
            if hi > 1e12:                                # ask-human is mandatory beyond the budget
                break
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if self.ask_human_fraction(samples, mid) > self.human_budget:
                lo = mid
            else:
                hi = mid
        self._lambda = hi
        return hi

    @property
    def lambda_(self) -> float:
        return self._lambda

    # --- evaluation ---
    def evaluate(self, samples: Sequence[tuple[SelectorFeatures, GridRow]]) -> SelectorReport:
        """Realized mean cost + mean recovery-regret of the selector's choices vs the per-failure oracle."""
        if not samples:
            return SelectorReport(0.0, 0.0, 0.0, self._lambda)
        costs, regrets, human = [], [], 0
        for feats, row in samples:
            arm = self.select(feats)
            realized = row.arm_total(arm)
            costs.append(realized)
            regrets.append(recovery_regret(realized, row.oracle_cost))
            if arm is RecoveryArm.ASK_HUMAN:
                human += 1
        return SelectorReport(
            mean_cost=sum(costs) / len(costs),
            mean_regret=sum(regrets) / len(regrets),
            ask_human_fraction=human / len(samples),
            lambda_=self._lambda,
        )


def _profile_cost(arm: RecoveryArm, weights: CostWeights) -> float:
    """Fallback predicted cost for an unseen feature cell: the nominal arm-cost profile, scalarized."""
    return total_cost(ARM_COST_PROFILE[arm], recovered=True, weights=weights)
