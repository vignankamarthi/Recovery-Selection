"""Competence signals and the four competence tiers (Part 2, our contribution -- step 2.2).

Types a failure by the current state's relation to the frozen policy's competence region, not by a
semantic error label. Three signals:
  - latent-state density: how familiar the state is in the frozen policy's representation;
  - action-head ensemble disagreement: how much a small ensemble disagrees on the action;
  - control-invariant safe set: a hard safety floor (a boolean membership).
The state's position sorts it into one of four `CompetenceTier`s, each mapped to a recovery arm:
  in_region -> retry, boundary -> rewind, outside_plannable -> replan, outside_risky -> ask-human.
The safe set is a FLOOR beneath the tiers, not a tier: a genuinely-unsafe state escalates straight
to outside_risky (ask-human) no matter how familiar the latent looks.

Two model paths, mirroring the trainer pattern:
  - `ProxyCompetenceModel` reads the 1.10 proxy tier already written into `Episode.metadata`
    (schema-only, works NOW, drives the recovery smoke test 2.2b);
  - `ACTCompetenceModel` reads the real signals off a frozen `BasePolicy` (latent density + ensemble
    disagreement). It is torch-free itself: it consumes whatever the policy's `latent`/`action_ensemble`
    return, so it runs locally against `StubBasePolicy` and on AICR against `FrozenACTPolicy`.

FENCE: imports `schema` (for CompetenceTier/Episode) and `recovery` siblings only, never `harvest`.
The base policy (ACT, frozen; inference/LoRA only) stays frozen throughout: recovery trains only the
selector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from recovery.metric.recovery_regret import RecoveryArm
from recovery.policy.base_policy import BasePolicy, Observation
from schema.episode import CompetenceTier, Episode

# ---------------------------------------------------------------------------
# tier <-> arm map (the proposal's four-tier -> four-arm correspondence).
# ---------------------------------------------------------------------------
_TIER_ARM: dict[CompetenceTier, RecoveryArm] = {
    CompetenceTier.IN_REGION: RecoveryArm.RETRY,
    CompetenceTier.BOUNDARY: RecoveryArm.REWIND,
    CompetenceTier.OUTSIDE_PLANNABLE: RecoveryArm.REPLAN,
    CompetenceTier.OUTSIDE_RISKY: RecoveryArm.ASK_HUMAN,
}


def tier_to_arm(tier: CompetenceTier) -> RecoveryArm:
    """The default recovery arm for a competence tier (the selector may deviate under budget)."""
    return _TIER_ARM[tier]


@dataclass(frozen=True)
class CompetenceSignals:
    """The three competence signals read off the frozen policy at one state.

    `latent_density` and `ensemble_disagreement` are normalized to [0, 1] by convention (higher
    density = more familiar; higher disagreement = less certain). `in_safe_set` is the hard floor.
    """

    latent_density: float
    ensemble_disagreement: float
    in_safe_set: bool


@dataclass(frozen=True)
class CompetenceThresholds:
    """Bands that turn the continuous signals into a tier. FLAGGED research default; these are
    calibrated against the real ACT latent distribution on hardware (here they are sensible priors).

    The tier is decided on a single scalar `competence = latent_density - disagreement_penalty *
    ensemble_disagreement` in [~ -1, 1]: familiar + confident is high (in_region), unfamiliar or
    uncertain is low (outside_risky). `in_region_min`/`boundary_min`/`plannable_min` are the cut points.
    """

    in_region_min: float = 0.75
    boundary_min: float = 0.50
    plannable_min: float = 0.25
    disagreement_penalty: float = 0.5


def classify_signals(
    signals: CompetenceSignals, thresholds: Optional[CompetenceThresholds] = None
) -> CompetenceTier:
    """Map competence signals to a tier. The safe set is a hard floor: an unsafe state is
    outside_risky (-> ask-human) regardless of the latent signals."""
    if not signals.in_safe_set:
        return CompetenceTier.OUTSIDE_RISKY
    th = thresholds or CompetenceThresholds()
    competence = signals.latent_density - th.disagreement_penalty * signals.ensemble_disagreement
    if competence >= th.in_region_min:
        return CompetenceTier.IN_REGION
    if competence >= th.boundary_min:
        return CompetenceTier.BOUNDARY
    if competence >= th.plannable_min:
        return CompetenceTier.OUTSIDE_PLANNABLE
    return CompetenceTier.OUTSIDE_RISKY


@runtime_checkable
class CompetenceModel(Protocol):
    """Sorts a state (an observation, or a tagged episode) into a competence tier, hence a default arm."""

    def classify(self, observation: object) -> CompetenceTier: ...

    def default_arm(self, observation: object) -> RecoveryArm: ...


class ProxyCompetenceModel:
    """Reads the 1.10 PROXY tier already written into `Episode.metadata['competence_tier']` (the
    held-out realized-margin proxy, schema-only). This is the now-path that drives the recovery smoke
    test (2.2b) before the real ACT latent signals exist. `classify`/`default_arm` accept an `Episode`."""

    _KEY = "competence_tier"

    def tier(self, episode: Episode) -> CompetenceTier:
        return CompetenceTier(episode.metadata[self._KEY])

    def classify(self, observation: object) -> CompetenceTier:
        assert isinstance(observation, Episode), "ProxyCompetenceModel classifies tagged Episodes"
        return self.tier(observation)

    def default_arm(self, observation: object) -> RecoveryArm:
        return tier_to_arm(self.classify(observation))


def _mean_dist(z: Sequence[float], ref: Sequence[Sequence[float]]) -> float:
    """Mean Euclidean distance from `z` to a reference set of latents (stdlib-only, no numpy)."""
    if not ref:
        return 0.0
    total = 0.0
    for r in ref:
        total += math.sqrt(sum((a - b) ** 2 for a, b in zip(z, r)))
    return total / len(ref)


def _disagreement(ensemble: Sequence[Sequence[float]]) -> float:
    """Mean per-dimension standard deviation across the action ensemble, normalized to [0, 1]."""
    if len(ensemble) < 2:
        return 0.0
    dim = len(ensemble[0])
    stds = []
    for d in range(dim):
        col = [a[d] for a in ensemble]
        mu = sum(col) / len(col)
        var = sum((c - mu) ** 2 for c in col) / len(col)
        stds.append(math.sqrt(var))
    return min(1.0, sum(stds) / dim)


class ACTCompetenceModel:
    """Reads the real competence signals off a FROZEN `BasePolicy`.

    `fit(train_observations)` records the reference latent distribution (the training states the
    policy is competent on) and a density scale. `signals(obs)` then computes latent density (from the
    mean distance to that reference, squashed to [0, 1]), ensemble disagreement (spread across the
    policy's small action ensemble), and safe-set membership (from an optional `safe_set` predicate,
    default: always in-set). `classify(obs)` bands those into a tier.

    Torch-free by construction: it only calls the policy's `latent`/`action_ensemble`, so it runs
    locally on `StubBasePolicy` and on AICR on `FrozenACTPolicy` (which imports torch lazily). The
    backbone is never trained here (frozen).
    """

    def __init__(
        self,
        policy: BasePolicy,
        thresholds: Optional[CompetenceThresholds] = None,
        safe_set: Optional[Callable[[Observation], bool]] = None,
    ) -> None:
        self.policy = policy
        self.thresholds = thresholds or CompetenceThresholds()
        self.safe_set = safe_set
        self._ref: list[list[float]] = []
        self._scale: float = 1.0

    def fit(self, train_observations: Sequence[Observation]) -> "ACTCompetenceModel":
        self._ref = [list(self.policy.latent(o)) for o in train_observations]
        # Density scale = the typical in-distribution spread, so an in-distribution state reads ~1.
        if len(self._ref) >= 2:
            dists = [_mean_dist(z, self._ref) for z in self._ref]
            self._scale = max(1e-6, sum(dists) / len(dists))
        return self

    def signals(self, observation: Observation) -> CompetenceSignals:
        z = list(self.policy.latent(observation))
        d = _mean_dist(z, self._ref)
        density = math.exp(-d / self._scale) if self._ref else 1.0          # familiar -> ~1, far -> ~0
        disagreement = _disagreement(self.policy.action_ensemble(observation))
        in_safe = True if self.safe_set is None else bool(self.safe_set(observation))
        return CompetenceSignals(latent_density=density, ensemble_disagreement=disagreement, in_safe_set=in_safe)

    def classify(self, observation: Observation) -> CompetenceTier:
        return classify_signals(self.signals(observation), self.thresholds)

    def default_arm(self, observation: Observation) -> RecoveryArm:
        return tier_to_arm(self.classify(observation))
