"""Competence signals (Part 2, our contribution).

Types a failure by the current state's relation to the frozen policy's competence
region, not by a semantic error label. Three signals:
  - latent-state density: how familiar the state is in the policy's representation;
  - action-head ensemble disagreement: how much a small ensemble disagrees on the act;
  - control-invariant safe set: a hard safety floor / anchor.
The state's position implies the recovery arm (in-region -> retry; boundary -> rewind;
outside-but-plannable -> replan; outside-and-risky -> ask-human).

FENCE: imports `schema` only, never `harvest`. Build only after Part 1 (ANTIPATTERN 1).
The base policy (pi0-FAST vs OpenVLA, inference/LoRA only) is an open decision. Stub.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetenceSignals:
    latent_density: float
    ensemble_disagreement: float
    in_safe_set: bool


class CompetenceModel:
    """Reads competence signals off a frozen base policy. Backbone stays frozen
    (inference / LoRA only; no policy training). Stub."""

    def signals(self, observation: object) -> CompetenceSignals:
        """Compute the three competence signals for a state. Stub."""
        raise NotImplementedError

    def classify_failure(self, signals: CompetenceSignals) -> str:
        """Map signals to a competence-grounded failure type. Stub."""
        raise NotImplementedError
