"""Failure-injection API (Part 2, step 2.3).

The proposal injects 100-150 scripted failures across the can-manipulation tasks, on top of the
organic condition/orientation failures the base policy already makes, so the recovery grid has a
controlled, repeatable failure population (with a known mode and severity) to replay the four arms on.

This module is backend-agnostic (schema-only): an `InjectedFailure` only DESCRIBES a failure. A
backend APPLIES it -- the data-level `ScriptedRecoveryBackend` (a stylized outcome model) for the fast
dry-run and tests, or the sim harness (which drives a real `SimWorld`, e.g. via the in-hand slip roll)
outside the fence. Keeping the description here means the same catalog runs on either backend.

FLAGGED research decision (the failure taxonomy): the five modes below are the scripted failure
"kinds" the recovery arms are differentiated on. They are chosen so that no single fixed arm handles
all of them (transient contact wants a cheap retry; an occluded label wants a rewind-and-re-approach;
an out-of-distribution pose wants a replan; a genuinely-unsafe state wants a human), which is the
whole premise of a learned selector. Severities span [0, 1]. Ratify the taxonomy + the mode->recovery
correspondence (encoded in the outcome model, `recovery.backend`) at GATE 4 against real failures.

FENCE: imports `schema` only, never `harvest`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from schema.episode import CompetenceTier, ConditionClass, Episode


class FailureMode(str, Enum):
    """The scripted failure kinds. Each is best recovered by a different arm (see `recovery.backend`)."""

    TRANSIENT_SLIP = "transient_slip"        # a momentary contact loss; a cheap retry usually fixes it
    POSE_PERTURBATION = "pose_perturbation"  # the can settled off-nominal; re-approach or replan
    LABEL_OCCLUSION = "label_occlusion"      # the label is presented but occluded; rewind and re-approach
    PLAN_FAILURE = "plan_failure"            # no reachable presentation from this grasp; replan
    UNSAFE_STATE = "unsafe_state"            # the state turned genuinely unsafe; hand off to a human


@dataclass(frozen=True)
class InjectedFailure:
    """One scripted failure: what kind, on what can-condition, how severe, and a seed so a backend can
    apply it reproducibly. `failure_id` is unique within a catalog."""

    failure_id: str
    mode: FailureMode
    condition: ConditionClass
    severity: float
    seed: int


def generate_failures(n: int = 120, seed: int = 0) -> list[InjectedFailure]:
    """A deterministic scripted catalog of `n` failures (default 120, in the proposal's 100-150 range),
    spread across all five modes and all five conditions with severities in [0, 1].

    Round-robin over (mode, condition) guarantees full coverage of both; a seeded RNG sets each
    severity and per-failure seed, so the catalog is reproducible and re-orderable but never trivial.
    """
    if not 100 <= n <= 150:
        raise ValueError("the proposal specifies 100-150 injected failures")
    rng = random.Random(seed)
    modes = list(FailureMode)
    conditions = list(ConditionClass)
    failures: list[InjectedFailure] = []
    for i in range(n):
        mode = modes[i % len(modes)]
        condition = conditions[(i // len(modes)) % len(conditions)]
        failures.append(
            InjectedFailure(
                failure_id=f"inj-{i:03d}",
                mode=mode,
                condition=condition,
                severity=round(rng.uniform(0.15, 1.0), 4),
                seed=rng.randint(0, 2**31 - 1),
            )
        )
    return failures


# ---------------------------------------------------------------------------
# Tier <-> scripted-failure glue (FLAGGED, part of the failure-injection design). It lets the recovery
# SMOKE TEST (2.2b) and the dry-run (2.6) turn a proxy-tagged episode into a scripted failure whose
# kind matches the tier's home arm. Tier and mode are therefore correlated (as they should be), not
# identical: severity still varies. This is deliberate smoke-test glue, not evidence.
# ---------------------------------------------------------------------------
_TIER_MODE: dict[CompetenceTier, FailureMode] = {
    CompetenceTier.IN_REGION: FailureMode.TRANSIENT_SLIP,       # -> retry
    CompetenceTier.BOUNDARY: FailureMode.LABEL_OCCLUSION,       # -> rewind
    CompetenceTier.OUTSIDE_PLANNABLE: FailureMode.PLAN_FAILURE, # -> replan
    CompetenceTier.OUTSIDE_RISKY: FailureMode.UNSAFE_STATE,     # -> ask-human
}
_TIER_SEVERITY: dict[CompetenceTier, float] = {
    CompetenceTier.IN_REGION: 0.25,
    CompetenceTier.BOUNDARY: 0.50,
    CompetenceTier.OUTSIDE_PLANNABLE: 0.70,
    CompetenceTier.OUTSIDE_RISKY: 0.95,
}


def mode_for_tier(tier: CompetenceTier) -> FailureMode:
    """The scripted failure mode whose home arm is the tier's default arm."""
    return _TIER_MODE[tier]


def failure_from_episode(episode: Episode, seed: int = 0) -> InjectedFailure:
    """Build a scripted failure from a proxy-tagged episode (reads `metadata['competence_tier']`).
    Raises `KeyError` if the episode was never tagged (step 1.10)."""
    tier = CompetenceTier(episode.metadata["competence_tier"])
    return InjectedFailure(
        failure_id=f"{episode.episode_id}-inj",
        mode=mode_for_tier(tier),
        condition=episode.condition,
        severity=_TIER_SEVERITY[tier],
        seed=seed,
    )
