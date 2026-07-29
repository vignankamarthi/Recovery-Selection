"""The recovery-arm interface (Part 2, step 2.4).

An arm is a BACKEND-AGNOSTIC behavior: it composes the `RecoveryBackend` control vocabulary
(perturb / retreat / reapproach / replan / present / request_human) into one recovery attempt, exactly
as `control/policy.py::ManipulationPolicy` composes a `RobotBackend` into a grasp. The backend (the
schema-only reference model, or a real `SimWorld` via the harness) decides the outcome.

`execute` runs the attempt from the backend's CURRENT state (the grid restores the failure snapshot
before each arm), polling the control-invariant safe set after every primitive to count violations,
and returns an `ArmOutcome` carrying the realized `ArmCost` (time from the backend, human effort
intrinsic to the arm, safety violations observed) and whether the task was recovered.

FENCE: imports `recovery` siblings + `schema`-nothing, never `harvest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from recovery.backend import RecoveryBackend
from recovery.metric.recovery_regret import ArmCost, RecoveryArm


@dataclass(frozen=True)
class ArmOutcome:
    """The result of running one arm on one failure: which arm, its realized cost, did it recover."""

    arm: RecoveryArm
    cost: ArmCost
    recovered: bool


@runtime_checkable
class Arm(Protocol):
    """A recovery behavior. `arm` is its `RecoveryArm` identity; `execute` runs it on a backend."""

    @property
    def arm(self) -> RecoveryArm: ...

    def execute(self, backend: RecoveryBackend) -> ArmOutcome: ...


class _BaseArm:
    """Shared machinery: run a sequence of backend primitives, counting safe-set violations, then
    assemble the `ArmOutcome`. `human_effort` is intrinsic to the arm (only ask-human spends it)."""

    _arm: RecoveryArm = RecoveryArm.RETRY
    _human_effort: float = 0.0

    @property
    def arm(self) -> RecoveryArm:
        return self._arm

    def _run(self, backend: RecoveryBackend, primitives: list[Callable[[], object]]) -> ArmOutcome:
        violations = 0
        for prim in primitives:
            prim()
            if not backend.is_safe():
                violations += 1
        cost = ArmCost(
            time_s=backend.elapsed_s(),
            human_effort=self._human_effort,
            safety_violations=violations,
        )
        return ArmOutcome(arm=self._arm, cost=cost, recovered=backend.task_success())

    def execute(self, backend: RecoveryBackend) -> ArmOutcome:  # pragma: no cover - overridden
        raise NotImplementedError
