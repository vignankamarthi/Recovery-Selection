"""Rewind arm (competence tier: boundary). Retreat the effector to an in-distribution waypoint and
re-approach, then present again. Recovers failures at the edge of competence (e.g. an occluded label)
where a bare retry from the failed state would not."""

from __future__ import annotations

from recovery.arms.base import ArmOutcome, _BaseArm
from recovery.backend import RecoveryBackend
from recovery.metric.recovery_regret import RecoveryArm


class RewindArm(_BaseArm):
    _arm = RecoveryArm.REWIND

    def __init__(self, retreat_height_m: float = 0.10) -> None:
        self.retreat_height_m = retreat_height_m

    def execute(self, backend: RecoveryBackend) -> ArmOutcome:
        return self._run(backend, [
            lambda: backend.retreat(self.retreat_height_m),
            lambda: backend.reapproach(),
            lambda: backend.present(),
        ])
