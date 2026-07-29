"""Replan arm (competence tier: outside-but-plannable). Retreat, recompute a reachable presentation
plan, then re-approach and present. Recovers failures that are outside the policy's competence but
still geometrically plannable (an off-nominal settled pose, no reachable presentation from the current
grasp)."""

from __future__ import annotations

from recovery.arms.base import ArmOutcome, _BaseArm
from recovery.backend import RecoveryBackend
from recovery.metric.recovery_regret import RecoveryArm


class ReplanArm(_BaseArm):
    _arm = RecoveryArm.REPLAN

    def __init__(self, retreat_height_m: float = 0.12) -> None:
        self.retreat_height_m = retreat_height_m

    def execute(self, backend: RecoveryBackend) -> ArmOutcome:
        # replan() reports whether a reachable plan was found; the attempt still re-approaches and
        # presents (a live sim can fail to find one, in which case the present simply does not recover).
        return self._run(backend, [
            lambda: backend.retreat(self.retreat_height_m),
            lambda: backend.replan(),
            lambda: backend.reapproach(),
            lambda: backend.present(),
        ])
