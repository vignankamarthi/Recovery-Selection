"""Retry arm (competence tier: in-region). Re-execute the base policy from a slightly perturbed
in-distribution state, which recovers the transient-contact failures a deterministic re-run would only
reproduce. The cheapest arm."""

from __future__ import annotations

from recovery.arms.base import ArmOutcome, _BaseArm
from recovery.backend import RecoveryBackend
from recovery.metric.recovery_regret import RecoveryArm


class RetryArm(_BaseArm):
    _arm = RecoveryArm.RETRY

    def __init__(self, perturb_scale: float = 0.05) -> None:
        self.perturb_scale = perturb_scale

    def execute(self, backend: RecoveryBackend) -> ArmOutcome:
        return self._run(backend, [
            lambda: backend.perturb(self.perturb_scale),
            lambda: backend.present(),
        ])
