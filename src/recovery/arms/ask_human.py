"""Ask-human arm (competence tier: outside-and-risky). Hand off to a human, the terminal safety
fallback a person always resolves. It spends human-intervention effort (the budgeted resource the
Lagrangian selector rations) but never trips the control-invariant safe set, since it does not act
autonomously in a risky state."""

from __future__ import annotations

from recovery.arms.base import ArmOutcome, _BaseArm
from recovery.backend import RecoveryBackend
from recovery.metric.recovery_regret import RecoveryArm


class AskHumanArm(_BaseArm):
    _arm = RecoveryArm.ASK_HUMAN
    _human_effort = 1.0            # one full hand-off; the resource the budget rations

    def execute(self, backend: RecoveryBackend) -> ArmOutcome:
        return self._run(backend, [
            lambda: backend.request_human(),
        ])
