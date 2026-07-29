"""The four recovery arms (Part 2, step 2.4), one per competence tier, as backend-agnostic behaviors.
The fence holds: recovery imports schema only, never harvest internals.

`ALL_ARMS` is the escalation ladder in order (retry -> rewind -> replan -> ask-human), the set the
counterfactual grid replays and the selector chooses among.
"""

from recovery.arms.ask_human import AskHumanArm
from recovery.arms.base import Arm, ArmOutcome, _BaseArm
from recovery.arms.replan import ReplanArm
from recovery.arms.retry import RetryArm
from recovery.arms.rewind import RewindArm

ALL_ARMS: list[Arm] = [RetryArm(), RewindArm(), ReplanArm(), AskHumanArm()]

__all__ = [
    "Arm",
    "ArmOutcome",
    "RetryArm",
    "RewindArm",
    "ReplanArm",
    "AskHumanArm",
    "ALL_ARMS",
]
