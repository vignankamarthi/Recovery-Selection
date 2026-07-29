"""The abstract recovery backend (Part 2), the seam the four arms drive.

This is the one architectural subtlety of the recovery layer. The counterfactual grid (2.5) must
RE-EXECUTE each arm from an identical failure state, which needs a world with snapshot/restore. To
keep the fence, `recovery` does NOT import `SimWorld`; it depends only on this small `RecoveryBackend`
Protocol, which provides `snapshot`/`restore`/`step` plus the arm-relevant control vocabulary. Two
things satisfy it:

  - `ScriptedRecoveryBackend` (here): a schema-only, deterministic REFERENCE outcome model. It is not
    a physics sim; it encodes the FLAGGED mode->required-recovery hypothesis so the arms, the grid, and
    the selector are fully testable and the dry-run is reproducible with no MuJoCo. This mirrors how
    `harvest.sensors.MockSource` stands in for a real sensor.
  - the SIM harness `SimRecoveryBackend` (OUTSIDE the fence, in `tests/recovery/sim_harness.py`): the
    same Protocol implemented against a real `SimWorld`. It imports BOTH `recovery` and `harvest`, so
    it lives outside `src/recovery/`. That is the only place the arms-to-sim wiring exists.

The arm vocabulary mirrors `control/policy.py::ManipulationPolicy` driving a `RobotBackend`: an arm
composes these primitives, and the backend's model (or physics) decides the outcome.

FENCE: imports `recovery` siblings + `schema`-nothing (stdlib only), never `harvest`, never mujoco/torch.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from recovery.failures.injection import FailureMode, InjectedFailure


@runtime_checkable
class RecoveryBackend(Protocol):
    """The world an arm drives to attempt a recovery, with snapshot/restore for counterfactual replay.

    Core (a real `SimWorld` already satisfies these): `snapshot`, `restore`, `step`.
    Arm-relevant control (the sim harness adds these on top of `SimWorld`):
      - `perturb`  -- a small in-distribution nudge (the retry arm);
      - `retreat`/`reapproach` -- back the effector off and re-approach (rewind, replan);
      - `replan`   -- recompute a reachable presentation plan, returns whether one was found (replan);
      - `present`  -- execute the frozen base policy / plan to present the label;
      - `request_human` -- hand off to a human (the terminal ask-human arm).
    Reads: `task_success` (did the label read succeed), `is_safe` (control-invariant safe-set floor),
    `elapsed_s` (time the current attempt consumed; resets on `restore`).
    """

    def snapshot(self) -> object: ...

    def restore(self, snap: object) -> None: ...

    def step(self, n: int = 1) -> None: ...

    def perturb(self, scale: float, seed: int = 0) -> None: ...

    def retreat(self, height_m: float = 0.10) -> None: ...

    def reapproach(self) -> None: ...

    def replan(self) -> bool: ...

    def present(self) -> None: ...

    def request_human(self) -> None: ...

    def task_success(self) -> bool: ...

    def is_safe(self) -> bool: ...

    def elapsed_s(self) -> float: ...


# ---------------------------------------------------------------------------
# The reference outcome model (FLAGGED research decision: the mode -> required-recovery mapping).
#
# Recovery "level" of the escalation ladder: retry=1, rewind=2, replan=3, ask-human=4. A stronger
# recovery subsumes weaker ones (rewind also re-presents, replan also re-approaches, a human fixes
# anything), so a failure is recovered iff the applied level >= the mode's required level. The mapping
# below is the hypothesis a learned selector exploits and the dry-run validates the machinery on; it is
# NOT evidence. Ratify against real failures at GATE 4.
# ---------------------------------------------------------------------------
_LEVEL_RETRY, _LEVEL_REWIND, _LEVEL_REPLAN, _LEVEL_HUMAN = 1, 2, 3, 4

_REQUIRED_LEVEL: dict[FailureMode, int] = {
    FailureMode.TRANSIENT_SLIP: _LEVEL_RETRY,      # a cheap retry from a perturbed state clears it
    FailureMode.LABEL_OCCLUSION: _LEVEL_REWIND,    # retreat + re-approach to un-occlude the label
    FailureMode.POSE_PERTURBATION: _LEVEL_REPLAN,  # the settled pose is off-nominal; recompute the plan
    FailureMode.PLAN_FAILURE: _LEVEL_REPLAN,       # no reachable presentation from this grasp; replan
    FailureMode.UNSAFE_STATE: _LEVEL_HUMAN,        # only a human may resolve a genuinely-unsafe state
}

# Nominal primitive durations (seconds). The realized `elapsed_s` sums these; it is what the arms'
# time cost is read from (a live sim backend would read real time instead). FLAGGED default.
_DURATION = {
    "perturb": 4.0,
    "retreat": 5.0,
    "reapproach": 5.0,
    "replan": 12.0,
    "present": 4.0,
    "request_human": 5.0,
}


class ScriptedRecoveryBackend:
    """A deterministic, schema-only reference backend: it models WHICH recovery clears a given failure
    mode, without any physics. Used by the tests, the recovery smoke test (2.2b), and the data-level
    dry-run (2.6). It is a stylized outcome model, never evidence.

    An arm composes the primitive vocabulary; this backend tracks the highest recovery level applied
    and reports success when it meets the mode's required level. Acting autonomously in an UNSAFE_STATE
    failure trips the control-invariant safe set; deferring to a human does not.
    """

    def __init__(self, failure: InjectedFailure) -> None:
        self.failure = failure
        self._required = _REQUIRED_LEVEL[failure.mode]
        self._reset_attempt()

    def _reset_attempt(self) -> None:
        self._level = 0
        self._human = False
        self._elapsed = 0.0
        self._safety_tripped_count = 0

    # --- core ---
    def snapshot(self) -> object:
        # The failure state is fixed; a snapshot only needs to reset the per-attempt accumulators.
        return {"mode": self.failure.mode}

    def restore(self, snap: object) -> None:
        self._reset_attempt()

    def step(self, n: int = 1) -> None:
        pass

    # --- arm-relevant control (each may trip the safe set on an UNSAFE_STATE failure) ---
    def _autonomous(self, name: str, level: int) -> None:
        self._elapsed += _DURATION[name]
        self._level = max(self._level, level)
        if self.failure.mode is FailureMode.UNSAFE_STATE and not self._human:
            self._safety_tripped_count += 1     # acted autonomously while genuinely unsafe

    def perturb(self, scale: float, seed: int = 0) -> None:
        self._autonomous("perturb", _LEVEL_RETRY)

    def retreat(self, height_m: float = 0.10) -> None:
        self._autonomous("retreat", _LEVEL_REWIND)

    def reapproach(self) -> None:
        self._autonomous("reapproach", _LEVEL_REWIND)

    def replan(self) -> bool:
        self._autonomous("replan", _LEVEL_REPLAN)
        return True                              # a plan is always found in the reference model

    def present(self) -> None:
        # `present` alone carries no recovery level; it is the execution step every autonomous arm ends
        # on. It still counts as autonomous action (time + safe-set exposure).
        self._elapsed += _DURATION["present"]
        if self.failure.mode is FailureMode.UNSAFE_STATE and not self._human:
            self._safety_tripped_count += 1

    def request_human(self) -> None:
        self._human = True
        self._level = max(self._level, _LEVEL_HUMAN)
        self._elapsed += _DURATION["request_human"]

    # --- reads ---
    def task_success(self) -> bool:
        return self._level >= self._required

    def is_safe(self) -> bool:
        return self._safety_tripped_count == 0

    def safety_violations(self) -> int:
        """How many autonomous primitives ran while the state was genuinely unsafe (0 unless UNSAFE)."""
        return self._safety_tripped_count

    def elapsed_s(self) -> float:
        return self._elapsed
