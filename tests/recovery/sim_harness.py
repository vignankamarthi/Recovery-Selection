"""The fence-crossing sim harness (OUTSIDE `src/recovery/`, on purpose).

`src/recovery/` is schema-only and never imports `SimWorld`. But the counterfactual grid must be able
to re-execute the four arms in a world with snapshot/restore, and we want to run it on the REAL sim in a
test. This module is the ONE place that wiring lives: `SimRecoveryBackend` adapts a `harvest.sim.SimWorld`
to the abstract `recovery.backend.RecoveryBackend` Protocol, so the arms (which only know the Protocol)
drive the real MuJoCo world. It imports BOTH `harvest` and `recovery`, which is exactly why it lives in
`tests/recovery/` and not under the fence.

Scope note: the sim is a FINISHED SMOKE TEST (Vignan's call, 2026-07-22). This harness reuses the
existing reorient pipeline to make the arms genuinely drive `SimWorld`; it deliberately does NOT invest
in new sim realism or tune the arms for a favorable recovery-regret number. Its job is to prove the
grid runs on the real sim across the fence. The reported directional dry-run number comes from the
deterministic data-level backend, never from this harness.
"""

from __future__ import annotations

import random

import numpy as np

from harvest.sim.reorient import (
    LABEL_VISIBLE_PX,
    UP_THRESH,
    Weld,
    _glide,
    _grasp_head,
    _overhead_px,
    _plan,
    _slip_roll,
)
from harvest.sim.world import SimWorld
from harvest.sim.scene import can_seed_from_id
from recovery.failures.injection import InjectedFailure

# Nominal per-primitive durations (seconds). Cost-time is a MODELED quantity here, not sim physics
# time: `_glide` restores earlier full-state snapshots, which rewinds `data.time`, so an accumulator is
# the robust clock (on hardware this is measured wall time). Matches the scripted backend's durations.
_DURATION = {"perturb": 4.0, "retreat": 5.0, "reapproach": 5.0, "replan": 12.0, "present": 4.0, "request_human": 5.0}


class SimRecoveryBackend:
    """A `RecoveryBackend` backed by a real `SimWorld`. On construction it drives the standard
    grasp -> reorient -> present pipeline and then applies the injected slip to reach a FAILED presented
    state; `snapshot`/`restore` then let each arm replay from that identical state. The arm primitives
    are real sim motions (weld-assisted, reusing the reorient internals)."""

    def __init__(self, world: SimWorld, failure: InjectedFailure, severity: float | None = None) -> None:
        self.world = world
        self.failure = failure
        sev = float(failure.severity if severity is None else severity)

        _grasp_head(world)
        self._offset = world.pinch_rotation() @ world.can_orientation().T
        plan = _plan(world, self._offset)
        self._present_snap = plan[0] if plan is not None else None
        if self._present_snap is not None:
            _glide(world, self._present_snap)              # good label-up presentation
        _slip_roll(world, sev)                             # ...then slip it off (the failure)

        self._failure_snap = world.snapshot()
        self._weld = Weld(world)
        self._elapsed = 0.0
        self._human = False

    # --- core ---
    def snapshot(self) -> object:
        return self.world.snapshot()

    def restore(self, snap: object) -> None:
        self.world.restore(np.asarray(snap))
        self._weld = Weld(self.world)                      # re-capture the can-in-hand offset
        self._elapsed = 0.0
        self._human = False

    def step(self, n: int = 1) -> None:
        self.world.step(n)

    # --- arm-relevant control (real sim motions, weld-assisted) ---
    def perturb(self, scale: float, seed: int = 0) -> None:
        rng = random.Random(seed)
        jitter = np.array([rng.uniform(-scale, scale) for _ in range(3)]) * 0.05
        self.world.move_pinch_to(self.world.pinch_position() + jitter, on_step=self._weld.follow)
        self._elapsed += _DURATION["perturb"]

    def retreat(self, height_m: float = 0.10) -> None:
        self.world.move_pinch_to(self.world.pinch_position() + np.array([0, 0, height_m]),
                                 on_step=self._weld.follow)
        self._elapsed += _DURATION["retreat"]

    def reapproach(self) -> None:
        from harvest.sim.reorient import HOVER_Z, INSPECT_XY
        self.world.move_pinch_to(np.array([INSPECT_XY[0], INSPECT_XY[1], HOVER_Z]),
                                 on_step=self._weld.follow)
        self._elapsed += _DURATION["reapproach"]

    def replan(self) -> bool:
        plan = _plan(self.world, self._offset)
        self._elapsed += _DURATION["replan"]
        if plan is not None:
            self._present_snap = plan[0]
            return True
        return False

    def present(self) -> None:
        if self._present_snap is not None:
            _glide(self.world, self._present_snap, on_step=self._weld.follow)
        self._elapsed += _DURATION["present"]

    def request_human(self) -> None:
        self._human = True
        self._elapsed += _DURATION["request_human"]

    # --- reads ---
    def task_success(self) -> bool:
        if self._human:
            return True                                    # a human resolves the terminal fallback
        px = _overhead_px(self.world)
        _, normal = self.world.can_label_pose()
        return bool(px is not None and px >= LABEL_VISIBLE_PX and float(normal[2]) > UP_THRESH)

    def is_safe(self) -> bool:
        # A minimal control-invariant safe-set stand-in: the can has not fallen off the workspace and
        # the joints are finite. (On hardware this is a real safe-set membership check.)
        can_z = float(self.world.can_position()[2])
        joints = self.world.proprioception()
        return bool(can_z > 0.0 and np.all(np.isfinite(joints)))

    def elapsed_s(self) -> float:
        return self._elapsed


def make_sim_backend_factory():
    """A grid `backend_factory`: build a fresh `SimWorld` + `SimRecoveryBackend` per injected failure."""
    def factory(failure: InjectedFailure) -> SimRecoveryBackend:
        seed = can_seed_from_id(failure.condition.value + failure.failure_id)
        world = SimWorld(condition=failure.condition, can_seed=seed,
                         can_quat=(np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0))
        return SimRecoveryBackend(world, failure)
    return factory
