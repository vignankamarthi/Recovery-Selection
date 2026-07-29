"""Scripted manipulation policies (the sim demonstration generator, Part 1).

Backend-agnostic. A policy drives any `RobotBackend`, so the same policy runs on `SimWorld`
now and a real-robot backend later. The scripted grasp is the frozen, imperfect base policy
whose failures the recovery layer consumes. On hardware the demonstrations come from
teleoperation, this scripted policy is the sim stand-in. MuJoCo-free by construction, the
backend owns the physics and the IK.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import numpy as np

from harvest.control.backend import GraspBackend, RobotBackend

OnStep = Optional[Callable[[], None]]


class ManipulationPolicy(Protocol):
    """Runs a manipulation on a backend, driving control and returning grasp success. Needs both
    control and scene ground truth, so it is typed against `GraspBackend` (the composition)."""

    def run(self, backend: GraspBackend, on_step: OnStep = None) -> bool: ...


class ScriptedGraspPolicy:
    """Orientation-aware top-down grasp.

    Reads the settled can pose. An upright can is rotationally symmetric from above, so any
    finger yaw works. A lying can needs the fingers across its short (diameter) axis, so the
    wrist roll is held at long-axis + 90 degrees while joints 1-6 reach. Returns grasp success
    from backend ground truth (F7, never the tactile stream).

    Two knobs let the sim demonstration reuse this exact open->approach->descend->close->settle
    sequence: `head_offset_m` grasps that far along a lying can's long axis (the head, not the
    middle), and `lift=False` skips the final lift (the weld-assisted reorient does not lift).
    """

    def __init__(self, settle: int = 40, top_settle: int = 20,
                 head_offset_m: float = 0.0, lift: bool = True, lift_height_m: float = 0.18) -> None:
        self.settle = settle
        self.top_settle = top_settle
        self.head_offset_m = head_offset_m
        self.lift = lift
        self.lift_height_m = lift_height_m

    def run(self, backend: GraspBackend, on_step: OnStep = None) -> bool:
        can = backend.can_position()
        if backend.can_is_upright():
            wrist, descend = None, 0.02
        else:
            axis = backend.can_long_axis()
            axis = axis / np.linalg.norm(axis)
            wrist = backend.aligned_wrist(float(np.arctan2(axis[1], axis[0])) + np.pi / 2.0)
            descend = 0.0
            can = can + self.head_offset_m * axis                    # grasp the head, not the middle

        backend.set_gripper(0.0)                                      # open
        backend.move_pinch_to(can + [0, 0, 0.12], wrist=wrist, on_step=on_step)   # approach
        backend.move_pinch_to(can + [0, 0, descend], wrist=wrist, on_step=on_step)  # descend
        backend.set_gripper(1.0)                                      # close
        self._settle(backend, self.settle, on_step)
        if self.lift:
            backend.move_pinch_to(can + [0, 0, self.lift_height_m], wrist=wrist, on_step=on_step)  # lift
            self._settle(backend, self.top_settle, on_step)
        return backend.grasp_success()

    @staticmethod
    def _settle(backend: RobotBackend, n: int, on_step: OnStep) -> None:
        for _ in range(n):
            backend.step(5)
            if on_step is not None:
                on_step()

# The in-hand reorient / present is sim-specific (weld + plan-then-execute on a hidden scratch
# copy), so it lives in `harvest.sim.reorient`, not here. This module stays backend-agnostic: the
# scripted grasp is the piece a real-robot backend could reuse; the presentation is the sim
# stand-in for teleoperated demos. The old overhead-tilt `ScriptedReorientPolicy` was retired
# when the weld pipeline replaced it (a rigid pad cannot hold a can through a reorient).
