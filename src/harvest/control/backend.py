"""The robot control backend interface (Part 1).

`RobotBackend` is the control + kinematic-read surface a manipulation policy needs, so the
policy is backend-agnostic. `SimWorld` satisfies it now, and a real-robot `RosBackend`
satisfies the same interface later, so a hardware backend drops in without touching the
policy. This is the swappable-interface rule applied to control, the counterpart to the
`SensorSource` protocol for sensing. It is a `typing.Protocol`, so a backend needs no base
class, it just provides the methods.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RobotBackend(Protocol):
    """Control + kinematic reads a scripted manipulation policy drives."""

    def reset(self) -> None: ...

    def step(self, n: int = 1) -> None: ...

    def set_gripper(self, closed_frac: float) -> None: ...

    def move_pinch_to(self, target, wrist: Optional[float] = None,
                      on_step: Optional[Callable[[], None]] = None) -> None: ...

    def move_pinch_pose(self, target_pos, target_rot, max_steps: int = 140,
                        on_step: Optional[Callable[[], None]] = None) -> None: ...

    def aligned_wrist(self, target_angle: float) -> float: ...

    def pinch_position(self) -> np.ndarray: ...

    def pinch_rotation(self) -> np.ndarray: ...

    def can_position(self) -> np.ndarray: ...

    def can_is_upright(self, tol_deg: float = 35.0) -> bool: ...

    def can_long_axis(self) -> np.ndarray: ...

    def can_label_pose(self) -> tuple[np.ndarray, np.ndarray]: ...

    def grasp_success(self) -> bool: ...

    def overhead_label_visibility(self) -> float: ...
