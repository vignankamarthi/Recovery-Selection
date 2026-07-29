"""The robot control + scene-oracle interfaces (Part 1).

Two swappable seams, split so a hardware backend and a hardware perception module drop in
independently. This is the biggest hardware-swappability win: on the real robot, control comes
from a `RosBackend` while the scene ground truth comes from a separate perception stack, whereas
in sim the one `SimWorld` provides both.

`RobotBackend` is the generic control + proprioception surface (reset/step, gripper, IK moves,
pinch reads). `SceneOracle` is the task/scene ground-truth surface (can pose, uprightness, label
pose, grasp success, overhead label visibility). A scripted manipulation policy needs BOTH, so it
is typed against `GraspBackend` (the composition). All three are `typing.Protocol`s, so a backend
needs no base class, it just provides the methods. `SimWorld` satisfies all of them structurally.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class RobotBackend(Protocol):
    """Generic control + proprioception a scripted manipulation policy drives."""

    def reset(self) -> None: ...

    def step(self, n: int = 1) -> None: ...

    def set_gripper(self, closed_frac: float) -> None: ...

    def move_pinch_to(self, target: "np.ndarray | Sequence[float]", wrist: Optional[float] = None,
                      on_step: Optional[Callable[[], None]] = None) -> None: ...

    def move_pinch_pose(self, target_pos: "np.ndarray | Sequence[float]",
                        target_rot: np.ndarray, max_steps: int = 140,
                        on_step: Optional[Callable[[], None]] = None) -> None: ...

    def aligned_wrist(self, target_angle: float) -> float: ...

    def pinch_position(self) -> np.ndarray: ...

    def pinch_rotation(self) -> np.ndarray: ...


@runtime_checkable
class SceneOracle(Protocol):
    """Task/scene ground truth. In sim this is physics truth; on hardware a perception module."""

    def can_position(self) -> np.ndarray: ...

    def can_is_upright(self, tol_deg: float = 35.0) -> bool: ...

    def can_long_axis(self) -> np.ndarray: ...

    def can_label_pose(self) -> tuple[np.ndarray, np.ndarray]: ...

    def grasp_success(self) -> bool: ...

    def overhead_label_visibility(self) -> float: ...


@runtime_checkable
class GraspBackend(RobotBackend, SceneOracle, Protocol):
    """A backend that is both controllable AND provides scene ground truth. The scripted grasp
    policy needs both seams; `SimWorld` satisfies this composition."""
