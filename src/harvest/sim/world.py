"""SimWorld: the MuJoCo physics world for HARVEST (Phase 1.6).

Wraps the composed Gen3 + Robotiq 2F-85 + can scene with a clean read/control surface:
step the physics, command the arm and gripper, read proprioception and the can pose,
render the overhead and wrist cameras (RGB or depth), and read grasp-success ground
truth. This is the source of physically-grounded episodes for the SimSource (1.6d).

SimWorld is a thin surface: the IK solvers live in `sim/ik.py`, the sensing reads (force-torque,
tactile, render, sample) in `sim/sensing.py`, and the shared renderer in `sim/_render.py`. The
methods here delegate to them, keeping public signatures stable for both backends.
"""

from __future__ import annotations

import os
import platform

# macOS needs the CGL offscreen GL backend for headless rendering; set before any
# Renderer is created. (On the Linux cluster we do inference, not sim rendering.)
if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

from typing import Callable, Optional, Sequence

import mujoco
import numpy as np

from harvest.sim import ik, sensing
from harvest.sim._render import label_pixel_count
from harvest.sim.scene import CAN_HALF_HEIGHT, CAN_RADIUS, build_scene, can_label_yaw
from schema.episode import ConditionClass
from schema.streams import Modality, Sample

# Gen3 "home" arm pose (the Menagerie keyframe), 7 joints.
HOME_QPOS = np.array(
    [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633]
)
# The can must rise this far above its rest height to count as lifted/grasped.
LIFT_THRESHOLD_M = 0.10
# Overhead label pixels a well-presented label shows at the narrow-FOV inspection framing,
# the normalizer for the visibility score (measured: a fully-turned-up label covers ~220 px).
_OVERHEAD_LABEL_REF_PX = 220.0


class SimWorld:
    """A steppable Gen3 + gripper + can world with modality reads and control."""

    def __init__(
        self,
        can_pos: tuple[float, float, float] = (0.5, 0.0, CAN_HALF_HEIGHT),
        condition: ConditionClass = ConditionClass.NOMINAL,
        can_seed: int = 0,
        can_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        settle_steps: int = 250,
    ):
        self.model = build_scene(can_pos, condition, can_seed, can_quat)
        self.data = mujoco.MjData(self.model)
        self._settle_steps = int(settle_steps)
        self._label_yaw = can_label_yaw(can_seed)
        self._can_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "can")
        self._can_rest_z = can_pos[2]  # provisional; the settled value is captured in reset()
        self._wrist_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")
        self._pinch_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch")

        def _gid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

        self._left_pad_geoms = {_gid("gripper_left_pad1"), _gid("gripper_left_pad2")}
        self._right_pad_geoms = {_gid("gripper_right_pad1"), _gid("gripper_right_pad2")}
        self._label_gid = _gid("can_label")
        # Full-physics state spec for deterministic snapshot/restore (validity fix C2).
        self._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self._state_size = mujoco.mj_stateSize(self.model, self._state_spec)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = HOME_QPOS
        self.data.ctrl[:7] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        # Let the can settle under gravity from its (possibly non-upright) spawn pose to a
        # stable rest, the proposal's unknown orientation. The arm sits clear at HOME. The
        # settled height becomes the lift-baseline, so a lying can (lower centroid) is scored
        # against its own rest, not the spawn height.
        for _ in range(self._settle_steps):
            mujoco.mj_step(self.model, self.data)
        self._can_rest_z = float(self.data.xpos[self._can_bid][2])

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)

    def snapshot(self) -> np.ndarray:
        """Full-physics state snapshot, for replaying recovery arms from an identical
        failure state (the counterfactual grid / recovery-regret oracle, C2)."""
        s = np.zeros(self._state_size)
        mujoco.mj_getState(self.model, self.data, s, self._state_spec)
        return s

    def restore(self, snap: np.ndarray) -> None:
        """Restore a snapshot exactly, enabling deterministic per-failure replay."""
        mujoco.mj_setState(self.model, self.data, np.asarray(snap), self._state_spec)
        mujoco.mj_forward(self.model, self.data)

    @property
    def time_ns(self) -> int:
        return int(round(self.data.time * 1e9))

    # --- control ---
    def set_arm(self, q: "np.ndarray | Sequence[float]") -> None:
        self.data.ctrl[:7] = np.asarray(q, dtype=float)[:7]

    def set_gripper(self, closed_frac: float) -> None:
        """0.0 = fully open, 1.0 = fully closed (maps to the 0-255 tendon actuator)."""
        self.data.ctrl[7] = float(np.clip(closed_frac, 0.0, 1.0)) * 255.0

    def pinch_position(self) -> np.ndarray:
        """World position of the gripper pinch site (the grasp point)."""
        return self.data.site_xpos[self._pinch_sid].copy()

    def aligned_wrist(self, target_angle: float) -> float:
        """Wrap a target wrist-roll angle into joint 7's range (fingers are pi-symmetric)."""
        lo, hi = self.model.jnt_range[6]
        if not bool(self.model.jnt_limited[6]):
            return float(target_angle)
        a = float(target_angle)
        while a > hi:
            a -= np.pi
        while a < lo:
            a += np.pi
        return float(np.clip(a, lo, hi))

    def wrap_continuous_joints(self) -> None:
        """Wrap each unlimited (continuous) arm joint into [-pi, pi]. Iterative IK can wind a
        continuous joint through many turns (recorded proprioception ran out to +/-65 rad), which
        makes the action a per-can idiosyncratic winding number instead of a bounded, learnable pose.
        Wrapping to the principal range is physically equivalent for a continuous joint (the pose is
        identical) and keeps the recorded action bounded so a policy can predict it."""
        for j in range(7):
            if not bool(self.model.jnt_limited[j]):
                adr = int(self.model.jnt_qposadr[j])
                self.data.qpos[adr] = (self.data.qpos[adr] + np.pi) % (2 * np.pi) - np.pi
        mujoco.mj_forward(self.model, self.data)

    def move_pinch_to(self, target: "np.ndarray | Sequence[float]", wrist: Optional[float] = None,
                      max_steps: int = 150, tol: float = 0.008, damping: float = 0.1,
                      gain: float = 0.6, on_step: Optional[Callable[[], None]] = None) -> None:
        """Damped-least-squares position IK driving the gripper_pinch site to `target`. A thin
        delegate to `harvest.sim.ik.move_pinch_to` (the IK math lives there so the world file stays
        a small control + read surface, and a manipulation policy stays MuJoCo-free)."""
        ik.move_pinch_to(self, target, wrist, max_steps, tol, damping, gain, on_step)

    def pinch_rotation(self) -> np.ndarray:
        """World rotation matrix (3x3) of the gripper pinch frame."""
        return self.data.site_xmat[self._pinch_sid].reshape(3, 3).copy()

    def move_pinch_pose(self, target_pos: "np.ndarray | Sequence[float]", target_rot: np.ndarray,
                        max_steps: int = 140, damping: float = 0.15, pos_gain: float = 0.5,
                        rot_gain: float = 0.15, on_step: Optional[Callable[[], None]] = None) -> None:
        """6-DOF damped-least-squares IK driving the pinch site to a target position AND
        orientation (to reorient a grasped can to present its label). A thin delegate to
        `harvest.sim.ik.move_pinch_pose`."""
        ik.move_pinch_pose(self, target_pos, target_rot, max_steps, damping, pos_gain, rot_gain, on_step)

    def overhead_label_visibility(self, height: int = 200, width: int = 200) -> float:
        """Label-visibility ground truth in [0, 1] from the overhead camera.

        The exposed nutrition-label coverage (segmentation, a vision read, never tactile).
        0 means the label is not exposed to the camera, 1 a well-presented label. This is the
        overhead-RGB label-visibility signal the pick-and-reorient task is scored on (F7, the
        vision channel is independent of the grasp-stability label). Uses the single process-global
        renderer in `sim/_render.py`."""
        label_px = label_pixel_count(self.model, self.data, self._label_gid, "overhead", height, width)
        return min(1.0, label_px / _OVERHEAD_LABEL_REF_PX)

    # --- reads (ground truth) ---
    def proprioception(self) -> np.ndarray:
        """The 7 arm joint angles, the recorded action space. Continuous joints are reported WRAPPED
        into [-pi, pi] (the standard convention) so the recorded action stays bounded and learnable
        even if the IK wound a joint through several turns on a hard reach. The pose is physically
        identical, and control reads raw `qpos` directly, so only the recorded stream is affected."""
        q = self.data.qpos[:7].copy()
        for j in range(7):
            if not bool(self.model.jnt_limited[j]):
                q[j] = (q[j] + np.pi) % (2 * np.pi) - np.pi
        return q

    def can_position(self) -> np.ndarray:
        return self.data.xpos[self._can_bid].copy()

    def can_orientation(self) -> np.ndarray:
        """Can body rotation matrix (3x3) in the world frame."""
        return self.data.xmat[self._can_bid].reshape(3, 3).copy()

    def can_long_axis(self) -> np.ndarray:
        """The can's cylinder axis (its local z) in the world frame."""
        return self.can_orientation()[:, 2].copy()

    def can_is_upright(self, tol_deg: float = 35.0) -> bool:
        """True if the cylinder axis is within tol of vertical (settled standing, not lying)."""
        tilt = np.degrees(np.arccos(np.clip(abs(float(self.can_long_axis()[2])), 0.0, 1.0)))
        return bool(tilt <= tol_deg)

    def can_label_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and outward normal of the nutrition-label patch on the can wall.

        The reorient (1.7c) turns this normal toward the overhead camera, and overhead depth
        verifies the exposure. The patch sits at radius `CAN_RADIUS` and angle `_label_yaw`
        in the body frame, its +x face the outward normal.
        """
        rot = self.can_orientation()
        local = np.array(
            [CAN_RADIUS * np.cos(self._label_yaw), CAN_RADIUS * np.sin(self._label_yaw), 0.0]
        )
        normal_local = np.array([np.cos(self._label_yaw), np.sin(self._label_yaw), 0.0])
        return (self.can_position() + rot @ local, rot @ normal_local)

    def grasp_success(self) -> bool:
        """Sim ground truth: has the can been lifted past the threshold?"""
        return bool(self.can_position()[2] - self._can_rest_z >= LIFT_THRESHOLD_M)

    def force_torque(self) -> np.ndarray:
        """Wrist wrench (Fx,Fy,Fz,Tx,Ty,Tz) from the arm joint torques (see harvest.sim.sensing)."""
        return sensing.force_torque(self)

    def tactile(self) -> np.ndarray:
        """A (4x7) tactile pressure-map PROXY from finger-pad contacts (see harvest.sim.sensing)."""
        return sensing.tactile(self)

    def sample(self, modality: Modality, timestamp_ns: int) -> Sample:
        """Read one modality from the current sim state as a schema Sample (see harvest.sim.sensing)."""
        return sensing.sample(self, modality, timestamp_ns)

    def render(self, camera: str = "overhead", depth: bool = False,
               height: int = 96, width: int = 96) -> np.ndarray:
        """One RGB or depth camera frame (see harvest.sim.sensing)."""
        return sensing.render(self, camera, depth, height, width)
