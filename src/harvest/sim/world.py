"""SimWorld: the MuJoCo physics world for HARVEST (Phase 1.6).

Wraps the composed Gen3 + Robotiq 2F-85 + can scene with a clean read/control surface:
step the physics, command the arm and gripper, read proprioception and the can pose,
render the overhead and wrist cameras (RGB or depth), and read grasp-success ground
truth. This is the source of physically-grounded episodes for the SimSource (1.6d).
"""

from __future__ import annotations

import os
import platform

# macOS needs the CGL offscreen GL backend for headless rendering; set before any
# Renderer is created. (On the Linux cluster we do inference, not sim rendering.)
if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco
import numpy as np

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

# ONE process-global render context, closed when the model or size changes. A `mujoco.Renderer`
# holds a GL framebuffer + context; a new SimWorld per episode meant a new renderer per episode,
# and Python GC does not free the GL context, so it leaked one per episode and exhausted memory on
# a long generation run. Keyed on model identity so each episode reuses (then replaces) the single
# live renderer. (Same fix pattern as `_overhead_px` in `sim/reorient.py`.)
_RENDER: dict = {"model": None, "hw": None, "r": None}


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
    def set_arm(self, q) -> None:
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

    def move_pinch_to(self, target, wrist=None, max_steps: int = 150, tol: float = 0.008,
                      damping: float = 0.1, gain: float = 0.6, on_step=None) -> None:
        """Damped-least-squares IK driving the gripper_pinch site to `target` by stepping.

        With `wrist=None` all 7 joints solve for position. With a `wrist` angle, joint 7
        (wrist roll) is held to aim the finger-closing axis while joints 1-6 solve for
        position, so the fingers aim without moving the grasp point. The IK lives here in the
        backend, so a manipulation policy stays MuJoCo-free and drives any backend that
        provides this method.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        target = np.asarray(target, dtype=float)
        ndof = 6 if wrist is not None else 7
        for _ in range(int(max_steps)):
            if wrist is not None:
                self.data.ctrl[6] = wrist
            err = target - self.data.site_xpos[self._pinch_sid]
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._pinch_sid)
            j = jacp[:, :ndof]
            dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(3), err)
            q = self.data.qpos[:7].copy()
            q[:ndof] += gain * dq
            if wrist is not None:
                q[6] = wrist
            self.set_arm(q)
            self.step(5)
            if on_step is not None:
                on_step()

    def pinch_rotation(self) -> np.ndarray:
        """World rotation matrix (3x3) of the gripper pinch frame."""
        return self.data.site_xmat[self._pinch_sid].reshape(3, 3).copy()

    def move_pinch_pose(self, target_pos, target_rot, max_steps: int = 140, damping: float = 0.15,
                        pos_gain: float = 0.5, rot_gain: float = 0.15, on_step=None) -> None:
        """6-DOF damped-least-squares IK driving the pinch site to a target position AND
        orientation. Used to reorient a grasped can to present its label. Kept gentle (low
        rotation gain) so the grasp is not shocked loose. Lives in the backend so the policy
        stays MuJoCo-free."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        tpos = np.asarray(target_pos, dtype=float)
        rt = np.asarray(target_rot, dtype=float)
        for _ in range(int(max_steps)):
            pe = tpos - self.data.site_xpos[self._pinch_sid]
            rc = self.data.site_xmat[self._pinch_sid].reshape(3, 3)
            re = 0.5 * (np.cross(rc[:, 0], rt[:, 0]) + np.cross(rc[:, 1], rt[:, 1])
                        + np.cross(rc[:, 2], rt[:, 2]))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._pinch_sid)
            jac = np.vstack([jacp[:, :7], jacr[:, :7]])
            e = np.concatenate([pos_gain * pe, rot_gain * re])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), e)
            self.set_arm(self.data.qpos[:7] + dq)
            self.step(5)
            if on_step is not None:
                on_step()

    def overhead_label_visibility(self, height: int = 200, width: int = 200) -> float:
        """Label-visibility ground truth in [0, 1] from the overhead camera.

        The exposed nutrition-label coverage (segmentation, a vision read, never tactile).
        0 means the label is not exposed to the camera, 1 a well-presented label. This is the
        overhead-RGB label-visibility signal the pick-and-reorient task is scored on (F7, the
        vision channel is independent of the grasp-stability label)."""
        ren = mujoco.Renderer(self.model, height, width)
        ren.update_scene(self.data, camera="overhead")
        ren.enable_segmentation_rendering()
        seg = ren.render()
        ren.disable_segmentation_rendering()
        label_px = int((seg[..., 0] == self._label_gid).sum())
        return min(1.0, label_px / _OVERHEAD_LABEL_REF_PX)

    # --- reads (ground truth) ---
    def proprioception(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

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
        """Wrist wrench (Fx,Fy,Fz,Tx,Ty,Tz), derived from the arm joint torques via the
        wrist Jacobian. This is the joint-torque-derived F/T of correction F3, the same
        estimate the real Gen3 provides by default without the optional 6-axis add-on."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self._wrist_bid)
        jac_arm = np.vstack([jacp, jacr])[:, :7]  # 6 x 7 (arm DOFs)
        # Subtract the gravity + Coriolis bias so the wrench carries the CONTACT load,
        # not the arm holding its own weight (validity fix C5). At a settled no-contact
        # hold the actuators just cancel the bias, so this reads ~0.
        tau = self.data.qfrc_actuator[:7] - self.data.qfrc_bias[:7]
        return (np.linalg.pinv(jac_arm.T) @ tau).copy()

    def tactile(self) -> np.ndarray:
        """A (4x7) tactile pressure-map PROXY built from finger-pad contacts.

        Each pad contact's normal force is binned into a cell by its height on the can
        (row) and which pad it is (left -> columns 0-2, right -> columns 4-6; column 3 is
        the inter-finger gap), with a small lateral spread. This yields a spatially varying
        map rather than one value per pad. A stand-in for the real 28-taxel TSF-85 (whose
        taxel layout differs), replaced on the real robot in Phase 3 (validity fix C4)."""
        m = np.zeros((4, 7), dtype=float)
        buf = np.zeros(6)
        span = 2.0 * CAN_HALF_HEIGHT
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self._left_pad_geoms:
                base = 1
            elif pair & self._right_pad_geoms:
                base = 5
            else:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, buf)
            f = abs(float(buf[0]))
            row = int(np.clip((c.pos[2] - self._can_rest_z + CAN_HALF_HEIGHT) / span * 4.0, 0, 3))
            m[row, base] += f
            m[row, base - 1] += 0.5 * f
            m[row, base + 1] += 0.5 * f
        return m

    def sample(self, modality: Modality, timestamp_ns: int) -> Sample:
        """Read one modality from the current sim state as a schema Sample."""
        readers = {
            Modality.PROPRIOCEPTION: lambda: self.proprioception(),
            Modality.FORCE_TORQUE: lambda: self.force_torque(),
            Modality.TACTILE: lambda: self.tactile(),
            Modality.RGB_OVERHEAD: lambda: self.render("overhead"),
            Modality.DEPTH_OVERHEAD: lambda: self.render("overhead", depth=True),
            Modality.RGB_WRIST: lambda: self.render("wrist"),
            Modality.DEPTH_WRIST: lambda: self.render("wrist", depth=True),
        }
        if modality not in readers:
            raise ValueError(f"sim has no reader for modality {modality}")
        return Sample(modality=modality, timestamp_ns=timestamp_ns, data=readers[modality](), notes="sim")

    def render(self, camera: str = "overhead", depth: bool = False, height: int = 96, width: int = 96) -> np.ndarray:
        if _RENDER["model"] is not self.model or _RENDER["hw"] != (height, width):
            if _RENDER["r"] is not None:
                _RENDER["r"].close()                 # free the previous GL context before replacing
            _RENDER["r"] = mujoco.Renderer(self.model, height, width)
            _RENDER["model"], _RENDER["hw"] = self.model, (height, width)
        r = _RENDER["r"]
        r.update_scene(self.data, camera=camera)
        if depth:
            r.enable_depth_rendering()
            img = r.render().copy()
            r.disable_depth_rendering()
            return img
        return r.render().copy()
