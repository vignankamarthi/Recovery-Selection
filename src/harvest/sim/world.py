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

from harvest.sim.scene import CAN_HALF_HEIGHT, build_scene
from schema.episode import ConditionClass
from schema.streams import Modality, Sample

# Gen3 "home" arm pose (the Menagerie keyframe), 7 joints.
HOME_QPOS = np.array(
    [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633]
)
# The can must rise this far above its rest height to count as lifted/grasped.
LIFT_THRESHOLD_M = 0.10


class SimWorld:
    """A steppable Gen3 + gripper + can world with modality reads and control."""

    def __init__(
        self,
        can_pos: tuple[float, float, float] = (0.5, 0.0, CAN_HALF_HEIGHT),
        condition: ConditionClass = ConditionClass.NOMINAL,
        can_seed: int = 0,
    ):
        self.model = build_scene(can_pos, condition, can_seed)
        self.data = mujoco.MjData(self.model)
        self._can_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "can")
        self._can_rest_z = can_pos[2]
        self._wrist_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")
        self._pinch_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch")

        def _gid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

        self._left_pad_geoms = {_gid("gripper_left_pad1"), _gid("gripper_left_pad2")}
        self._right_pad_geoms = {_gid("gripper_right_pad1"), _gid("gripper_right_pad2")}
        # Full-physics state spec for deterministic snapshot/restore (validity fix C2).
        self._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self._state_size = mujoco.mj_stateSize(self.model, self._state_spec)
        self._renderer: mujoco.Renderer | None = None
        self._renderer_hw: tuple[int, int] | None = None
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = HOME_QPOS
        self.data.ctrl[:7] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

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

    # --- reads (ground truth) ---
    def proprioception(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

    def can_position(self) -> np.ndarray:
        return self.data.xpos[self._can_bid].copy()

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
        if self._renderer is None or self._renderer_hw != (height, width):
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._renderer_hw = (height, width)
        r = self._renderer
        r.update_scene(self.data, camera=camera)
        if depth:
            r.enable_depth_rendering()
            img = r.render().copy()
            r.disable_depth_rendering()
            return img
        return r.render().copy()
