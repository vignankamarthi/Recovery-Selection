"""Sensing reads for SimWorld (Part 1): force-torque, tactile, camera render, and the sample
dispatch. Free functions the thin `SimWorld` methods delegate to, keeping the world file a small
control + read surface while the modality math lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from harvest.sim._render import render_camera
from harvest.sim.scene import CAN_HALF_HEIGHT
from schema.streams import Modality, Sample

if TYPE_CHECKING:  # avoid a circular import; world.py imports this module.
    from harvest.sim.world import SimWorld


def force_torque(world: "SimWorld") -> np.ndarray:
    """Wrist wrench (Fx,Fy,Fz,Tx,Ty,Tz), derived from the arm joint torques via the wrist Jacobian.
    This is the joint-torque-derived F/T of correction F3, the same estimate the real Gen3 provides
    by default without the optional 6-axis add-on."""
    jacp = np.zeros((3, world.model.nv))
    jacr = np.zeros((3, world.model.nv))
    mujoco.mj_jacBody(world.model, world.data, jacp, jacr, world._wrist_bid)
    jac_arm = np.vstack([jacp, jacr])[:, :7]  # 6 x 7 (arm DOFs)
    # Subtract the gravity + Coriolis bias so the wrench carries the CONTACT load, not the arm
    # holding its own weight (validity fix C5). At a settled no-contact hold the actuators just
    # cancel the bias, so this reads ~0.
    tau = world.data.qfrc_actuator[:7] - world.data.qfrc_bias[:7]
    return (np.linalg.pinv(jac_arm.T) @ tau).copy()


def tactile(world: "SimWorld") -> np.ndarray:
    """A (4x7) tactile pressure-map PROXY built from finger-pad contacts.

    Each pad contact's normal force is binned into a cell by its height on the can (row) and which
    pad it is (left -> columns 0-2, right -> columns 4-6; column 3 is the inter-finger gap), with a
    small lateral spread. This yields a spatially varying map rather than one value per pad. A
    stand-in for the real 28-taxel TSF-85 (whose taxel layout differs), replaced on the real robot
    in Phase 3 (validity fix C4)."""
    m = np.zeros((4, 7), dtype=float)
    buf = np.zeros(6)
    span = 2.0 * CAN_HALF_HEIGHT
    for i in range(world.data.ncon):
        c = world.data.contact[i]
        pair = {c.geom1, c.geom2}
        if pair & world._left_pad_geoms:
            base = 1
        elif pair & world._right_pad_geoms:
            base = 5
        else:
            continue
        mujoco.mj_contactForce(world.model, world.data, i, buf)
        f = abs(float(buf[0]))
        row = int(np.clip((c.pos[2] - world._can_rest_z + CAN_HALF_HEIGHT) / span * 4.0, 0, 3))
        m[row, base] += f
        m[row, base - 1] += 0.5 * f
        m[row, base + 1] += 0.5 * f
    return m


def render(world: "SimWorld", camera: str = "overhead", depth: bool = False,
           height: int = 96, width: int = 96) -> np.ndarray:
    """One RGB or depth frame from `camera`, via the single process-global renderer in `_render.py`."""
    return render_camera(world.model, world.data, camera, depth, height, width)


def sample(world: "SimWorld", modality: Modality, timestamp_ns: int) -> Sample:
    """Read one modality from the current sim state as a schema Sample."""
    readers = {
        Modality.PROPRIOCEPTION: lambda: world.proprioception(),
        Modality.FORCE_TORQUE: lambda: force_torque(world),
        Modality.TACTILE: lambda: tactile(world),
        Modality.RGB_OVERHEAD: lambda: render(world, "overhead"),
        Modality.DEPTH_OVERHEAD: lambda: render(world, "overhead", depth=True),
        Modality.RGB_WRIST: lambda: render(world, "wrist"),
        Modality.DEPTH_WRIST: lambda: render(world, "wrist", depth=True),
    }
    if modality not in readers:
        raise ValueError(f"sim has no reader for modality {modality}")
    return Sample(modality=modality, timestamp_ns=timestamp_ns, data=readers[modality](), notes="sim")
