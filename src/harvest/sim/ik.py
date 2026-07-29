"""Damped-least-squares IK solvers for SimWorld (Part 1).

Free functions the thin `SimWorld` methods delegate to, so the world file stays a small control +
read surface and the IK math lives on its own. Each takes the `SimWorld` and drives it by stepping
the physics (the IK lives in the backend so a manipulation policy stays MuJoCo-free).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import mujoco
import numpy as np

if TYPE_CHECKING:  # avoid a circular import; world.py imports this module.
    from harvest.sim.world import SimWorld


def move_pinch_to(world: "SimWorld", target: "np.ndarray | Sequence[float]",
                  wrist: Optional[float] = None, max_steps: int = 150, tol: float = 0.008,
                  damping: float = 0.1, gain: float = 0.6,
                  on_step: Optional[Callable[[], None]] = None) -> None:
    """Damped-least-squares IK driving the gripper_pinch site to `target` by stepping.

    With `wrist=None` all 7 joints solve for position. With a `wrist` angle, joint 7 (wrist roll)
    is held to aim the finger-closing axis while joints 1-6 solve for position, so the fingers aim
    without moving the grasp point.
    """
    jacp = np.zeros((3, world.model.nv))
    jacr = np.zeros((3, world.model.nv))
    target = np.asarray(target, dtype=float)
    ndof = 6 if wrist is not None else 7
    for _ in range(int(max_steps)):
        if wrist is not None:
            world.data.ctrl[6] = wrist
        err = target - world.data.site_xpos[world._pinch_sid]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(world.model, world.data, jacp, jacr, world._pinch_sid)
        j = jacp[:, :ndof]
        dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(3), err)
        q = world.data.qpos[:7].copy()
        q[:ndof] += gain * dq
        if wrist is not None:
            q[6] = wrist
        world.set_arm(q)
        world.step(5)
        if on_step is not None:
            on_step()


def move_pinch_pose(world: "SimWorld", target_pos: "np.ndarray | Sequence[float]",
                    target_rot: np.ndarray, max_steps: int = 140, damping: float = 0.15,
                    pos_gain: float = 0.5, rot_gain: float = 0.15,
                    on_step: Optional[Callable[[], None]] = None) -> None:
    """6-DOF damped-least-squares IK driving the pinch site to a target position AND orientation.
    Used to reorient a grasped can to present its label. Kept gentle (low rotation gain) so the
    grasp is not shocked loose."""
    jacp = np.zeros((3, world.model.nv))
    jacr = np.zeros((3, world.model.nv))
    tpos = np.asarray(target_pos, dtype=float)
    rt = np.asarray(target_rot, dtype=float)
    for _ in range(int(max_steps)):
        pe = tpos - world.data.site_xpos[world._pinch_sid]
        rc = world.data.site_xmat[world._pinch_sid].reshape(3, 3)
        re = 0.5 * (np.cross(rc[:, 0], rt[:, 0]) + np.cross(rc[:, 1], rt[:, 1])
                    + np.cross(rc[:, 2], rt[:, 2]))
        mujoco.mj_jacSite(world.model, world.data, jacp, jacr, world._pinch_sid)
        jac = np.vstack([jacp[:, :7], jacr[:, :7]])
        e = np.concatenate([pos_gain * pe, rot_gain * re])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), e)
        world.set_arm(world.data.qpos[:7] + dq)
        world.step(5)
        if on_step is not None:
            on_step()
