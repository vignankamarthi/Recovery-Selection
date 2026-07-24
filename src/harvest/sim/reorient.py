"""Sim demonstration generator for the lying-can pick-and-reorient task (Part 1, 1.7d).

All cans spawn LYING. The demonstration is a 3-stage graded pipeline: grasp the can's head,
RIGHT it to upright, then PRESENT the nutrition label flat to the overhead camera.

Two design choices, both forced by what the sim does well and poorly (see the sim audit):

  WELD. The sim's rigid-pad friction cannot hold a can through a reorient, so we kinematically
  attach the can to the hand (force its pose to follow the pinch each step). The weld is used
  ONLY where contact physics fails (the flip and the righting reorient). Grasp-stability is
  therefore a separate SIMULATOR-default label, not read off the welded grasp (it is a real
  tactile/physics signal on hardware; here it is a placeholder the dataset card flags).

  PLAN-THEN-EXECUTE. The label sits at a fixed spot in the hand, so "upright" and "label-up"
  are each a whole family of poses whose reachable member differs per can. We SEARCH that family
  on a HIDDEN scratch copy of the sim (the recorder and viewer see nothing), keep the winning
  full-state snapshots, then GLIDE the real sim to each one as a single smooth motion. The
  search never touches the demonstration; only the glide is recorded.

This module is deliberately MuJoCo-specific: it is the sim stand-in for teleoperated demos, so
it lives in `sim/`, not in the backend-agnostic `control/policy.py`. On hardware the
demonstrations come from teleoperation instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import mujoco

from harvest.sim.world import SimWorld
from schema.episode import ConditionClass

OnStep = Optional[Callable[[], None]]

HOVER_Z = 0.40
INSPECT_XY = (0.5, 0.0)          # under the fixed overhead camera (the can spawn spot)
N_CANDIDATES = 24
UP_THRESH = 0.92                 # label normal z to count as facing the overhead camera
GOOD_PX = 200                    # overhead label pixels that count as CLEARLY seen (search target)
# Coverage bar for `label_visible`. Set low enough that a CLEANLY presented can (even one the
# gripper partly occludes) clears it, so baseline occlusion does not manufacture condition-free
# failures. A can that SLIPS rolls its label off the top, coverage collapses, and it drops below.
LABEL_VISIBLE_PX = 70
# Exactly ONE live renderer, ever. Each episode builds a new SimWorld (a new MjModel), so caching a
# renderer per model leaked an OpenGL framebuffer + context per episode and exhausted memory on a
# long generation run. We keep a single renderer and close it the moment the model changes.
_RENDERER: dict = {"model": None, "r": None}

# Condition-correlated in-hand SLIP (the failure source). A damaged can slips in the grasp during
# the reorient: the can rolls about its own long axis, so the label rolls off the top and the
# overhead read fails. Severity is condition-scaled (nominal barely slips, low-friction/out-of-round
# bulge and rust slip most), with a per-can jitter. This is where tactile earns its keep on real
# hardware, so it is the failure the tactile-vs-vision ablation is built to measure. The label
# normal z after a roll of `a` is cos(a), and UP_THRESH=0.92 is a ~23 deg cliff, so MAX_SLIP_DEG
# rolls the worst cans well past it while nominal cans stay clean.
# A roll past ~23 deg (severity ~0.46 at MAX_SLIP_DEG=50) drops the label below UP_THRESH, so the
# bases put nominal well clear of that cliff and escalate damage across it: nominal ~always passes,
# body/seam dents fail some, bulge and rust (out-of-round, low-friction) fail most. A seeded jitter
# spreads each condition across the cliff so no class is all-or-nothing.
MAX_SLIP_DEG = 50.0
_SLIP_BASE = {
    ConditionClass.NOMINAL: 0.10,
    ConditionClass.BODY_DENT: 0.40,
    ConditionClass.SEAM_DENT: 0.46,
    ConditionClass.BULGE: 0.60,
    ConditionClass.RUST: 0.66,
}


def slip_severity(condition: ConditionClass, seed: int) -> float:
    """Deterministic per-can slip severity in [0, 1]: a condition base plus a seeded jitter, so a
    damaged can slips more than a nominal one but individual cans still vary."""
    base = _SLIP_BASE.get(condition, 0.35)
    jitter = ((int(seed) * 2654435761) % 1000) / 1000.0 * 0.36 - 0.18   # deterministic in [-0.18, 0.18]
    return float(min(1.0, max(0.0, base + jitter)))


@dataclass
class ReorientResult:
    """The two REAL graded stage signals of one demonstration (grasp-stability is a caller-side
    sim default). The nutrition label sits on the cylinder WALL, so it faces the fixed OVERHEAD
    camera only when the can is held HORIZONTAL with the label rolled to the top. There is NO
    stand-the-can-upright step: that pose puts the label on the side, away from the camera.

    `upright_success` (`upright` = presented label-up and right-side-up, NOT axis-vertical): the
    in-hand reorient brought the label normal to point at the overhead camera. `label_visible`:
    the overhead camera actually reads enough label pixels (coverage clears `LABEL_VISIBLE_PX`,
    Padir's legibility criterion), which the gripper can occlude even when the label faces up.
    """

    upright_success: bool
    label_visible: bool
    label_nz: float
    overhead_px: Optional[int]


def _overhead_px(w: SimWorld) -> Optional[int]:
    """Overhead label pixel count via segmentation, or None if a renderer is unavailable."""
    try:
        if _RENDERER["model"] is not w.model:          # identity compare, never id() (it gets reused)
            if _RENDERER["r"] is not None:
                _RENDERER["r"].close()                 # free the previous GL context before replacing
            _RENDERER["r"] = mujoco.Renderer(w.model, 200, 200)
            _RENDERER["model"] = w.model
        r = _RENDERER["r"]
        r.update_scene(w.data, camera="overhead")
        r.enable_segmentation_rendering()
        seg = r.render()
        r.disable_segmentation_rendering()
        return int((seg[..., 0] == w._label_gid).sum())
    except Exception:
        return None


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two (w,x,y,z) quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:                              # nearly parallel: lerp + normalize
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d) * t
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


class Weld:
    """Glue the can rigidly to the hand: capture the can-in-hand offset, then force the can to
    follow the pinch each step. Models a firm grasp without fighting the sim's weak pad friction."""

    def __init__(self, w: SimWorld) -> None:
        self.w = w
        jid = int(w.model.body_jntadr[w._can_bid])
        self.q = int(w.model.jnt_qposadr[jid])
        self.dof = int(w.model.jnt_dofadr[jid])
        Rp, pp = w.pinch_rotation(), w.pinch_position()
        self.p_rel = Rp.T @ (w.can_position() - pp)
        self.R_rel = Rp.T @ w.can_orientation()

    def follow(self) -> None:
        w = self.w
        Rp, pp = w.pinch_rotation(), w.pinch_position()
        pc = pp + Rp @ self.p_rel
        Rc = Rp @ self.R_rel
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rc.flatten())
        w.data.qpos[self.q:self.q + 3] = pc
        w.data.qpos[self.q + 3:self.q + 7] = q
        w.data.qvel[self.dof:self.dof + 6] = 0
        mujoco.mj_forward(w.model, w.data)


def _grasp_head(w: SimWorld, on_step: OnStep = None) -> None:
    """Approach the lying can's head (not its middle), close, and settle. The caller then welds;
    the physical lift need not succeed, the sim cannot hold this contact through a reorient anyway."""
    can = w.can_position()
    axis = w.can_long_axis()
    axis = axis / np.linalg.norm(axis)
    end = can + 0.035 * axis                              # the head, not the middle
    wrist = w.aligned_wrist(np.arctan2(axis[1], axis[0]) + np.pi / 2)
    w.set_gripper(0.0)
    w.move_pinch_to(end + [0, 0, 0.12], wrist=wrist, on_step=on_step)
    w.move_pinch_to(end, wrist=wrist, on_step=on_step)
    w.set_gripper(1.0)
    for _ in range(30):
        w.step(5)
        if on_step is not None:
            on_step()


def _search_present(w: SimWorld, offset: np.ndarray):
    """On the CURRENT (scratch) data (can freshly grasped, lying), search the presentation family
    DIRECTLY for the horizontal-label-up pose the overhead camera sees best, with no upright
    waypoint. The label sits at a fixed spot in the hand, so "label-up" is a whole family of
    wrist rolls; we sweep it and keep the best-seen reachable member. Returns (winning full-state
    snapshot, label_nz, overhead_px) or (None, -1.0, None)."""
    hover = np.array([INSPECT_XY[0], INSPECT_XY[1], HOVER_Z])
    w.move_pinch_pose(hover, w.pinch_rotation(), max_steps=20)
    Rp0 = w.pinch_rotation()
    _, nrm = w.can_label_pose()
    label_in_hand = Rp0.T @ nrm
    long_in_hand = Rp0.T @ w.can_long_axis()
    label_in_hand /= np.linalg.norm(label_in_hand)
    long_in_hand /= np.linalg.norm(long_in_hand)
    src = np.column_stack([label_in_hand, long_in_hand, np.cross(label_in_hand, long_in_hand)])
    snap = w.snapshot()
    best_snap, best_score, best_nz, best_px = None, -1.0, -1.0, None
    for h_ang in np.linspace(0, 2 * np.pi, N_CANDIDATES, endpoint=False):
        w.restore(snap)
        weld = Weld(w)
        h = np.array([np.cos(h_ang), np.sin(h_ang), 0.0])
        target = np.column_stack([[0, 0, 1.0], h, np.cross([0, 0, 1.0], h)]) @ src.T
        for _ in range(18):
            w.move_pinch_pose(hover, target, max_steps=10, damping=0.1, rot_gain=0.4, on_step=weld.follow)
            weld.follow()
        _, n = w.can_label_pose()
        nz = float(n[2])
        if nz <= UP_THRESH:
            continue
        px = _overhead_px(w)
        score = float(px) if px is not None else nz
        if score > best_score:
            best_score, best_snap, best_nz, best_px = score, w.snapshot(), nz, px
        if px is not None and px >= GOOD_PX:
            break
    return best_snap, best_nz, best_px


def _plan(w: SimWorld, offset: np.ndarray):
    """Search the presentation family on a HIDDEN scratch copy (viewer/recorder untouched).
    Returns (present_snap, nz, px) or None. The real sim is left exactly as it was."""
    real = w.data
    scratch = mujoco.MjData(w.model)
    mujoco.mj_copyData(scratch, w.model, real)
    w.data = scratch
    try:
        present_snap, nz, px = _search_present(w, offset)
        if present_snap is None:
            return None
        return present_snap, nz, px
    finally:
        w.data = real


def _glide(w: SimWorld, snap: np.ndarray, on_step: OnStep = None, steps: int = 26) -> None:
    """Smoothly move the real sim from its current state to `snap` (arm joints + can position
    linear, can orientation slerp). Kinematic, so it lands exactly on the target snapshot."""
    q0 = w.data.qpos.copy()
    here = w.snapshot()                                   # remember where we are
    w.restore(snap)                                       # peek at the target qpos
    q1 = w.data.qpos.copy()
    w.restore(here)                                       # ...then step back
    jid = int(w.model.body_jntadr[w._can_bid])
    cq = int(w.model.jnt_qposadr[jid])
    r0, r1 = q0[cq + 3:cq + 7].copy(), q1[cq + 3:cq + 7].copy()
    if np.dot(r0, r1) < 0:
        r1 = -r1
    for i in range(1, steps + 1):
        t = i / steps
        q = q0 + (q1 - q0) * t
        q[cq:cq + 3] = q0[cq:cq + 3] * (1 - t) + q1[cq:cq + 3] * t     # can position
        q[cq + 3:cq + 7] = _slerp(r0, r1, t)                           # can orientation
        w.data.qpos[:] = q
        mujoco.mj_forward(w.model, w.data)
        if on_step is not None:
            on_step()
    w.restore(snap)                                       # land exactly on the target
    if on_step is not None:
        on_step()


def _axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    """Rotation matrix for a rotation of `ang` radians about unit `axis` (Rodrigues' formula)."""
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = a
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _slip_roll(w: SimWorld, severity: float, on_step: OnStep = None, steps: int = 8) -> None:
    """A damaged can SLIPS: the ARM rolls the grasped can about the can's OWN long axis, driven by IK
    on the end-effector orientation, so the presented label rolls off the top of the overhead view.
    Rolling THROUGH THE ARM (not by rewriting the can's weld offset) puts the failure in the ACTION
    SPACE ACT imitates: IK adjusts all seven arm joints, so a damaged can's arm trajectory differs
    from a nominal can's, the demos are teachable, and Part 2's recovery has an arm action to act on.
    The weld carries the can rigidly with the hand, and the roll is spread over several recorded steps
    so the slip is a progressive shift, not a teleport."""
    weld = Weld(w)                                       # keep the can-in-hand offset fixed
    total = np.radians(MAX_SLIP_DEG) * float(severity)
    axis = w.can_long_axis()
    axis = axis / np.linalg.norm(axis)                   # roll about the can's own long axis (world frame)
    pos = w.pinch_position()                             # hold the pinch point; only its orientation rolls
    R0 = w.pinch_rotation()
    for i in range(1, steps + 1):
        target = _axis_angle(axis, total * i / steps) @ R0
        w.move_pinch_pose(pos, target, max_steps=10, damping=0.1, rot_gain=0.4, on_step=weld.follow)
        weld.follow()                                    # can follows the rolled hand rigidly
        if on_step is not None:
            on_step()


def demonstrate(world: SimWorld, on_step: OnStep = None, slip: float = 0.0) -> ReorientResult:
    """Run one lying-can pick-and-reorient demonstration on `world` and return the stage signals.

    Grasp the head + weld, PLAN the horizontal-label-up presentation on a hidden scratch copy
    (no upright detour), then EXECUTE by gliding smoothly to that one planned state. `slip` (0..1,
    condition-scaled by the caller) then rolls the can in-hand so a damaged can's label rolls off
    the top and the read fails, while a nominal can stays clean. `on_step` (the recorder tick)
    fires during the grasp, glide, and slip only, never during the invisible search. A plan
    failure (no reachable presentation from this grasp) is reported, never raised.
    """
    _grasp_head(world, on_step)                                   # grasp the lying can (weld-assist)
    offset = world.pinch_rotation() @ world.can_orientation().T
    plan = _plan(world, offset)                                   # PLAN (invisible)
    if plan is None:
        _, n = world.can_label_pose()
        return ReorientResult(False, False, float(n[2]), None)
    present_snap, _, _ = plan
    _glide(world, present_snap, on_step)                          # EXECUTE: one smooth reorient
    if slip > 0.0:
        _slip_roll(world, slip, on_step)                         # condition-correlated in-hand slip
    _, n = world.can_label_pose()
    nz = float(n[2])
    px = _overhead_px(world)
    upright_success = bool(nz > UP_THRESH)                        # label still faces up after any slip
    label_visible = bool(px is not None and px >= LABEL_VISIBLE_PX)   # overhead coverage clears the bar
    return ReorientResult(upright_success, label_visible, nz, px)
