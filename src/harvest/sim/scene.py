"""MuJoCo scene composition for HARVEST (Phase 1.6 / 1.7).

Composes the real hardware stack from MuJoCo Menagerie: a Kinova Gen3 arm with a
Robotiq 2F-85 gripper attached at the tool flange, plus a can on a table and an
overhead top-down camera (the Gen3 already carries a wrist camera).

The can is built per ConditionClass as a RIGID geometric variant (C1). MuJoCo rigid
geoms are convex, so a true concave dent is not representable; dents are approximated by
out-of-round / crimped geometry that a depth sensor and the grasp physics can still tell
apart. Each can is randomized from a per-can seed, so every can_id is a distinct physical
unit and the fixed scripted grasp fails more on deformed / low-friction cans (organic,
condition-correlated failures, C3). numpy/mujoco live here, never in schema.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions import gen3_mj_description

from schema.episode import ConditionClass

_MENAGERIE = Path(gen3_mj_description.MJCF_PATH).parents[1]
_GEN3_SCENE = _MENAGERIE / "kinova_gen3" / "scene.xml"
_GRIPPER = _MENAGERIE / "robotiq_2f85" / "2f85.xml"

# A standard food can: ~66 mm diameter, ~100 mm tall.
CAN_RADIUS = 0.033
CAN_HALF_HEIGHT = 0.05
_METAL = [0.80, 0.25, 0.20, 1.0]
_RUST = [0.42, 0.26, 0.12, 1.0]
_LABEL = [0.95, 0.95, 0.98, 1.0]  # bright nutrition-label patch

# Overhead RGB-D camera, ceiling-mounted. High standoff + narrow FOV so a small gripper does
# not occlude the label the robot presents beneath it (matches a real ceiling Kinect).
OVERHEAD_CAM_HEIGHT = 1.7
OVERHEAD_CAM_FOVY = 26.0  # frames the inspection region where the robot presents the flat label

# The organic-failure lever (C3). Under the firm 2F-85 pinch, friction and width barely
# affect grasp success; the reliable, physically-grounded failure driver is displacing the
# can's graspable centroid from the nominal grid pose the fixed script reaches for. A
# deformation shifts that centroid, so the script grabs off-center and the lift can fail.
# Radial magnitude range (metres) per condition; the fixed script starts to miss past
# ~22 mm. rust is a surface condition (little geometric displacement), so its organic
# failures are rare here; its real slip failures are INJECTED in Phase 1.9.
# Tuned against the measured success-vs-offset curve (flat 100% to ~10 mm, transition
# band 12-20 mm) to give each class a realistic organic failure rate: every class mostly
# succeeds, and failure rises with deformation severity.
_OFFSET_RANGE = {
    ConditionClass.NOMINAL: (0.000, 0.000),   # ~100% success
    ConditionClass.BODY_DENT: (0.008, 0.020),  # most deformed, ~55-65%
    ConditionClass.SEAM_DENT: (0.004, 0.014),  # localized rim dent, ~85%
    ConditionClass.BULGE: (0.006, 0.017),      # swelling, ~70%
    ConditionClass.RUST: (0.000, 0.010),       # surface only, ~100%
}


def can_seed_from_id(can_id: str) -> int:
    """Deterministic per-can seed so each can_id maps to a fixed, distinct geometry."""
    return int(hashlib.sha256(can_id.encode()).hexdigest()[:8], 16)


def can_label_yaw(can_seed: int) -> float:
    """The angle (radians) of the nutrition-label patch around the can wall.

    Fixed per physical can (seeded by `can_id`), on a stream independent of the geometry
    randomization so it stays stable. The label's location on the can is a property of the
    can, while the can's placement orientation varies per episode.
    """
    return float(np.random.default_rng(can_seed + 999983).uniform(0.0, 2.0 * np.pi))


def _add_can(
    world: mujoco.MjSpec,
    condition: ConditionClass,
    rng: np.random.Generator,
    pos,
    quat,
    label_yaw: float,
) -> None:
    """Add a condition-specific, per-can-randomized can body to the world.

    The distinctive shape (ellipsoid / crimped rim / bulge / rust colour) gives each class a
    look a depth/vision sensor can separate; the per-can lateral offset gives it the
    condition-correlated organic grasp-failure rate. `quat` is the per-episode placement
    orientation (the can settles under gravity from it, the proposal's unknown orientation),
    and a bright nutrition-label patch sits on the wall at `label_yaw`.
    """
    g = mujoco.mjtGeom
    r = CAN_RADIUS * float(rng.uniform(0.95, 1.05))
    h = CAN_HALF_HEIGHT * float(rng.uniform(0.95, 1.05))
    lo, hi = _OFFSET_RANGE[condition]
    mag = float(rng.uniform(lo, hi))
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    ox, oy = mag * float(np.cos(ang)), mag * float(np.sin(ang))  # graspable-centroid shift
    can = world.worldbody.add_body(name="can", pos=list(pos))
    can.quat = list(quat)
    can.add_freejoint()

    if condition is ConditionClass.BODY_DENT:
        squash = float(rng.uniform(0.62, 0.82))  # out-of-round cross-section
        can.add_geom(name="can_geom", type=g.mjGEOM_ELLIPSOID,
                     size=[r * squash, r, h], pos=[ox, oy, 0.0], rgba=_METAL)
    elif condition is ConditionClass.SEAM_DENT:
        can.add_geom(name="can_geom", type=g.mjGEOM_CYLINDER,
                     size=[r, h, 0.0], pos=[ox, oy, 0.0], rgba=_METAL)
        crimp = float(rng.uniform(0.60, 0.82))
        can.add_geom(name="can_seam", type=g.mjGEOM_CYLINDER,
                     size=[r * crimp, h * 0.14, 0.0], pos=[ox, oy, h], rgba=_METAL)
    elif condition is ConditionClass.BULGE:
        can.add_geom(name="can_geom", type=g.mjGEOM_CYLINDER,
                     size=[r, h, 0.0], pos=[ox, oy, 0.0], rgba=_METAL)
        bulge = float(rng.uniform(1.18, 1.45))
        can.add_geom(name="can_bulge", type=g.mjGEOM_ELLIPSOID,
                     size=[r * bulge, r * bulge, h * 0.40], pos=[ox, oy, 0.0], rgba=_METAL)
    elif condition is ConditionClass.RUST:
        geom = can.add_geom(name="can_geom", type=g.mjGEOM_CYLINDER,
                            size=[r, h, 0.0], pos=[ox, oy, 0.0], rgba=_RUST)
        geom.friction = [float(rng.uniform(0.35, 0.65)), 0.005, 0.0001]  # flaky surface
    else:  # NOMINAL
        can.add_geom(name="can_geom", type=g.mjGEOM_CYLINDER, size=[r, h, 0.0], rgba=_METAL)

    # The nutrition-label band: a tall bright panel on the lower wall, its +x face pointing
    # radially out (the label normal). It sits low on the can so a grasp near the top rim
    # leaves it exposed (the gripper does not cover the label), and the robot reorients the can
    # to turn this face toward the inspection camera for verification (proposal task).
    lx, ly = r * float(np.cos(label_yaw)), r * float(np.sin(label_yaw))
    label = can.add_geom(
        name="can_label", type=g.mjGEOM_BOX, size=[0.002, 0.014, 0.024],
        pos=[lx + ox, ly + oy, -0.018], rgba=_LABEL,
    )
    label.quat = [float(np.cos(label_yaw / 2)), 0.0, 0.0, float(np.sin(label_yaw / 2))]
    # "This side up" marker: a bright stripe at the TOP edge of the label (toward the can's local
    # +z, deterministic). Makes the label's up-direction visible in renders, so a presented label
    # is verifiably right-side-up, never upside down.
    top_marker = can.add_geom(
        name="can_label_top", type=g.mjGEOM_BOX, size=[0.0025, 0.014, 0.004],
        pos=[lx + ox, ly + oy, -0.018 + 0.024 - 0.004], rgba=[0.10, 0.45, 0.95, 1.0],
    )
    top_marker.quat = label.quat


def build_scene(
    can_pos: tuple[float, float, float] = (0.5, 0.0, CAN_HALF_HEIGHT),
    condition: ConditionClass = ConditionClass.NOMINAL,
    can_seed: int = 0,
    can_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> mujoco.MjModel:
    """Compose Gen3 + Robotiq 2F-85 + a condition-specific can + overhead camera.

    `can_quat` is the can's initial placement orientation (identity = upright). The caller
    spawns it slightly above the table and lets it settle, so an arbitrary `can_quat` gives
    the proposal's unknown initial orientation (upright or lying).
    """
    world = mujoco.MjSpec.from_file(str(_GEN3_SCENE))
    gripper = mujoco.MjSpec.from_file(str(_GRIPPER))

    # Contact solver settings the Robotiq gripper wants for a stable grasp (the attach
    # otherwise keeps the arm scene's weaker defaults). Elliptic friction cone plus a
    # higher impratio so the fingers hold the can instead of letting it slip.
    world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    world.option.impratio = 10.0

    # Attach the gripper at the Gen3 tool flange (pinch_site on bracelet_link).
    world.attach(gripper, prefix="gripper_", site=world.site("pinch_site"))

    _add_can(
        world, condition, np.random.default_rng(can_seed),
        can_pos, can_quat, can_label_yaw(can_seed),
    )

    # Static top-down overhead camera, a ceiling-mounted RGB-D looking straight down. Mounted
    # high (real ceiling-Kinect standoff) with a narrow FOV so the small gripper does not
    # occlude the label the robot presents beneath it (a low camera occludes, a real one does
    # not). This camera reads the label-visibility ground truth (RGB legibility + coverage).
    cam = world.worldbody.add_camera(
        name="overhead",
        pos=[can_pos[0], can_pos[1], OVERHEAD_CAM_HEIGHT],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )
    cam.fovy = OVERHEAD_CAM_FOVY

    return world.compile()
