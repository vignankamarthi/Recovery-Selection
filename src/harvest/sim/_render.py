"""Shared MuJoCo render helper: ONE process-global renderer, closed on model/size change.

A `mujoco.Renderer` holds a GL framebuffer + context that Python GC does not free, so a new
renderer per episode leaked one GL context per episode and once exhausted machine memory on a long
generation run. This module keeps EXACTLY ONE live renderer, keyed on (model identity, height,
width), and closes the previous one before opening a replacement. Both the RGB/depth reads
(`world.render`) and the overhead segmentation label-pixel count (`world.overhead_label_visibility`
and `reorient._overhead_px`) go through it, so there is one renderer, not three. This is load-bearing:
the renderer leak once crashed the machine, so the single-renderer-closed-on-model-change semantics
must be preserved exactly.
"""

from __future__ import annotations

import mujoco
import numpy as np

# The single live renderer. Keyed on model IDENTITY (never id(), the model object is reused across
# episodes) and the (height, width). Replaces the two former module-global caches (`world._RENDER`
# and `reorient._RENDERER`).
_RENDER: dict = {"model": None, "hw": None, "r": None}


def _renderer(model, height: int, width: int) -> "mujoco.Renderer":
    """The process-global renderer for (model, height, width), closing and replacing the previous
    one whenever the model identity or the size changes (frees the old GL context first)."""
    if _RENDER["model"] is not model or _RENDER["hw"] != (height, width):
        if _RENDER["r"] is not None:
            _RENDER["r"].close()                 # free the previous GL context before replacing
        _RENDER["r"] = mujoco.Renderer(model, height, width)
        _RENDER["model"], _RENDER["hw"] = model, (height, width)
    return _RENDER["r"]


def render_camera(model, data, camera: str = "overhead", depth: bool = False,
                  height: int = 96, width: int = 96) -> np.ndarray:
    """One RGB (uint8 HxWx3) or depth (float32 HxW) frame from `camera`."""
    r = _renderer(model, height, width)
    r.update_scene(data, camera=camera)
    if depth:
        r.enable_depth_rendering()
        img = r.render().copy()
        r.disable_depth_rendering()
        return img
    return r.render().copy()


def label_pixel_count(model, data, label_gid: int, camera: str = "overhead",
                      height: int = 200, width: int = 200) -> int:
    """Overhead segmentation pixel count for geom `label_gid` (the exposed-label read)."""
    r = _renderer(model, height, width)
    r.update_scene(data, camera=camera)
    r.enable_segmentation_rendering()
    seg = r.render()
    r.disable_segmentation_rendering()
    return int((seg[..., 0] == label_gid).sum())
