"""Modality definitions and per-modality sample types (the contract side of sensing).

Stdlib-only. The corrected HARVEST sensor suite (flags F3, F4 reconciled): the wrist
camera is RGB-D, so wrist depth is named explicitly and overhead depth is a distinct
viewpoint. The F/T stream's source (wrist add-on vs joint-torque-derived) is recorded
in `notes` until decided. Stubs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    RGB_OVERHEAD = "rgb_overhead"
    RGB_WRIST = "rgb_wrist"
    DEPTH_WRIST = "depth_wrist"      # the Gen3 wrist module is RGB-D (F4)
    DEPTH_OVERHEAD = "depth_overhead"  # distinct viewpoint, justified not redundant (F4)
    TACTILE = "tactile"             # Robotiq TSF-85 fingertips, 28-taxel array (F2)
    FORCE_TORQUE = "force_torque"   # source recorded in notes (F3)
    PROPRIOCEPTION = "proprioception"


@dataclass(frozen=True)
class Sample:
    """A single timestamped sample from one modality. `data` is opaque at the schema
    level (an array handle, a path, a tensor ref); the I/O layer knows the encoding."""

    modality: Modality
    timestamp_ns: int
    data: object
    notes: str = ""


@dataclass(frozen=True)
class StreamSpec:
    """Declares an expected stream and its nominal rate, for sync verification."""

    modality: Modality
    nominal_hz: float
    required: bool = True
