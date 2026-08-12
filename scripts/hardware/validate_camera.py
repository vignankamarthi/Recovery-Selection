"""Bench validation for the overhead RealSense D435i stream.

Run on the Linux capture host with the D435i plugged into a USB-3 port. It confirms our
`RealSenseSource` opens the device and delivers a sane color + aligned-depth pair off one camera:
warm up a few framesets (so auto-exposure settles), then read color (RGB) and depth (uint16 mm) and
print a per-stream summary. This checks the pipeline and the device, not the aim. The overhead framing
and the label-read tuning come after the camera is mounted and ArUco-calibrated.

Prereqs: `pip install pyrealsense2`. USB-3 port (depth needs the bandwidth).

Usage:
    python scripts/hardware/validate_camera.py                 # 30-frame warmup, print summary
    python scripts/hardware/validate_camera.py --save-npy /tmp # also dump color/depth as .npy to pull back
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from harvest.sensors.camera import RealSenseSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the RealSense D435i color + depth decode on real hardware.")
    ap.add_argument("--warmup", type=int, default=30, help="framesets to poll before reading (auto-exposure settle)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--save-npy", default=None, help="directory to write color.npy + depth.npy for offline viewing")
    args = ap.parse_args()

    src = RealSenseSource(width=args.width, height=args.height, fps=args.fps)
    try:
        src.start()
    except Exception as e:
        print(f"FAILED to open the D435i: {e}")
        print("Check: `pip install pyrealsense2`, a USB-3 port, `lsusb` shows 8086:0b3a, no other process holds it.")
        return 1

    try:
        for _ in range(max(1, args.warmup)):
            src.poll()
        color = src.color()
        depth = src.depth()
    finally:
        src.stop()

    ok = True
    print("=== color ===")
    print(f"shape {color.shape} dtype {color.dtype} "
          f"mean RGB [{color[..., 0].mean():.0f}, {color[..., 1].mean():.0f}, {color[..., 2].mean():.0f}] "
          f"range [{color.min()}, {color.max()}]")
    if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        print("  WARN: expected HxWx3 uint8 RGB")
        ok = False

    print("=== depth (uint16, mm) ===")
    valid = depth > 0
    cov = 100.0 * valid.mean()
    vmin = int(depth[valid].min()) if valid.any() else 0
    vmax = int(depth[valid].max()) if valid.any() else 0
    print(f"shape {depth.shape} dtype {depth.dtype} valid {cov:.1f}% "
          f"range(valid) [{vmin}, {vmax}] mm (~{vmin/1000:.2f}..{vmax/1000:.2f} m)")
    if depth.dtype != np.uint16:
        print("  WARN: expected uint16 depth")
        ok = False
    if cov < 5.0:
        print("  WARN: very low valid-depth coverage (low-texture scene or too-close/too-far), retune when mounted")

    if args.save_npy:
        np.save(os.path.join(args.save_npy, "color.npy"), color)
        np.save(os.path.join(args.save_npy, "depth.npy"), depth)
        print(f"wrote color.npy + depth.npy to {args.save_npy}")

    print("OK, the D435i streams color + aligned depth through RealSenseSource." if ok
          else "Issues above, see WARN lines.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
