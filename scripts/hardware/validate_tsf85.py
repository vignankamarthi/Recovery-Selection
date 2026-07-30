"""Bench validation for the TSF-85 tactile stream.

Run on the Linux capture host with the fingertips wired and the USB plugged in. It confirms our
`TSF85Source` decodes the real device: auto-detect the USB port, optionally record a no-load baseline
(gripper OPEN), then stream frames and print a sane-value summary per finger. Cross-check these numbers
against Robotiq's own `quick_connect.py` / web viewer on the same device, they should track. If the
numbers respond when you press the pads, the decode is good.

Prereqs: `pip install pyserial`. Hub LED solid green (powered) + each finger LED solid green (valid
packets), see HARDWARE.md section 3.6.

Usage:
    python scripts/hardware/validate_tsf85.py                 # stream 200 frames
    python scripts/hardware/validate_tsf85.py --baseline 500  # bias-subtract first (gripper OPEN)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from harvest.sensors.tsf85 import TSF85Source  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the TSF-85 tactile decode on real hardware.")
    ap.add_argument("--frames", type=int, default=200, help="how many frames to print")
    ap.add_argument("--port", default=None, help="serial port (default: auto-detect by USB VID:PID)")
    ap.add_argument("--baseline", type=int, default=0,
                    help="record N no-load baseline frames first (keep the gripper OPEN) and bias-subtract")
    args = ap.parse_args()

    src = TSF85Source(port=args.port)
    try:
        src.start()
    except Exception as e:
        print(f"FAILED to open the TSF-85: {e}")
        print("Check: `pip install pyserial`, hub LED solid green, USB seated, VID:PID 16D0:14CC / 04B4:F232.")
        print("Also try Robotiq's quick_connect.py to confirm the device itself streams.")
        return 1

    if args.baseline:
        print(f"recording {args.baseline} no-load baseline frames -- keep the gripper OPEN ...")
        src.set_calibration(src.collect_baseline(args.baseline))
        print("baseline captured; pressure is now bias-subtracted.")

    print("streaming (Ctrl-C to stop). Press the pads and watch the pressure sums move.")
    n = 0
    try:
        for s in src.stream():
            f = s.data
            p, d, a = f.pressure, f.dynamic, f.accel
            print(f"[{n:4d}] F0 press sum {p[0].sum():8.0f} rng[{p[0].min():6.0f},{p[0].max():6.0f}] "
                  f"dyn {d[0]:7.0f} acc {a[0].round(0)} | "
                  f"F1 press sum {p[1].sum():8.0f} rng[{p[1].min():6.0f},{p[1].max():6.0f}] dyn {d[1]:7.0f}")
            n += 1
            if n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
    print(f"done, {n} frames. If both finger LEDs were solid green and the sums responded to pressure, "
          f"the decode matches the device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
