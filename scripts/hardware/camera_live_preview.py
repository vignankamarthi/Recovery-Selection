"""Live overhead-aim preview for the D435i, shown on the capture host's local screen.

Opens a window with the live color stream plus a center crosshair and a thirds grid, so the camera can
be positioned top-down over the presentation spot without snapshot round-trips. Color-only, for a smooth
high-rate preview. Press q or ESC (or close the window) to quit.

Run it ON the laptop's display, e.g. over SSH: `DISPLAY=:1 .venv/bin/python scripts/hardware/camera_live_preview.py`
Prereqs: `pip install pyrealsense2 opencv-python`.

`--rotate {0,90,180,270}` rotates the image in software (both the view and the saved snapshot), for when a
mount forces the camera sideways and physical de-rotation is awkward. `--snap-path` gets a clean frame
written every `--snap-every` frames, so a remote operator can watch aiming progress without stealing the device.
"""
from __future__ import annotations

import argparse

import numpy as np
import pyrealsense2 as rs
import cv2

_ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def main() -> int:
    ap = argparse.ArgumentParser(description="Live overhead-aim preview for the D435i.")
    ap.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    ap.add_argument("--snap-path", default="/tmp/aim_latest.png", help="clean frame written here periodically")
    ap.add_argument("--snap-every", type=int, default=30, help="write the snapshot every N frames")
    args = ap.parse_args()

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)

    win = "D435i overhead aim  (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    n = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data()).copy()
            if _ROT[args.rotate] is not None:
                img = cv2.rotate(img, _ROT[args.rotate])
            if n % max(1, args.snap_every) == 0:
                cv2.imwrite(args.snap_path, img)   # clean frame, before overlays
            n += 1
            h, w = img.shape[:2]
            # rule-of-thirds grid
            for i in (1, 2):
                cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (70, 70, 70), 1)
                cv2.line(img, (0, h * i // 3), (w, h * i // 3), (70, 70, 70), 1)
            # center crosshair (aim the presentation spot here, straight down)
            cx, cy = w // 2, h // 2
            cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 44, 2)
            cv2.circle(img, (cx, cy), 7, (0, 255, 0), 2)
            cv2.putText(img, "center the presentation spot, straight down", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
