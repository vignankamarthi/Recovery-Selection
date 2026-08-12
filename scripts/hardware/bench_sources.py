"""Shared builder for the live HARVEST bench sensor sources (Linux capture host only).

Composes the streams this bench exposes into a `{key: SensorSource}` dict plus a `cleanup` callback
that stops every source and shuts the ROS bridge + rclpy down cleanly (so the process exits without the
'terminate called' noise you get when the RealSense pipeline and the rclpy node are torn down at exit).

  proprioception  <- /joint_states arm positions   (RosSource)
  force_torque    <- /joint_states arm efforts      (RosSource, F3 joint-torque-derived)
  rgb_overhead    <- Intel RealSense D435i color    (CameraSource + RealSenseGrabber)
  depth_overhead  <- Intel RealSense D435i depth     (CameraSource + RealSenseGrabber, one shared device)
  tactile         <- Robotiq TSF-85 fingertips       (TSF85Source over USB serial)

rclpy / pyrealsense2 / pyserial are all imported lazily inside the source classes, so importing THIS
module off-box (Mac) stays safe.
"""
from __future__ import annotations

from typing import Callable, Optional


def find_tactile_port() -> Optional[str]:
    """Locate the Robotiq TSF-85 serial port (VID 0x16d0, PID 0x14cc), else the first /dev/ttyACM*."""
    try:
        from serial.tools import list_ports
    except Exception:
        return None
    ports = list(list_ports.comports())
    for p in ports:
        if (p.vid, p.pid) == (0x16D0, 0x14CC):
            return p.device
    acms = sorted(p.device for p in ports if "ACM" in (p.device or ""))
    return acms[0] if acms else None


def build_bench_sources(
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    tactile_port: Optional[str] = None,
    with_camera: bool = True,
    with_tactile: bool = True,
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[dict, Callable[[], None]]:
    """Return `(sources, cleanup)`. `sources` is ready for `record_episode` / `record_ticks`; `cleanup`
    stops every source, the shared ROS bridge, and rclpy. The arm is only READ; nothing commands motion."""
    from harvest.sensors.camera import CameraSource, RealSenseGrabber, RealSenseSource
    from harvest.sensors.ros_source import (
        LiveRosBridge,
        force_torque_source,
        proprioception_source,
    )
    from schema.streams import Modality

    sources: dict = {}

    # ORDER MATTERS for tick sync: the camera is read FIRST because its wait_for_frames blocks up to
    # 1/fps. Reading it first means the fast streams (arm, tactile) sample right after the frame lands,
    # so all five per-tick timestamps cluster within a few ms instead of a full frame period apart.
    if with_camera:
        # align=False: skip the depth->color align block (it corrupts under USB-2 + concurrent-sensor
        # contention on this bench). Raw depth is stored in the depth sensor frame; align offline if needed.
        rs = RealSenseSource(width=width, height=height, fps=fps, align=False)
        # ONE poll per tick backs both streams: the color grabber (read first) polls a fresh frameset;
        # the depth grabber follows and serves the cached depth from that SAME frameset. Halves the USB
        # reads on the shared USB-2 bus and pins color/depth to one frameset.
        sources["rgb_overhead"] = CameraSource(
            RealSenseGrabber("color", source=rs, poll_on_read=True),
            modality=Modality.RGB_OVERHEAD, notes="d435i-overhead",
        )
        sources["depth_overhead"] = CameraSource(
            RealSenseGrabber("depth", source=rs, poll_on_read=False),
            modality=Modality.DEPTH_OVERHEAD, notes="d435i-overhead (same frameset as rgb_overhead)",
        )
        log(f"  + rgb_overhead, depth_overhead  <- D435i {width}x{height}@{fps} (one poll/tick, no align)")

    bridge = LiveRosBridge()
    sources["proprioception"] = proprioception_source(bridge)
    sources["force_torque"] = force_torque_source(bridge)
    log("  + proprioception, force_torque  <- /joint_states (LiveRosBridge)")

    if with_tactile:
        port = tactile_port or find_tactile_port()
        if port is None:
            log("  ! tactile port not found (no VID:PID 16d0:14cc, no /dev/ttyACM*); skipping tactile")
        else:
            from harvest.sensors.tsf85 import TSF85Source

            sources["tactile"] = TSF85Source(port=port)
            log(f"  + tactile  <- TSF-85 on {port}")

    def cleanup() -> None:
        for s in sources.values():
            try:
                s.stop()
            except Exception:
                pass
        try:
            bridge.stop()
        except Exception:
            pass
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    return sources, cleanup
