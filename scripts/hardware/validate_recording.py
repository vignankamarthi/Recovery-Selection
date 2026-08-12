#!/usr/bin/env python3
"""On-box recording validation: compose the real sensors into one `record_episode` and write it.

Runs on the Linux capture host with the arm IDLE (read-only, NO motion is ever commanded). It builds
the live HARVEST streams available on this bench and drives them through the same recorder + io path
the dataset uses:

  proprioception  <- /joint_states arm positions   (RosSource)
  force_torque    <- /joint_states arm efforts      (RosSource, F3 joint-torque-derived)
  rgb_overhead    <- Intel RealSense D435i color    (CameraSource + RealSenseGrabber)
  depth_overhead  <- Intel RealSense D435i depth     (CameraSource + RealSenseGrabber, one shared device)
  tactile         <- Robotiq TSF-85 fingertips       (TSF85Source over USB serial)

That is 5 of the 7 HARVEST streams. The two wrist streams (rgb_wrist, depth_wrist) need a wrist camera
this bench does not have, so they are out of scope here and reported as such.

Run it with the ROS system python so rclpy is importable:

  source /opt/ros/humble/setup.bash
  PYTHONPATH=src python3 scripts/hardware/validate_recording.py

verify=False on the recorder is deliberate: record_episode reads each source sequentially, so the real
streams do NOT start within the 10 ms skew tolerance (that check assumes near-simultaneous starts). This
script measures and reports the real start skew itself, then writes the episode and reads it back to
prove the capture -> io round-trip works end to end on real hardware.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so `harvest`/`schema` import when run from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def find_tactile_port() -> str | None:
    """Locate the Robotiq TSF-85 serial port (VID 0x16d0, PID 0x14cc), else first /dev/ttyACM*."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=5, help="samples per stream")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--tactile-port", default=None, help="override auto-detected TSF-85 serial port")
    ap.add_argument("--no-tactile", action="store_true", help="skip the tactile stream")
    ap.add_argument("--no-camera", action="store_true", help="skip both overhead camera streams")
    ap.add_argument("--out-dir", default=str(Path.home() / "harvest_bench_recording"))
    args = ap.parse_args()

    from harvest.io.flat_npz_adapter import load_export, write_episode_streams
    from harvest.recorder.recorder import record_episode
    from harvest.sensors.camera import CameraSource, RealSenseGrabber, RealSenseSource
    from harvest.sensors.ros_source import (
        LiveRosBridge,
        force_torque_source,
        proprioception_source,
    )
    from schema.episode import ConditionClass, Episode
    from schema.streams import Modality

    print("== HARVEST on-box recording validation (arm IDLE, read-only) ==")

    sources: dict = {}

    # Arm streams: one shared bridge feeds both proprioception and force-torque.
    bridge = LiveRosBridge()
    sources["proprioception"] = proprioception_source(bridge)
    sources["force_torque"] = force_torque_source(bridge)
    print("  + proprioception, force_torque  <- /joint_states (LiveRosBridge)")

    # Overhead RGB-D: ONE D435i backs both streams.
    if not args.no_camera:
        rs = RealSenseSource(width=args.width, height=args.height, fps=args.fps)
        sources["rgb_overhead"] = CameraSource(
            RealSenseGrabber("color", source=rs), modality=Modality.RGB_OVERHEAD, notes="d435i-overhead"
        )
        sources["depth_overhead"] = CameraSource(
            RealSenseGrabber("depth", source=rs), modality=Modality.DEPTH_OVERHEAD, notes="d435i-overhead"
        )
        print(f"  + rgb_overhead, depth_overhead  <- D435i {args.width}x{args.height}@{args.fps}")

    # Tactile: Robotiq TSF-85 over USB serial.
    if not args.no_tactile:
        port = args.tactile_port or find_tactile_port()
        if port is None:
            print("  ! tactile port not found (no VID:PID 16d0:14cc, no /dev/ttyACM*); skipping tactile")
        else:
            from harvest.sensors.tsf85 import TSF85Source

            sources["tactile"] = TSF85Source(port=port)
            print(f"  + tactile  <- TSF-85 on {port}")

    episode = Episode(
        episode_id="bench_smoke_0001",
        can_id="bench-can-A",
        condition=ConditionClass.NOMINAL,
        metadata={"context": "bench recording validation", "arm_state": "idle-read-only"},
    )

    print(f"\nRecording {args.n_samples} samples from {len(sources)} streams ...")
    rec = record_episode(episode, sources, n_samples=args.n_samples, verify=False)

    # Per-stream report: count, modality, payload shape, monotonicity, and the observed start skew.
    print("\n== captured streams ==")
    starts: list[int] = []
    ok = True
    for key, samples in rec.streams.items():
        ts = [s.timestamp_ns for s in samples]
        monotonic = all(b > a for a, b in zip(ts, ts[1:]))
        starts.append(ts[0])
        first = samples[0].data
        if hasattr(first, "as_arrays"):
            shp = {k: getattr(v, "shape", None) for k, v in first.as_arrays().items()}
        else:
            import numpy as np

            a = np.asarray(first)
            shp = f"{a.shape} {a.dtype}"
        mono = "monotonic" if monotonic else "NON-MONOTONIC(!)"
        ok = ok and monotonic and len(samples) == args.n_samples
        print(f"  {key:16s} n={len(samples)} modality={samples[0].modality.value:14s} {mono}  data={shp}")
    if len(starts) > 1:
        skew_ms = (max(starts) - min(starts)) / 1e6
        print(f"\n  cross-stream start skew: {skew_ms:.1f} ms (sequential reads; not the recorder skew check)")

    # io round-trip: write the episode, then reload it and confirm the streams survive.
    out_dir = Path(args.out_dir)
    print(f"\n== writing episode to {out_dir} ==")
    manifest = write_episode_streams(rec, out_dir)
    npz = out_dir / "data" / f"{episode.episode_id}.npz"
    size_kb = npz.stat().st_size / 1024 if npz.exists() else 0
    print(f"  wrote {npz}  ({size_kb:.1f} KB)")

    from harvest.io.flat_npz_adapter import write_index

    write_index([manifest], out_dir)
    reloaded = load_export(out_dir)
    rt = reloaded[0]
    rt_ok = set(rt.streams.keys()) == set(rec.streams.keys()) and all(
        len(rt.streams[k]) == len(rec.streams[k]) for k in rec.streams
    )
    print(f"  reloaded {len(reloaded)} episode(s), streams={sorted(rt.streams.keys())}, round-trip={'OK' if rt_ok else 'MISMATCH'}")

    print("\n== result ==")
    print(f"  streams captured: {len(rec.streams)} / 7 HARVEST streams (wrist rgb+depth need a wrist camera, absent on this bench)")
    verdict = ok and rt_ok and npz.exists()
    print(f"  VALIDATION: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
