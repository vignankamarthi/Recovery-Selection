#!/usr/bin/env python3
"""Live teleoperation session capture with an ON/OFF button (Linux capture host).

Records ALL live HARVEST streams SYNCHRONIZED per tick while you teleoperate, so the sensor readings
line up in time with what you do. It is stop-controlled:

  1. run it, wait for the READY prompt
  2. press ENTER to START recording
  3. teleoperate and complete the pick-and-reorient task
  4. press ENTER again to STOP  (Ctrl-C also stops cleanly)

The captured raw streams are written as ONE .npz (the artifact to send Padir next to your operation
video). The arm is driven by YOUR teleop process; this one only READS sensors, it never commands motion.

  source /opt/ros/humble/setup.bash
  PYTHONPATH=src:$PYTHONPATH python3 scripts/hardware/capture_session.py --out ~/harvest_sessions

Headless self-test (no keyboard, auto-stop after N seconds):
  ... capture_session.py --seconds 3
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                              # bench_sources (same dir)
sys.path.insert(0, str(_HERE.parents[1] / "src"))          # harvest / schema

from bench_sources import build_bench_sources


def _summarize(rec, duration_s: float) -> bool:
    """Print a per-stream + synchronization summary. Return True if the capture looks healthy."""
    import numpy as np

    keys = list(rec.streams.keys())
    print("\n== captured streams ==")
    ok = True
    n_ticks = min((len(v) for v in rec.streams.values()), default=0)
    for key in keys:
        samples = rec.streams[key]
        ts = [s.timestamp_ns for s in samples]
        hz = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
        first = samples[0].data
        if hasattr(first, "as_arrays"):
            shp = {k: getattr(v, "shape", None) for k, v in first.as_arrays().items()}
        else:
            a = np.asarray(first)
            shp = f"{a.shape} {a.dtype}"
        monotonic = all(b > a for a, b in zip(ts, ts[1:]))
        ok = ok and monotonic and len(samples) == n_ticks
        print(f"  {key:16s} n={len(samples):4d}  {hz:5.1f} Hz  {'mono' if monotonic else 'NON-MONO(!)'}  data={shp}")

    # Per-tick cross-stream spread: for each tick, how far apart (ns) are the streams' timestamps.
    # This is the real synchronization quality of the synchronized recorder (want tens of ms, not ~1 s).
    if len(keys) > 1 and n_ticks > 0:
        spreads = []
        for i in range(n_ticks):
            tv = [rec.streams[k][i].timestamp_ns for k in keys]
            spreads.append(max(tv) - min(tv))
        spreads.sort()
        med_ms = spreads[len(spreads) // 2] / 1e6
        max_ms = spreads[-1] / 1e6
        print(f"\n  per-tick cross-stream spread: median {med_ms:.1f} ms, max {max_ms:.1f} ms  ({n_ticks} ticks)")
    print(f"  session duration: {duration_s:.1f} s, streams: {len(keys)} / 7 (wrist rgb+depth = Gen3 vision module, not wired into capture yet)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path.home() / "harvest_sessions"))
    ap.add_argument("--rate-hz", type=float, default=20.0, help="target capture rate")
    ap.add_argument("--episode-id", default=None)
    ap.add_argument("--can-id", default="bench-can-A")
    ap.add_argument("--condition", default="NOMINAL", help="ConditionClass name")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15, help="D435i fps; 15 is USB-2-comfortable, 30 needs USB-3")
    ap.add_argument("--warmup", type=int, default=10, help="reads discarded before recording (pipeline warm-up)")
    ap.add_argument("--seconds", type=float, default=None, help="auto-stop after N s (non-interactive self-test)")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-tactile", action="store_true")
    args = ap.parse_args()

    from harvest.io.flat_npz_adapter import write_episode_streams, write_index
    from harvest.recorder.recorder import record_ticks
    from schema.episode import ConditionClass, Episode

    print("== HARVEST live session capture (arm driven by YOUR teleop; this process only READS) ==")
    sources, cleanup = build_bench_sources(
        width=args.width, height=args.height, fps=args.fps,
        with_camera=not args.no_camera, with_tactile=not args.no_tactile, log=print,
    )

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    try:
        for s in sources.values():
            s.start()

        # Liveness probe: FAIL FAST and loud if any source cannot produce one good sample. This
        # separates a permanent fault (unsupported camera mode, unplugged sensor, wrong port) from
        # the transient hiccups the tolerant record loop is allowed to drop. Without it, a permanent
        # error masquerades as "every tick dropped" and writes an empty capture.
        print("\nliveness probe (one real read per source) ...")
        for key, s in sources.items():
            try:
                s.read()
                print(f"  {key:16s} OK")
            except Exception as exc:
                print(f"  {key:16s} FAILED: {exc}")
                print(f"\nABORT: source {key!r} is not producing data. Fix it before recording.")
                if key in ("rgb_overhead", "depth_overhead"):
                    print("  (camera: 640x480 is the known-good D435i mode on this USB-2 link; "
                          "424x240 / 848x480 are NOT supported and give 'Couldn't resolve requests')")
                return 2

        # Warm-up: discard a few more reads so the camera auto-exposure settles before the record clock.
        print(f"warming up ({args.warmup} discarded reads) ...")
        for _ in range(args.warmup):
            for s in sources.values():
                try:
                    s.read()
                except Exception:
                    pass

        if args.seconds is not None:
            print(f"\n[non-interactive] recording for {args.seconds:.1f} s ...")
            threading.Timer(args.seconds, stop_event.set).start()
        else:
            try:
                input("\n>>> READY. Press ENTER to START recording <<<")
            except EOFError:
                print("\nno interactive terminal (stdin closed). Use --seconds N for a headless run.")
                return 4
            print(">>> RECORDING. Teleoperate, complete the task, then press ENTER (or Ctrl-C) to STOP <<<")

            def _wait_enter() -> None:
                try:
                    input()
                except EOFError:
                    pass
                stop_event.set()

            threading.Thread(target=_wait_enter, daemon=True).start()

        episode = Episode(
            episode_id=args.episode_id or f"session_{int(time.time())}",
            can_id=args.can_id,
            condition=ConditionClass[args.condition.upper()],
            metadata={"context": "live teleop session", "arm_state": "teleop-driven", "capture": "record_ticks"},
        )

        drops = {"n": 0}
        drop_causes: dict = {}

        def _on_drop(exc: Exception) -> None:
            drops["n"] += 1
            key = f"{type(exc).__name__}: {str(exc)[:60]}"
            drop_causes[key] = drop_causes.get(key, 0) + 1

        t0 = time.monotonic()
        rec = record_ticks(
            episode, sources, should_stop=stop_event.is_set, rate_hz=args.rate_hz,
            tolerate_errors=True, on_drop=_on_drop,
        )
        duration_s = time.monotonic() - t0
    finally:
        cleanup()

    if drops["n"]:
        print(f"\n  note: {drops['n']} tick(s) dropped on transient sensor hiccups (session kept alive)")
        for cause, count in sorted(drop_causes.items(), key=lambda kv: -kv[1]):
            print(f"    {count:4d} x  {cause}")

    # Empty-capture guard: never write a hollow npz. If a stream never committed a sample the capture
    # is not real, so report it instead of pretending it succeeded.
    empty = [k for k, v in rec.streams.items() if not v]
    if empty or not rec.streams:
        print(f"\nABORT: no data captured for streams {empty or list(rec.streams)}. Nothing written.")
        return 3

    healthy = _summarize(rec, duration_s)

    out = Path(args.out)
    manifest = write_episode_streams(rec, out)
    write_index([manifest], out)
    npz = out / "data" / f"{rec.episode.episode_id}.npz"
    size_mb = npz.stat().st_size / 1e6 if npz.exists() else 0
    print(f"\n== wrote {npz}  ({size_mb:.1f} MB) ==")
    print(f"  send this .npz to Padir alongside your operation video.")
    print(f"  RESULT: {'OK' if healthy and npz.exists() else 'CHECK OUTPUT'}")
    return 0 if healthy and npz.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
