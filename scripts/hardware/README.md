# Hardware bench bring-up playbook

The ordered steps for the Linux capture-host session (arm + tactile + camera). Hardware facts + wiring
live in `HARDWARE.md`. This is the runnable order of operations. Do the streams one at a time and confirm
each before moving on. The goal is a rock-solid rig, not collection volume (real damaged cans wait for
the Maine sprint, PLAN 3.7).

## 0. Prereqs (on the capture host)
- Ubuntu 22.04, native USB + GPU (Austin's setup / the lab river box). Confirm before committing collection.
- Repo pulled. A python with numpy (a venv is fine). ROS2 Humble for the arm/cameras (tactile needs no ROS).

## 1. Tactile (TSF-85) -- USB, NO ROS
1. Plug the hub USB into the host. Confirm the hub LED is SOLID GREEN and each finger LED solid green
   (valid packets). LED table in `HARDWARE.md` section 3.6.
2. `pip install pyserial`.
3. Optional, eyeball the device with Robotiq's own tools (`sensor_quickstart/run_quick_connect.sh` or the
   web viewer at localhost:8080) to see the live heatmap / FFT / IMU.
4. Our decode + cross-check:
   `python scripts/hardware/validate_tsf85.py --baseline 500`
   Press the pads, watch the per-finger pressure sums move. The numbers should track Robotiq's viewer.

## 2. Arm (Kinova Gen3) -- ROS
- Always start the driver with `scripts/hardware/start_arm.sh` (NOT the raw launch). It runs the driver at
  real-time priority, pins the control node to a core, auto-clears the arm fault, activates the twist
  controller, and auto-respawns on a crash. Stability is SOLVED by the lowlatency kernel (HARDWARE.md 2).
- Teleop: `joy_node` -> `teleop_twist_joy` with `scripts/hardware/f310_teleop.yaml` (F310, no deadman, the
  physical kill switch is the safety) -> `twist_controller`. Gripper via `scripts/hardware/gripper_teleop.py`
  (A = close, B = open, grip force = `MAX_EFFORT`, now 35 of a 0-50 ceiling). Restart the gripper node after
  editing `MAX_EFFORT`.
- GOTCHA: a hard fault / e-stop drops the arm OFF the network. Power-cycle the arm to recover (HARDWARE.md 2).

## 3. Cameras
- Overhead: the Intel RealSense D435i, mounted top-down over the presentation spot (matte gray mat
  background). Currently on a USB-2 link, so capture runs 640x480@15 with raw (unaligned) depth. A USB-3
  cable + port restores 30 fps + hardware-aligned depth (HARDWARE.md 5a/6).
- Validate the live device: `python scripts/hardware/validate_camera.py`. Aim it with
  `scripts/hardware/camera_live_preview.py` (a crosshair/grid window + rolling snapshot).
- Calibrate eye-to-hand with ArUco (`harvest/vision/aruco_calibration.py`) once the robot + camera are LOCKED.
- Real label read: use `CAMPBELL_RED_SPEC` from `harvest/vision/label_visibility.py` (HSV red), NOT the sim
  near-white default. Retune the HSV bands under the actual camera + lighting.

## 4. Recording (before ANY collection, PLAN 3.6)
Run everything with the ROS system python and `PYTHONPATH=src:$PYTHONPATH` (bare `PYTHONPATH=src` hides rclpy).
- One-shot sanity of all available streams: `python scripts/hardware/validate_recording.py`. It composites
  `RosSource` (proprioception + force_torque) + `CameraSource` (rgb_overhead + depth_overhead) + `TSF85Source`
  (tactile) into `record_episode`, checks each stream, writes an episode, and confirms the io round-trip.
- Live teleop session (the deliverable): `python scripts/hardware/capture_session.py --out ~/harvest_sessions`.
  It probes each sensor (fail-fast), warms up, then ON/OFF button, press ENTER to START, teleoperate the task,
  press ENTER again (or Ctrl-C) to STOP. Uses `record_ticks` (synchronized per-tick, atomic, drop-tolerant) and
  writes one flat-npz episode. Validated at ~14 Hz, 0 drops, <1 ms per-tick sync across all 5 available streams.
- Current bench = 5 of 7 streams (wrist rgb+depth need the Gen3 wrist module, absent here).
