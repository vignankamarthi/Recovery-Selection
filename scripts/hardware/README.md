# Hardware bench bring-up playbook

The ordered steps for the Linux capture-host session (arm + tactile + camera). Hardware facts + wiring
live in `HARDWARE.md`; this is the runnable order of operations. Do the streams one at a time and confirm
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
- `ros2_kortex` up; confirm proprioception streams and 2F-85 gripper open/close through the Kinova
  controller (the tactile path does NOT control the gripper).

## 3. Cameras
- Overhead: mount fixed + straight down. Prototype with a USB webcam via `CameraSource` (OpenCV path).
- Calibrate eye-to-hand with ArUco (`harvest/vision/aruco_calibration.py`) so overhead pixels map to
  table coords.
- Real label read: use `CAMPBELL_RED_SPEC` from `harvest/vision/label_visibility.py` (HSV red), NOT the
  sim near-white default. Retune the HSV bands under the actual camera + lighting.

## 4. Seven-stream sanity (before ANY collection, PLAN 3.6)
- Every stream present, at its expected rate, time-aligned (the recorder's timestamp-consistency check),
  no dropouts, sane values. A flaky USB read or a GPU-starved depth stream silently poisons the dataset.
- Then a small record -> io round-trip to confirm episodes persist.
