"""RosSource -- real Kinova streams over ROS 2, behind the `SensorSource` seam (Part 1, hardware).

Runs on the Linux capture host with ROS 2. `rclpy` is imported lazily, so this module still imports
on the Mac (schema + `harvest.sensors.base` only, fence-safe). One `LiveRosBridge` node subscribes to
the ROS topics (proprioception + force-torque from `/joint_states`; the Gen3 wrist RGB-D when present)
and caches the latest message. Per-modality `RosSource` wrappers read that cache as `Sample`s, so they
compose with `TSF85Source` (tactile, USB) and `RealSenseSource` (overhead RGB-D, USB) to give all seven
HARVEST streams behind one `record_episode`.

The bridge is a Protocol so tests inject a fake (no ROS, no arm) and the real path stays untested-here.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterator, Optional, Protocol, Sequence

from harvest.sensors.base import SensorSource
from schema.streams import Modality, Sample

# The 7-DOF Gen3 arm joints, in a fixed order, so proprioception/force-torque vectors are stable
# regardless of the order `/joint_states` happens to publish (it interleaves the gripper knuckle).
GEN3_ARM_JOINTS: tuple[str, ...] = (
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
)


class JointStateLike(Protocol):
    """The fields of sensor_msgs/JointState this module reads (also the shape tests fake)."""
    name: Sequence[str]
    position: Sequence[float]
    velocity: Sequence[float]
    effort: Sequence[float]


class RosBridge(Protocol):
    """The seam a `RosSource` reads through. Tests inject a fake; hardware uses `LiveRosBridge`."""
    def start(self) -> None: ...
    def latest_joint_state(self) -> Optional[JointStateLike]: ...
    def stop(self) -> None: ...


def ordered_by_name(js: JointStateLike, joints: Sequence[str], field: str) -> list[float]:
    """Pull one JointState field for `joints` in the given order, tolerant of extra joints
    (like the gripper knuckle) and of publish-order differences. Missing joints raise."""
    index = {n: i for i, n in enumerate(js.name)}
    values = getattr(js, field)
    out: list[float] = []
    for j in joints:
        if j not in index:
            raise KeyError(f"joint {j!r} not in /joint_states names {list(js.name)}")
        out.append(float(values[index[j]]))
    return out


class RosSource:
    """One `SensorSource` for a single modality, backed by a shared `RosBridge` + an extractor.

    `extract(js) -> data` turns the latest JointState into the sample payload (e.g. the 7 arm
    positions or efforts). `stamp_ns(js) -> int` supplies the timestamp; the default reads the wall
    clock at read() time, so all composed sources share one clock for the recorder's skew check.
    """

    def __init__(
        self,
        modality: Modality,
        bridge: RosBridge,
        extract: Callable[[JointStateLike], object],
        notes: str = "",
        poll_timeout_s: float = 5.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.modality = modality
        self._bridge = bridge
        self._extract = extract
        self._notes = notes
        self._poll_timeout_s = poll_timeout_s
        self._clock_ns = clock_ns
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._bridge.start()  # idempotent on the bridge
            self._started = True

    def _await_joint_state(self) -> JointStateLike:
        deadline = self._clock_ns() + int(self._poll_timeout_s * 1e9)
        while True:
            js = self._bridge.latest_joint_state()
            if js is not None:
                return js
            if self._clock_ns() >= deadline:
                raise TimeoutError(
                    f"no /joint_states received within {self._poll_timeout_s}s for {self.modality}"
                )
            time.sleep(0.002)

    def read(self) -> Sample:
        js = self._await_joint_state()
        return Sample(
            modality=self.modality,
            timestamp_ns=self._clock_ns(),
            data=self._extract(js),
            notes=self._notes,
        )

    def stream(self) -> Iterator[Sample]:
        while True:
            yield self.read()

    def stop(self) -> None:
        self._started = False  # the shared bridge is stopped by whoever owns it


def proprioception_source(bridge: RosBridge, joints: Sequence[str] = GEN3_ARM_JOINTS) -> RosSource:
    """PROPRIOCEPTION = the 7 arm joint positions, in `GEN3_ARM_JOINTS` order."""
    return RosSource(
        Modality.PROPRIOCEPTION, bridge,
        extract=lambda js: ordered_by_name(js, joints, "position"),
        notes="arm joint positions from /joint_states",
    )


def force_torque_source(bridge: RosBridge, joints: Sequence[str] = GEN3_ARM_JOINTS) -> RosSource:
    """FORCE_TORQUE derived from the joint effort field (F3, joint-torque source recorded in notes).
    Swap to a wrist 6-axis F/T topic here if the arm carries one."""
    return RosSource(
        Modality.FORCE_TORQUE, bridge,
        extract=lambda js: ordered_by_name(js, joints, "effort"),
        notes="joint-torque-derived (F3); no wrist F/T add-on",
    )


class LiveRosBridge:
    """Real bridge: a lazy-`rclpy` node subscribing to `/joint_states`, spun on a daemon thread.
    Import-safe off-box (rclpy only touched in start())."""

    def __init__(self, joint_states_topic: str = "/joint_states", node_name: str = "harvest_ros_bridge") -> None:
        self._topic = joint_states_topic
        self._node_name = node_name
        self._latest: Optional[JointStateLike] = None
        self._lock = threading.Lock()
        self._node = None
        self._thread: Optional[threading.Thread] = None
        self._rclpy = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        import rclpy  # lazy, box-only
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(self._node_name)
        self._node.create_subscription(JointState, self._topic, self._on_joint_state, 10)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        self._started = True

    def _on_joint_state(self, msg: JointStateLike) -> None:
        with self._lock:
            self._latest = msg

    def _spin(self) -> None:
        try:
            self._rclpy.spin(self._node)
        except Exception:
            pass  # shutdown races on stop() are benign

    def latest_joint_state(self) -> Optional[JointStateLike]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        if not self._started:
            return
        try:
            if self._node is not None:
                self._node.destroy_node()
        finally:
            self._started = False
