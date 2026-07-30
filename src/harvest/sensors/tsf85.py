"""TSF-85 tactile fingertips as a HARVEST SensorSource (hardware, Part 1).

The Robotiq TSF-85 is a PASSIVE USB2 sensor (no ROS, no external power). This module wraps
Robotiq's vendored USB frame parser (`vendor/robotiq_tsf85_protocol.py`) and exposes the
fingertips behind the `SensorSource` seam, so the recorder captures tactile exactly like
`MockSource` / `SimSource`. It emits `Modality.TACTILE` Samples whose `data` is a `TactileFrame`
(the 7x4 pressure grid + dynamic/vibration + IMU, for both fingers, the log-everything decision).

numpy lives here, never in schema. Serial I/O is injected as a `transport`, so the whole decode
+ calibration path is testable with synthetic frames and no hardware. At the bench, the default
`SerialTransport` auto-detects the device by USB VID:PID and streams over 115200 8N1; validate the
decode against Robotiq's `quick_connect.py` on the real device before trusting it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Iterator, List, Optional, Protocol, runtime_checkable

import numpy as np

from harvest.sensors.vendor import robotiq_tsf85_protocol as rp
from schema.streams import Modality, Sample

NUM_FINGERS = rp.NUM_FINGERS            # 2
TAXEL_ROWS, TAXEL_COLS = 7, 4           # 28 taxels, row-major (vendor: "7 rows, 4 columns")
PRESSURE_HZ = 60.0                      # the pressure grid rate (dynamic + IMU are 1000 Hz)
DEFAULT_BAUD = 115200
# USB identifiers the fingertip hub enumerates as (Robotiq / Cypress bridge).
ROBOTIQ_VID_PIDS = ((0x16D0, 0x14CC), (0x04B4, 0xF232))


@dataclass(frozen=True)
class TactileFrame:
    """One tactile snapshot for both fingers. Axis 0 is the finger index. Values are RAW device
    units (pressure = CDC counts ~9000-30000 baseline; dynamic = ADC +/-32767; accel/gyro/mag =
    int16; temperature = int16), calibration is applied separately by `TactileCalibration`."""

    pressure: np.ndarray            # (NUM_FINGERS, 7, 4) float
    dynamic: np.ndarray             # (NUM_FINGERS,) float
    accel: np.ndarray               # (NUM_FINGERS, 3) float
    gyro: np.ndarray                # (NUM_FINGERS, 3) float
    mag: np.ndarray                 # (NUM_FINGERS, 3) float
    temperature: np.ndarray         # (NUM_FINGERS,) float
    device_timestamp: np.ndarray    # (NUM_FINGERS,) int64 (device clock, per finger)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Flat dict of named arrays, the form the io layer serializes."""
        return {
            "pressure": self.pressure, "dynamic": self.dynamic, "accel": self.accel,
            "gyro": self.gyro, "mag": self.mag, "temperature": self.temperature,
            "device_timestamp": self.device_timestamp,
        }


def frame_from_sensor_data(sd: "rp.SensorData") -> TactileFrame:
    """Snapshot the vendor parser's accumulated `SensorData` into a `TactileFrame`."""
    P = np.zeros((NUM_FINGERS, TAXEL_ROWS, TAXEL_COLS), float)
    D = np.zeros(NUM_FINGERS, float)
    A = np.zeros((NUM_FINGERS, 3), float)
    G = np.zeros((NUM_FINGERS, 3), float)
    M = np.zeros((NUM_FINGERS, 3), float)
    T = np.zeros(NUM_FINGERS, float)
    TS = np.zeros(NUM_FINGERS, np.int64)
    for i, f in enumerate(sd.fingers[:NUM_FINGERS]):
        st = np.asarray(f.static_tactile, dtype=float)
        if st.size == TAXEL_ROWS * TAXEL_COLS:
            P[i] = st.reshape(TAXEL_ROWS, TAXEL_COLS)
        D[i] = float(f.dynamic_tactile)
        A[i] = np.asarray(f.accelerometer, dtype=float)[:3]
        G[i] = np.asarray(f.gyroscope, dtype=float)[:3]
        M[i] = np.asarray(f.magnetometer, dtype=float)[:3]
        T[i] = float(f.temperature)
        TS[i] = int(f.timestamp)
    return TactileFrame(P, D, A, G, M, T, TS)


@dataclass(frozen=True)
class TactileCalibration:
    """Per-taxel pressure calibration (per finger). `bias` is subtracted; the optional `gain` is a
    coarse per-taxel scale toward a target count (Robotiq's "align to 500 CDC counts"). Raw taxel
    response is non-linear, so gain is an approximation, multi-point calibration is a later option."""

    bias: np.ndarray                        # (NUM_FINGERS, 7, 4)
    gain: Optional[np.ndarray] = None       # (NUM_FINGERS, 7, 4) or None

    @classmethod
    def zero(cls) -> "TactileCalibration":
        return cls(bias=np.zeros((NUM_FINGERS, TAXEL_ROWS, TAXEL_COLS)))

    @classmethod
    def from_baseline(cls, frames: List[TactileFrame]) -> "TactileCalibration":
        """Per-taxel bias = mean pressure over no-load baseline frames (gripper open). This is the
        "cycle baseline" the guide recommends recording right before each grasp to cancel drift."""
        if not frames:
            raise ValueError("need at least one baseline frame")
        return cls(bias=np.stack([f.pressure for f in frames]).mean(axis=0))

    def with_sensitivity(
        self, no_load: List[TactileFrame], max_load: List[TactileFrame], target: float = 500.0
    ) -> "TactileCalibration":
        """Add a per-taxel gain that maps each taxel's (max_load - no_load) span to `target` counts."""
        lo = np.stack([f.pressure for f in no_load]).mean(axis=0)
        hi = np.stack([f.pressure for f in max_load]).mean(axis=0)
        span = np.clip(hi - lo, 1e-6, None)
        return replace(self, gain=target / span)

    def apply(self, frame: TactileFrame) -> TactileFrame:
        """Return `frame` with bias subtracted (and gain applied if present) on the pressure grid."""
        p = frame.pressure - self.bias
        if self.gain is not None:
            p = p * self.gain
        return replace(frame, pressure=p)


@runtime_checkable
class Transport(Protocol):
    """The minimal byte pipe TSF85Source needs. `SerialTransport` is the real one; tests inject a fake."""

    def read(self, n: int) -> bytes: ...
    def write(self, data: bytes) -> int: ...
    def close(self) -> None: ...


def _autodetect_port(list_ports_module) -> Optional[str]:
    """Return the device path of the first port matching a Robotiq/Cypress VID:PID, or None."""
    for p in list_ports_module.comports():
        vid, pid = getattr(p, "vid", None), getattr(p, "pid", None)
        if vid is not None and (vid, pid) in ROBOTIQ_VID_PIDS:
            return p.device
    return None


class SerialTransport:
    """Real USB-serial transport (lazy `pyserial`, so importing this module needs no serial dep)."""

    def __init__(self, port: Optional[str] = None, baud: int = DEFAULT_BAUD, timeout: float = 0.05):
        import serial                                   # lazy: only when actually opening hardware
        from serial.tools import list_ports
        if port is None:
            port = _autodetect_port(list_ports)
            if port is None:
                raise RuntimeError(
                    "no TSF-85 found (expected USB VID:PID 16D0:14CC or 04B4:F232). "
                    "Check the hub LED is solid green and the cable is seated."
                )
        self._ser = serial.Serial(port, baud, timeout=timeout)

    def read(self, n: int) -> bytes:
        return self._ser.read(n)

    def write(self, data: bytes) -> int:
        return self._ser.write(data)

    def close(self) -> None:
        self._ser.close()


class TSF85Source:
    """A `SensorSource` over the TSF-85 fingertips. Reads USB frames through Robotiq's parser and
    emits the latest `TactileFrame` per `read()`, matching the recorder's latest-sample model."""

    modality = Modality.TACTILE

    def __init__(
        self,
        transport: Optional[Transport] = None,
        *,
        port: Optional[str] = None,
        baud: int = DEFAULT_BAUD,
        autosend_period_ms: int = 1,
        calibration: Optional[TactileCalibration] = None,
        read_chunk: int = 64,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._transport = transport
        self._port = port
        self._baud = baud
        self._autosend_period_ms = autosend_period_ms
        self._calib = calibration
        self._read_chunk = read_chunk
        self._clock = clock
        self._parser = rp.UsbPacketParser()
        self._stopped = False

    def start(self) -> None:
        """Open the transport (if not injected) and ask the device to auto-stream sensor frames."""
        if self._transport is None:
            self._transport = SerialTransport(self._port, self._baud)
        self._transport.write(self._parser.create_autosend_command(self._autosend_period_ms))
        self._stopped = False

    def set_calibration(self, calib: Optional[TactileCalibration]) -> None:
        self._calib = calib

    def collect_baseline(self, n: int) -> TactileCalibration:
        """Read `n` frames (gripper should be OPEN / no load) and return the bias calibration."""
        frames = [self._read_frame() for _ in range(n)]
        return TactileCalibration.from_baseline(frames)

    def _pump_one_frame(self) -> bool:
        """Read bytes until at least one packet parses (updating the parser's accumulated state)."""
        if self._transport is None:
            raise RuntimeError("TSF85Source.start() not called")
        got = False
        while not got and not self._stopped:
            data = self._transport.read(self._read_chunk)
            if not data:
                break                                   # timeout / no data -> return latest state
            for pkt in self._parser.feed_bytes(data):
                self._parser.parse_sensor_packet(pkt)
                got = True
        return got

    def _read_frame(self) -> TactileFrame:
        self._pump_one_frame()
        frame = frame_from_sensor_data(self._parser.get_sensor_data())
        return self._calib.apply(frame) if self._calib is not None else frame

    def read(self) -> Sample:
        """Return the latest tactile snapshot as a `Modality.TACTILE` Sample."""
        frame = self._read_frame()
        return Sample(modality=Modality.TACTILE, timestamp_ns=int(self._clock()), data=frame, notes="tsf85")

    def stream(self) -> Iterator[Sample]:
        while not self._stopped:
            yield self.read()

    def stop(self) -> None:
        self._stopped = True
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:
                pass
