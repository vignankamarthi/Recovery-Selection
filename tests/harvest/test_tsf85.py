"""Synthetic-frame tests for the TSF-85 tactile SensorSource.

No hardware: packets are built in the real Robotiq wire format (start 0x9A, 4-byte header, a
sensor-type/finger byte, big-endian fields) and fed through an injected fake transport, so the
whole decode + calibration + SensorSource path is exercised end to end. The wire format itself is
Robotiq's vendored parser, this pins our wrapper (frame assembly, reshape, signedness, calibration).
"""
from typing import List, Optional

import numpy as np
import pytest

from harvest.sensors.base import SensorSource
from harvest.sensors.tsf85 import (
    NUM_FINGERS,
    TAXEL_COLS,
    TAXEL_ROWS,
    TactileCalibration,
    TactileFrame,
    TSF85Source,
    _autodetect_port,
)
from harvest.sensors.vendor import robotiq_tsf85_protocol as rp
from schema.streams import Modality, Sample


# --- wire-format packet builders (mirror the vendor encoding) ------------------------------------

def _be16(v: int) -> bytes:
    v &= 0xFFFF
    return bytes([(v >> 8) & 0xFF, v & 0xFF])


def _sensor_byte(type_hi: int, finger: int) -> int:
    return type_hi | ((finger & 0x03) << 2)


def build_packet(finger, *, static=None, dynamic=None, accel=None, gyro=None, mag=None, temp=None) -> bytes:
    body = bytearray()
    if static is not None:
        body.append(_sensor_byte(rp.SENSOR_TYPE_STATIC_TACTILE, finger))
        for x in static:
            body += _be16(x)
    if dynamic is not None:
        body.append(_sensor_byte(rp.SENSOR_TYPE_DYNAMIC_TACTILE, finger))
        body += _be16(dynamic)
    for type_hi, vec in ((rp.SENSOR_TYPE_ACCELEROMETER, accel),
                         (rp.SENSOR_TYPE_GYROSCOPE, gyro),
                         (rp.SENSOR_TYPE_MAGNETOMETER, mag)):
        if vec is not None:
            body.append(_sensor_byte(type_hi, finger))
            for x in vec:
                body += _be16(x)
    if temp is not None:
        body.append(_sensor_byte(rp.SENSOR_TYPE_TEMPERATURE, finger))
        body += _be16(temp)
    return bytes([rp.USB_PACKET_START_BYTE, 0x00, rp.USB_COMMAND_READ_SENSORS, len(body)]) + bytes(body)


class FakeTransport:
    """Feeds queued byte chunks to the source; records writes."""

    def __init__(self, chunks: List[bytes]):
        self._chunks = list(chunks)
        self.writes: List[bytes] = []

    def read(self, n: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        pass


def _source(chunks: List[bytes], calibration: Optional[TactileCalibration] = None) -> TSF85Source:
    src = TSF85Source(FakeTransport(chunks), calibration=calibration, clock=lambda: 0)
    src.start()
    return src


# --- decode ---------------------------------------------------------------------------------------

def test_static_pressure_decodes_and_reshapes():
    vals = list(range(100, 128))                       # 28 known taxels
    src = _source([build_packet(0, static=vals)])
    frame = src.read().data
    assert isinstance(frame, TactileFrame)
    np.testing.assert_array_equal(frame.pressure[0], np.array(vals).reshape(TAXEL_ROWS, TAXEL_COLS))
    np.testing.assert_array_equal(frame.pressure[1], np.zeros((TAXEL_ROWS, TAXEL_COLS)))


def test_dynamic_decodes_signed():
    src = _source([build_packet(0, dynamic=-1234)])
    assert src.read().data.dynamic[0] == -1234         # big-endian int16, two's complement


def test_imu_decodes_signed_vectors():
    src = _source([build_packet(1, accel=[1, -2, 3], gyro=[-10, 20, -30], mag=[100, -200, 300])])
    f = src.read().data
    np.testing.assert_array_equal(f.accel[1], [1, -2, 3])
    np.testing.assert_array_equal(f.gyro[1], [-10, 20, -30])
    np.testing.assert_array_equal(f.mag[1], [100, -200, 300])


def test_both_fingers_populate():
    src = _source([build_packet(0, static=[1] * 28), build_packet(1, static=[2] * 28)])
    src.read()                                         # pump the first packet
    f = src.read().data                                # latest state has both fingers
    assert f.pressure[0].mean() == 1 and f.pressure[1].mean() == 2


def test_packet_split_across_reads_is_reassembled():
    pkt = build_packet(0, static=list(range(28)))
    src = _source([pkt[:5], pkt[5:]])                  # header/body split across two reads
    frame = src.read().data
    np.testing.assert_array_equal(frame.pressure[0], np.arange(28).reshape(TAXEL_ROWS, TAXEL_COLS))


def test_start_sends_autosend_command():
    t = FakeTransport([])
    TSF85Source(t).start()
    assert len(t.writes) == 1 and t.writes[0][0] == rp.USB_PACKET_START_BYTE
    assert t.writes[0][2] == rp.USB_COMMAND_AUTOSEND_SYNC_SENSORS


# --- calibration ----------------------------------------------------------------------------------

def _frame(pressure: np.ndarray) -> TactileFrame:
    z3 = np.zeros((NUM_FINGERS, 3))
    return TactileFrame(pressure, np.zeros(NUM_FINGERS), z3, z3, z3.copy(),
                        np.zeros(NUM_FINGERS), np.zeros(NUM_FINGERS, np.int64))


def test_bias_calibration_zeroes_the_baseline():
    base = np.random.default_rng(0).random((NUM_FINGERS, TAXEL_ROWS, TAXEL_COLS)) * 20000
    calib = TactileCalibration.from_baseline([_frame(base), _frame(base)])
    out = calib.apply(_frame(base))
    np.testing.assert_allclose(out.pressure, 0.0, atol=1e-9)


def test_sensitivity_gain_maps_span_to_target():
    lo = np.zeros((NUM_FINGERS, TAXEL_ROWS, TAXEL_COLS))
    hi = np.full((NUM_FINGERS, TAXEL_ROWS, TAXEL_COLS), 250.0)
    calib = TactileCalibration.zero().with_sensitivity([_frame(lo)], [_frame(hi)], target=500.0)
    out = calib.apply(_frame(hi))                      # span 250 -> gain 2.0 -> 500
    np.testing.assert_allclose(out.pressure, 500.0)


def test_source_applies_calibration_on_read():
    vals = np.full(28, 5000.0)
    calib = TactileCalibration(bias=np.full((NUM_FINGERS, TAXEL_ROWS, TAXEL_COLS), 5000.0))
    src = _source([build_packet(0, static=[5000] * 28)], calibration=calib)
    np.testing.assert_allclose(src.read().data.pressure[0], 0.0)


# --- SensorSource conformance + lifecycle ---------------------------------------------------------

def test_is_a_sensorsource():
    assert isinstance(TSF85Source(FakeTransport([])), SensorSource)
    assert TSF85Source(FakeTransport([])).modality is Modality.TACTILE


def test_read_returns_tactile_sample():
    s = _source([build_packet(0, static=[1] * 28)]).read()
    assert isinstance(s, Sample) and s.modality is Modality.TACTILE and s.notes == "tsf85"


def test_stream_stops():
    src = _source([build_packet(0, static=[1] * 28)])
    it = src.stream()
    next(it)
    src.stop()
    assert list(it) == []


# --- port autodetect ------------------------------------------------------------------------------

class _Port:
    def __init__(self, vid, pid, device):
        self.vid, self.pid, self.device = vid, pid, device


class _ListPorts:
    def __init__(self, ports):
        self._ports = ports

    def comports(self):
        return self._ports


def test_autodetect_matches_robotiq_vid_pid():
    lp = _ListPorts([_Port(0x1234, 0x5678, "/dev/ttyX"), _Port(0x16D0, 0x14CC, "/dev/ttyUSB0")])
    assert _autodetect_port(lp) == "/dev/ttyUSB0"


def test_autodetect_returns_none_when_absent():
    assert _autodetect_port(_ListPorts([_Port(0x1234, 0x5678, "/dev/ttyX")])) is None


# --- io persistence of a TactileFrame (the rich payload survives export/reload) -------------------

def test_tactile_frame_round_trips_through_flat_export(tmp_path):
    from harvest.io.flat_npz_adapter import export_dataset, load_export
    from schema.episode import ConditionClass, Episode, Outcome, RecordedEpisode

    frame = _source([build_packet(0, static=list(range(28)), dynamic=-77,
                                  accel=[1, 2, 3], gyro=[4, 5, 6], mag=[7, 8, 9])]).read().data
    ep = Episode(episode_id="ep-t", can_id="c-1", condition=ConditionClass.BULGE,
                 outcome=Outcome.SUCCESS, stream_keys=("tactile",), labels=[], metadata={})
    rec = RecordedEpisode(episode=ep, streams={"tactile": [Sample(Modality.TACTILE, 500, frame, "tsf85")]})

    export_dataset([rec], tmp_path)
    (back,) = load_export(tmp_path)
    data = back.streams["tactile"][0].data              # structured payload reloads as a dict of arrays
    np.testing.assert_array_equal(data["pressure"][0], np.arange(28).reshape(TAXEL_ROWS, TAXEL_COLS))
    assert data["dynamic"][0] == -77
    np.testing.assert_array_equal(data["accel"][0], [1, 2, 3])
