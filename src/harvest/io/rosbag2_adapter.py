"""rosbag2 I/O adapter (the storage seam, Phase 1.3).

This is the hardware / ROS2 on-disk boundary: rosbag2 is the canonical lab/ROS2-native format,
and this adapter is the ONLY place that knows it. It is tested for round-trip fidelity but is NOT
on the shipped sim generation path (that writes flat npz via `io/flat_npz_adapter.py`); it is the
seam the real-robot collection uses on hardware. It uses the pure-Python `rosbags` library, so
files are written and read on macOS with NO ROS2 install, and load natively on the lab's Linux
ROS2 Humble machines.

Encoding: one `/episode/meta` topic (std_msgs/String, JSON of the episode metadata and a
per-stream manifest of dtype/shape/notes), and one `/stream/<key>` topic per stream
(std_msgs/UInt8MultiArray carrying each sample's raw array bytes). This preserves dtype,
shape, values, timestamp and notes exactly on round-trip.

Backend (GATE 1) defaults to mcap; sqlite3 is also supported via the same code path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.rosbag2.enums import StoragePlugin
from rosbags.typesys import Stores, get_typestore

from harvest.io._serde import episode_from_dict, episode_to_dict
from schema.episode import RecordedEpisode
from schema.streams import Modality, Sample

_TS = get_typestore(Stores.ROS2_HUMBLE)
_String = _TS.types["std_msgs/msg/String"]
_U8 = _TS.types["std_msgs/msg/UInt8MultiArray"]
_Layout = _TS.types["std_msgs/msg/MultiArrayLayout"]

_META_TOPIC = "/episode/meta"
_STREAM_PREFIX = "/stream/"
_BACKENDS = {"mcap": StoragePlugin.MCAP, "sqlite3": StoragePlugin.SQLITE3}


def write_episode(recorded: RecordedEpisode, path: Path | str, backend: str = "mcap") -> None:
    """Serialize a RecordedEpisode to a rosbag2 bag at `path` (a new directory)."""
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {sorted(_BACKENDS)}")
    path = Path(path)

    meta: dict = {"episode": episode_to_dict(recorded.episode), "streams": {}}
    ordered: dict[str, list[Sample]] = {}
    for key, samples in recorded.streams.items():
        s_sorted = sorted(samples, key=lambda s: s.timestamp_ns)
        ordered[key] = s_sorted
        meta["streams"][key] = {
            "modality": s_sorted[0].modality.value if s_sorted else None,
            "samples": [
                {
                    "dtype": str(np.asarray(s.data).dtype),
                    "shape": list(np.asarray(s.data).shape),
                    "notes": s.notes,
                }
                for s in s_sorted
            ],
        }

    with Writer(path, version=9, storage_plugin=_BACKENDS[backend]) as w:
        cmeta = w.add_connection(_META_TOPIC, _String.__msgtype__, typestore=_TS)
        w.write(cmeta, 0, _TS.serialize_cdr(_String(data=json.dumps(meta)), _String.__msgtype__))
        for key, samples in ordered.items():
            conn = w.add_connection(_STREAM_PREFIX + key, _U8.__msgtype__, typestore=_TS)
            for s in samples:
                raw = np.frombuffer(np.asarray(s.data).tobytes(), dtype=np.uint8)
                msg = _U8(layout=_Layout(dim=[], data_offset=0), data=raw)
                w.write(conn, s.timestamp_ns, _TS.serialize_cdr(msg, _U8.__msgtype__))


def read_episode(path: Path | str) -> RecordedEpisode:
    """Deserialize a rosbag2 bag at `path` back into a RecordedEpisode."""
    path = Path(path)
    meta: dict | None = None
    raws_by_topic: dict[str, list[bytes]] = {}
    ts_by_topic: dict[str, list[int]] = {}

    with Reader(path) as r:
        for conn, ts, raw in r.messages():
            if conn.topic == _META_TOPIC:
                meta = json.loads(_TS.deserialize_cdr(raw, conn.msgtype).data)
            elif conn.topic.startswith(_STREAM_PREFIX):
                m = _TS.deserialize_cdr(raw, conn.msgtype)
                raws_by_topic.setdefault(conn.topic, []).append(
                    np.asarray(m.data, dtype=np.uint8).tobytes()
                )
                ts_by_topic.setdefault(conn.topic, []).append(ts)

    if meta is None:
        raise ValueError(f"no {_META_TOPIC} message found in bag at {path}")

    episode = episode_from_dict(meta["episode"])
    streams: dict[str, list[Sample]] = {}
    for key, sm in meta["streams"].items():
        topic = _STREAM_PREFIX + key
        modality = Modality(sm["modality"])
        raws = raws_by_topic.get(topic, [])
        stamps = ts_by_topic.get(topic, [])
        samples: list[Sample] = []
        for raw_bytes, ts, man in zip(raws, stamps, sm["samples"]):
            arr = np.frombuffer(raw_bytes, dtype=np.dtype(man["dtype"])).reshape(man["shape"]).copy()
            samples.append(Sample(modality=modality, timestamp_ns=ts, data=arr, notes=man["notes"]))
        streams[key] = samples

    return RecordedEpisode(episode=episode, streams=streams)
