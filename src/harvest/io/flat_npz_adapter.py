"""Flat npz + jsonl dataset export adapter (the dissemination seam, Phase 1.3).

Separate from rosbag2: rosbag2 is the lab/robot boundary, this is the public-dataset
boundary. Neither leaks format knowledge upstream.

This writes a FLAT layout (hence the module name): one `episodes.jsonl` of episode metadata
plus a per-episode `.npz` of the stream arrays, HuggingFace-loadable and round-tripping metadata
and arrays exactly. It is NOT the LeRobot v2 on-disk format (chunked parquet, the standard `meta/`
schema); that conversion, when needed for lerobot-train, is done separately on the cluster by
`scripts/cluster/20_build_lerobot_dataset.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from harvest.io._serde import episode_from_dict, episode_to_dict
from schema.episode import RecordedEpisode
from schema.streams import Modality, Sample


def write_episode_streams(rec: RecordedEpisode, out_dir: Path | str) -> dict:
    """Persist ONE episode's streams to `out_dir/data/<episode_id>.npz` and return its manifest
    (episode metadata + per-stream sample metadata). This is the per-episode seam a parallel or
    resumable generation run uses: each worker writes its own .npz with no shared-write contention,
    and skips episodes whose .npz already exists. `write_index` then assembles episodes.jsonl."""
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    manifest: dict = {"episode": episode_to_dict(rec.episode), "streams": {}}
    arrays: dict[str, np.ndarray] = {}
    for key, samples in rec.streams.items():
        s_sorted = sorted(samples, key=lambda s: s.timestamp_ns)
        manifest["streams"][key] = {
            "modality": s_sorted[0].modality.value if s_sorted else None,
            "samples": [{"timestamp_ns": s.timestamp_ns, "notes": s.notes} for s in s_sorted],
        }
        for i, s in enumerate(s_sorted):
            arrays[f"{key}__{i}"] = np.asarray(s.data)
    np.savez(out_dir / "data" / f"{rec.episode.episode_id}.npz", **arrays)
    return manifest


def write_index(manifests: Iterable[dict], out_dir: Path | str) -> None:
    """Write episodes.jsonl from per-episode manifests (as returned by `write_episode_streams`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(m) for m in manifests]
    (out_dir / "episodes.jsonl").write_text("\n".join(lines) + "\n")


def export_dataset(episodes: Iterable[RecordedEpisode], out_dir: Path | str) -> None:
    """Write episodes to `out_dir` as episodes.jsonl + data/<episode_id>.npz. Streams each episode
    to disk as it goes (memory stays bounded to one episode's arrays), so a lazy generator of
    RecordedEpisodes can drive a large export without materializing them all."""
    manifests = [write_episode_streams(rec, out_dir) for rec in episodes]
    write_index(manifests, out_dir)


def load_export(out_dir: Path | str) -> list[RecordedEpisode]:
    """Reload a flat export written by `export_dataset`."""
    out_dir = Path(out_dir)
    result: list[RecordedEpisode] = []
    for line in (out_dir / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        manifest = json.loads(line)
        episode = episode_from_dict(manifest["episode"])
        npz = np.load(out_dir / "data" / f"{episode.episode_id}.npz")
        streams: dict[str, list[Sample]] = {}
        for key, sm in manifest["streams"].items():
            modality = Modality(sm["modality"])
            streams[key] = [
                Sample(modality, man["timestamp_ns"], npz[f"{key}__{i}"], man["notes"])
                for i, man in enumerate(sm["samples"])
            ]
        result.append(RecordedEpisode(episode=episode, streams=streams))
    return result
