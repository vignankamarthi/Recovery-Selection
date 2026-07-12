"""Dataset export adapter (the dissemination seam, Phase 1.3).

Separate from rosbag2: rosbag2 is the lab/robot boundary, this is the public-dataset
boundary. Neither leaks format knowledge upstream.

MINIMAL export (honest scope): a HuggingFace-loadable flat layout, one `episodes.jsonl`
of episode metadata plus a per-episode `.npz` of the stream arrays. This round-trips
metadata and arrays exactly. Full LeRobot v2 spec conformance (chunked parquet, the
standard `meta/` schema) is a later refinement, not claimed here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from harvest.io._serde import episode_from_dict, episode_to_dict
from schema.episode import RecordedEpisode
from schema.streams import Modality, Sample


def export_dataset(episodes: Iterable[RecordedEpisode], out_dir: Path | str) -> None:
    """Write episodes to `out_dir` as episodes.jsonl + data/<episode_id>.npz."""
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for rec in episodes:
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
        lines.append(json.dumps(manifest))

    (out_dir / "episodes.jsonl").write_text("\n".join(lines) + "\n")


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
