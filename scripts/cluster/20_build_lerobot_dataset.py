"""Convert the materialized HARVEST streams into a LeRobotDataset for ACT training (cluster only).

Reads our per-episode export (`load_export`: data/<eid>.npz + episodes.jsonl) and writes a
LeRobotDataset with the ACT feature keys:
  observation.state          -> proprioception (7 arm joints)
  observation.images.overhead-> rgb_overhead (H,W,3 uint8)
  observation.images.wrist   -> rgb_wrist
  action                     -> next-step joint targets (proprioception[t+1]; last frame repeats)

Only TRAIN-split episodes go in (the by-can split from metadata.jsonl); val/test are held out for
the sim rollout eval (40_eval_rollout.py), not seen by lerobot-train.

UNTESTED until the cluster (no local torch/lerobot). The LeRobotDataset creation API has churned
across LeRobot versions, so integration-test with ONE episode on the GPU before the full build:
    python scripts/cluster/20_build_lerobot_dataset.py --streams <dir> --out <dir> --limit 1
Verify the frame schema, then drop --limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "src")

from harvest.io.lerobot_adapter import load_export
from schema.streams import Modality

FPS = 10                          # nominal; our sim timestamps are synthetic, ACT uses frame order


def _split_can_ids(streams_dir: Path, wanted: set[str]) -> set[str]:
    """The by-can ids in the wanted split(s), e.g. {'train'} or {'val','test'} (held-out)."""
    meta = streams_dir / "metadata.jsonl"
    if meta.exists():
        return {json.loads(l)["can_id"] for l in meta.read_text().splitlines()
                if l.strip() and json.loads(l)["split"] in wanted}
    # Fallback: recompute the identical deterministic split from the episodes present.
    from harvest.dataset.splits import split_by_can
    from harvest.io._serde import episode_from_dict
    eps = [episode_from_dict(json.loads(l)["episode"] if "episode" in json.loads(l) else json.loads(l))
           for l in (streams_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    sp = split_by_can(eps, seed=0)
    ids = set()
    if "train" in wanted: ids |= set(sp.train_can_ids)
    if "val" in wanted: ids |= set(sp.val_can_ids)
    if "test" in wanted: ids |= set(sp.test_can_ids)
    return ids


def _sorted(samples):
    return [s.data for s in sorted(samples, key=lambda s: s.timestamp_ns)]


def build(streams_dir: Path, out_dir: Path, repo_id: str, limit: int | None,
          wanted_splits: set[str] = frozenset({"train"})) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset   # API path VERIFY on cluster

    cans = _split_can_ids(streams_dir, set(wanted_splits))
    recs = [r for r in load_export(streams_dir) if r.episode.can_id in cans]
    if limit:
        recs = recs[:limit]
    if not recs:
        raise SystemExit(f"no episodes for splits {wanted_splits} found under {streams_dir}")

    # Infer shapes from the first episode.
    r0 = recs[0]
    state_dim = int(np.asarray(_sorted(r0.streams[Modality.PROPRIOCEPTION.value])[0]).shape[0])
    img = np.asarray(_sorted(r0.streams[Modality.RGB_OVERHEAD.value])[0])
    h, w, _ = img.shape
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "observation.images.overhead": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "observation.images.wrist": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "action": {"dtype": "float32", "shape": (state_dim,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features, root=str(out_dir))

    for rec in recs:
        state = [np.asarray(x, np.float32) for x in _sorted(rec.streams[Modality.PROPRIOCEPTION.value])]
        over = [np.asarray(x, np.uint8) for x in _sorted(rec.streams[Modality.RGB_OVERHEAD.value])]
        wrist = [np.asarray(x, np.uint8) for x in _sorted(rec.streams[Modality.RGB_WRIST.value])]
        n = min(len(state), len(over), len(wrist))
        for t in range(n):
            action = state[t + 1] if t + 1 < n else state[t]      # next-step joint target (absolute)
            ds.add_frame({
                "observation.state": state[t],
                "observation.images.overhead": over[t],
                "observation.images.wrist": wrist[t],
                "action": action,
                "task": "present the nutrition label to the overhead camera",   # LeRobot 0.4.4: task is a frame key
            })
        ds.save_episode()
    print(f"built LeRobotDataset '{repo_id}' at {out_dir}: {len(recs)} episodes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True, help="dir written by gen_dataset_parallel (data/ + *.jsonl)")
    ap.add_argument("--out", required=True, help="LeRobotDataset root")
    ap.add_argument("--repo-id", default="harvest/act_sim_v1")
    ap.add_argument("--limit", type=int, default=None, help="build only N episodes (integration test)")
    ap.add_argument("--split", choices=["train", "heldout"], default="train",
                    help="train (for training) or heldout = val+test (for the prediction metric)")
    a = ap.parse_args()
    splits = {"train"} if a.split == "train" else {"val", "test"}
    build(Path(a.streams), Path(a.out), a.repo_id, a.limit, splits)
