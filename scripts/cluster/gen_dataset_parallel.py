"""Parallel, resumable, sharded dataset generation (committed cluster runner).

Lives in scripts/cluster/ and is invoked from the repo root (it puts src/ and . on sys.path):
    python scripts/cluster/gen_dataset_parallel.py <mode> ...

Each episode is a pure deterministic function of (condition, can index, pose index), so generation
is embarrassingly parallel: shard by episode index, and any worker reproduces any episode exactly.
Streams are persisted per-episode (data/<eid>.npz via the io seam) with a per-episode manifest
(meta/<eid>.json), so there is no shared-write contention and a killed shard resumes by skipping
episodes whose .npz already exists.

The heavy .npz streams are meant to live on CLUSTER storage (ACT trains there); the Mac keeps only
the assembled metadata + split + card. Modes:

    python scripts/cluster/gen_dataset_parallel.py shard    --k K --n N --out DIR  # one shard (SLURM array task)
    python scripts/cluster/gen_dataset_parallel.py assemble  --out DIR             # metadata.jsonl + split + card
    python scripts/cluster/gen_dataset_parallel.py local     --n W  --out DIR      # W shards + assemble

Sizing via --cans (per condition) and --poses. 600 episodes = --cans 40 --poses 3.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, "src")
sys.path.insert(0, ".")

_X_RANGE = (0.46, 0.54)
_Y_RANGE = (-0.06, 0.06)
_SPAWN_Z = 0.11


def episode_plan(cans_per_condition: int, poses_per_can: int):
    """The full deterministic episode list, in a fixed global order (shard index = position here)."""
    from schema.episode import ConditionClass
    plan = []
    for c in ConditionClass:
        for i in range(cans_per_condition):
            can_id = f"{c.value}-{i:03d}"
            for p in range(poses_per_can):
                plan.append((f"{can_id}-p{p}", can_id, c, p))
    return plan


def episode_pose(can_id: str, p: int):
    """Deterministic table pose for a lying can, seeded per episode-id so any shard reproduces it."""
    r = random.Random(f"pose::{can_id}::{p}")
    return (r.uniform(*_X_RANGE), r.uniform(*_Y_RANGE), _SPAWN_Z)


def run_shard(k: int, n: int, out: Path, cans: int, poses: int, streams: bool = True) -> None:
    from harvest.io.flat_npz_adapter import write_episode_streams
    from harvest.io._serde import episode_to_dict
    from harvest.sim.episode import LYING_QUAT, DEFAULT_MODALITIES, record_sim_demo
    from schema.episode import Episode

    (out / "meta").mkdir(parents=True, exist_ok=True)
    if streams:
        (out / "data").mkdir(parents=True, exist_ok=True)
    plan = episode_plan(cans, poses)
    mine = [(idx, *e) for idx, e in enumerate(plan) if idx % n == k]
    t0 = time.time()
    made = skipped = 0
    for idx, eid, can_id, cond, p in mine:
        meta_done = (out / "meta" / f"{eid}.json").exists()
        data_done = (not streams) or (out / "data" / f"{eid}.npz").exists()
        if meta_done and data_done:                                # resume: skip finished episodes
            skipped += 1
            continue
        # Metadata-only mode still runs the full sim (the outcome IS the sim result), it just does
        # not persist the heavy stream arrays. Those get materialized on the cluster at ACT time.
        mods = DEFAULT_MODALITIES if streams else (DEFAULT_MODALITIES[0],)
        rec = record_sim_demo(Episode(eid, can_id, cond), can_pos=episode_pose(can_id, p),
                              modalities=mods, can_quat=LYING_QUAT)
        manifest = write_episode_streams(rec, out) if streams else {"episode": episode_to_dict(rec.episode)}
        (out / "meta" / f"{eid}.json").write_text(json.dumps(manifest))
        made += 1
        if made % 10 == 0:
            rate = made / (time.time() - t0)
            print(f"[shard {k}/{n}] {made} made, {skipped} skipped, {rate:.2f} eps/s", flush=True)
    print(f"[shard {k}/{n}] DONE {made} made, {skipped} skipped in {(time.time()-t0)/60:.1f} min", flush=True)


def assemble(out: Path) -> None:
    from harvest.io.flat_npz_adapter import write_index
    from harvest.io._serde import episode_from_dict
    from harvest.dataset.card import dataset_stats, render_split_table
    from harvest.dataset.competence import tag_competence
    from harvest.dataset.export import export_hf
    from harvest.dataset.splits import split_by_can

    metas = sorted((out / "meta").glob("*.json"))
    manifests = [json.loads(p.read_text()) for p in metas]
    write_index(manifests, out)                                    # episodes.jsonl (streams index)
    episodes = [episode_from_dict(m["episode"]) for m in manifests]
    tag_competence(episodes)                                       # 1.10 proxy competence tier -> metadata
    split = split_by_can(episodes, seed=0)
    export_hf(episodes, split, out)                                # metadata.jsonl + README card
    stats = dataset_stats(episodes, split)
    print(f"assembled {stats['total_episodes']} episodes / {stats['total_cans']} cans, "
          f"leak_free={stats['leak_free']}, outcomes={stats['outcomes']}")
    print("success rate per condition:", {k: round(v, 2) for k, v in stats['success_rate_per_condition'].items()})
    print("\n" + render_split_table(episodes, split))


def run_local(nshards: int, out: Path, cans: int, poses: int, streams: bool = True) -> None:
    base = ["--out", str(out), "--cans", str(cans), "--poses", str(poses)]
    if not streams:
        base.append("--no-streams")
    procs = [subprocess.Popen([sys.executable, __file__, "shard", "--k", str(k), "--n", str(nshards)] + base)
             for k in range(nshards)]
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"WARNING: shard exit codes {codes}", flush=True)
    assemble(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("shard", "assemble", "local"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--cans", type=int, default=40)
        s.add_argument("--poses", type=int, default=3)
        s.add_argument("--no-streams", action="store_true",
                       help="metadata only (skip the heavy .npz; materialize streams on the cluster)")
        if name == "shard":
            s.add_argument("--k", type=int, required=True)
            s.add_argument("--n", type=int, required=True)
        if name == "local":
            s.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    streams = not a.no_streams
    if a.mode == "shard":
        run_shard(a.k, a.n, out, a.cans, a.poses, streams)
    elif a.mode == "assemble":
        assemble(out)
    else:
        run_local(a.n, out, a.cans, a.poses, streams)


if __name__ == "__main__":
    main()
