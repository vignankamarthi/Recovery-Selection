"""Roll the trained ACT policy out in our MuJoCo sim on the VAL/TEST cans and report per-condition
label-exposure success (Padir's criterion). This is the real ACT eval the StubTrainer only stands in
for locally. Writes metrics.json (atomic). Cluster only (needs torch + the trained checkpoint).

The rollout mirrors demonstration generation: reset the lying can, grasp the head + weld, then at
each step let ACT predict the next joint target from the observation (overhead + wrist RGB + joint
state), apply it, and let the weld carry the can. After the horizon, read label_up_cos + overhead
coverage -> success. The condition-scaled slip is NOT injected at eval (that is the demonstrator's
failure model; here we measure whether the learned policy presents the label).

UNTESTED until the cluster. Integration-test ONE can first (--limit 1). The ACTPolicy load path and
the action/observation key names VERIFY against the installed lerobot version.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import sys
sys.path.insert(0, "src")

import numpy as np

from harvest.sim.episode import LYING_QUAT
from harvest.sim.scene import can_seed_from_id
from harvest.sim.world import SimWorld
from harvest.sim import reorient
from schema.episode import ConditionClass

UP_THRESH = reorient.UP_THRESH
HORIZON = 40


def _obs(w):
    return {
        "observation.state": np.asarray(w.proprioception(), np.float32),
        "observation.images.overhead": np.asarray(w.render("overhead"), np.uint8),
        "observation.images.wrist": np.asarray(w.render("wrist"), np.uint8),
    }


def rollout_one(policy, cond: ConditionClass, can_id: str, device) -> bool:
    import torch

    seed = can_seed_from_id(can_id)
    w = SimWorld(can_pos=(0.5, 0.0, 0.11), condition=cond, can_seed=seed, can_quat=LYING_QUAT)
    reorient._grasp_head(w)                       # same grasp + weld setup as the demonstrations
    weld = reorient.Weld(w)
    policy.reset()
    for _ in range(HORIZON):
        obs = _obs(w)
        batch = {k: torch.from_numpy(v)[None].to(device) for k, v in obs.items()}
        with torch.no_grad():
            action = policy.select_action(batch)[0].cpu().numpy()   # next joint target (VERIFY API)
        w.data.qpos[:7] = action[:7]
        import mujoco
        mujoco.mj_forward(w.model, w.data)
        weld.follow()
    _, n = w.can_label_pose()
    px = reorient._overhead_px(w)
    return bool(float(n[2]) > UP_THRESH and px is not None and px >= reorient.LABEL_VISIBLE_PX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained ACT output dir (pretrained_model)")
    ap.add_argument("--streams", required=True, help="dataset dir with metadata.jsonl (for the split)")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy       # path VERIFY on cluster
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(a.checkpoint).to(device).eval()

    rows = [json.loads(l) for l in (Path(a.streams) / "metadata.jsonl").read_text().splitlines() if l.strip()]
    heldout = [(r["can_id"], r["condition"]) for r in rows if r["split"] in ("val", "test")]
    seen, cans = set(), []
    for cid, cond in heldout:                      # one rollout per held-out can
        if cid not in seen:
            seen.add(cid); cans.append((cid, cond))
    if a.limit:
        cans = cans[:a.limit]

    hits = defaultdict(lambda: [0, 0])
    for cid, cond in cans:
        ok = rollout_one(policy, ConditionClass(cond), cid, device)
        hits[cond][0] += int(ok); hits[cond][1] += 1
        print(f"{cid:<16} {cond:<10} success={ok}", flush=True)

    per_cond = {c: h[0] / h[1] for c, h in hits.items()}
    overall = sum(h[0] for h in hits.values()) / max(1, sum(h[1] for h in hits.values()))
    metrics = {"per_condition_success": per_cond, "overall_success": overall, "n_cans": len(cans)}
    tmp = a.out + ".tmp"
    Path(tmp).write_text(json.dumps(metrics, indent=2))
    os.replace(tmp, a.out)                         # atomic
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
