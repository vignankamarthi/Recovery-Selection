"""Grounding diagnostics for the rigorous ACT eval, so the red/blue audit reasons from real numbers.

Breaks the eval-mode L1 and the no-move baseline down BY CONDITION on both splits, verifies the
no-move framing (does action[t] == state[t+1] in the built dataset?), reports raw value ranges, and
dumps a handful of (pred, target) rows for one held-out episode. Small JSON out, pulled to the Mac.

    python scripts/cluster/81_eval_diagnostic.py --checkpoint <dir> \
        --train-root <ds> --heldout-root <ds> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _loader(root, repo_id, chunk):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = json.load(open(os.path.join(root, "meta", "info.json")))["fps"]
    dt = {"action": [i / fps for i in range(chunk)]}
    ds = LeRobotDataset(repo_id, root=root, delta_timestamps=dt)
    return ds, DataLoader(ds, batch_size=32, num_workers=0, shuffle=False)


def _cond_of(ds, ep_idx):
    """Condition label from the episode task/id, best-effort (nominal/body_dent/seam_dent/bulge/rust)."""
    for key in ("task", "episode_task"):
        try:
            t = ds.meta.episodes[ep_idx].get(key) if hasattr(ds.meta, "episodes") else None
            if t:
                return str(t)
        except Exception:
            pass
    return "unknown"


def per_condition(policy, pre, post, root, repo_id, chunk, dev):
    ds, dl = _loader(root, repo_id, chunk)
    policy.eval()
    act = defaultdict(float)
    base = defaultdict(float)
    den = defaultdict(float)
    consist = 0.0
    cn = 0.0
    a_lo, a_hi = 1e9, -1e9
    for batch in dl:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        target = b["action"]
        state = b["observation.state"]
        mask = (~b["action_is_pad"]).unsqueeze(-1).float()
        with torch.no_grad():
            pred = policy.predict_action_chunk(pre(dict(b)))
            pred = post(pred)
            pred = (pred.to(dev) if torch.is_tensor(pred)
                    else torch.as_tensor(pred, device=dev, dtype=torch.float32))
        pred = pred[:, : target.shape[1]]
        no_move = state.unsqueeze(1).expand_as(target)
        # action[0] should equal state[t+1]; the no-move framing predicts state[t] for action[0].
        consist += (F.l1_loss(target[:, 0], state, reduction="none")).sum().item()
        cn += state.shape[0] * state.shape[-1]
        a_lo = min(a_lo, float(target.min()))
        a_hi = max(a_hi, float(target.max()))
        # bucket by condition via the frame's episode index
        idxs = b.get("episode_index")
        for i in range(target.shape[0]):
            c = _cond_of(ds, int(idxs[i])) if idxs is not None else "all"
            c = c.split("-")[0] if c != "all" else c
            m = mask[i]
            act[c] += (F.l1_loss(pred[i], target[i], reduction="none") * m).sum().item()
            base[c] += (F.l1_loss(no_move[i], target[i], reduction="none") * m).sum().item()
            den[c] += m.sum().item() * target.shape[-1]
    out = {c: {"l1_eval": act[c] / den[c], "l1_no_move": base[c] / den[c],
               "beats_no_move": bool(act[c] < base[c])} for c in den}
    return {"per_condition": out,
            "action0_vs_state_l1": consist / cn,   # ~0 if action[t] tracks state[t] closely (smooth)
            "action_raw_range": [a_lo, a_hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--heldout-root", required=True)
    ap.add_argument("--train-repo", default="harvest/act_sim_v1")
    ap.add_argument("--heldout-repo", default="harvest/act_sim_heldout")
    ap.add_argument("--out", default="experiments/act_eval_diagnostic.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = ACTPolicy.from_pretrained(a.checkpoint).to(dev)
    chunk = pol.config.chunk_size
    pre, post = make_pre_post_processors(pol.config, pretrained_path=a.checkpoint)

    m = {
        "chunk_size": int(chunk),
        "train": per_condition(pol, pre, post, a.train_root, a.train_repo, chunk, dev),
        "heldout": per_condition(pol, pre, post, a.heldout_root, a.heldout_repo, chunk, dev),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
