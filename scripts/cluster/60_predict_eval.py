"""Held-out action-prediction metric for the trained ACT (weld-independent, cluster only).

The closed-loop rollout eval is invalid for weld-generated demos (see 1.11-run). This measures what
IS valid: given each recorded observation, how well does ACT predict the demonstration's next joint
action? Reports mean per-joint MSE on TRAIN vs HELD-OUT (val+test) cans. A low held-out MSE and a
small train->held-out gap means ACT learned the obs->action mapping and generalizes; a large gap is
overfitting. This is the standard imitation-learning generalization signal, no rollout needed.

    python scripts/cluster/60_predict_eval.py --checkpoint <dir> --streams <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "src")

import numpy as np
import torch

from harvest.io.lerobot_adapter import load_export
from schema.streams import Modality


def _sorted(samples):
    return [s.data for s in sorted(samples, key=lambda s: s.timestamp_ns)]


def _frames(rec):
    st = [np.asarray(x, np.float32) for x in _sorted(rec.streams[Modality.PROPRIOCEPTION.value])]
    ov = [np.asarray(x, np.uint8) for x in _sorted(rec.streams[Modality.RGB_OVERHEAD.value])]
    wr = [np.asarray(x, np.uint8) for x in _sorted(rec.streams[Modality.RGB_WRIST.value])]
    return st, ov, wr, min(len(st), len(ov), len(wr))


def _predict(policy, state, over, wrist, dev):
    obs = {"observation.state": state, "observation.images.overhead": over, "observation.images.wrist": wrist}
    b = {}
    for k, v in obs.items():
        t = torch.from_numpy(v)
        t = (t.permute(2, 0, 1).float() / 255.0) if "image" in k else t.float()
        b[k] = t[None].to(dev)
    policy.reset()                                   # fresh prediction per frame (single-step)
    with torch.no_grad():
        return policy.select_action(b)[0].cpu().numpy()


def eval_split(policy, recs, dev) -> tuple[float, int]:
    errs = []
    for rec in recs:
        st, ov, wr, n = _frames(rec)
        for t in range(n - 1):
            a = _predict(policy, st[t], ov[t], wr[t], dev)
            errs.append(float(np.mean((a[:7] - st[t + 1][:7]) ** 2)))   # predicted vs demo next joints
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


def _cans(streams_dir: str, wanted: set[str]) -> set[str]:
    return {json.loads(l)["can_id"] for l in open(os.path.join(streams_dir, "metadata.jsonl"))
            if l.strip() and json.loads(l)["split"] in wanted}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--out", default="experiments/act_predict_metrics.json")
    ap.add_argument("--limit", type=int, default=None, help="cap episodes per split (debug)")
    a = ap.parse_args()

    from lerobot.policies.act.modeling_act import ACTPolicy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = ACTPolicy.from_pretrained(a.checkpoint).to(dev).eval()

    recs = load_export(a.streams)
    tr, ho = _cans(a.streams, {"train"}), _cans(a.streams, {"val", "test"})
    tr_recs = [r for r in recs if r.episode.can_id in tr]
    ho_recs = [r for r in recs if r.episode.can_id in ho]
    if a.limit:
        tr_recs, ho_recs = tr_recs[:a.limit], ho_recs[:a.limit]

    tr_mse, ntr = eval_split(pol, tr_recs, dev)
    ho_mse, nho = eval_split(pol, ho_recs, dev)
    m = {"train_action_mse": tr_mse, "heldout_action_mse": ho_mse, "gap": ho_mse - tr_mse,
         "n_train_frames": ntr, "n_heldout_frames": nho,
         "note": "single-step joint MSE; low held-out + small gap = ACT learned + generalizes (weld-independent)"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    open(tmp, "w").write(json.dumps(m, indent=2))
    os.replace(tmp, a.out)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
