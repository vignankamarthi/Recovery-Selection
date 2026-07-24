"""Rigorous, weld-independent ACT eval scored THROUGH LeRobot's own dataloader (cluster only).

This is the metric the red/blue audit asked for after the arm-slip fix made the sim demos teachable.
It answers one question honestly: did ACT learn the observation -> action mapping, and does it beat a
trivial predictor and generalize to unseen cans? Everything is scored through LeRobot's dataloader, so
the observations match the training pipeline exactly (the earlier hand-rolled obs pipeline did not, and
scored a converged policy WORSE than no-move, which is impossible). Three numbers, TRAIN vs HELD-OUT:

  1. l1_eval_raw   -- EVAL-mode predicted-action L1 in RAW joint space (radians). Policy in eval mode, so
                      the VAE latent is the prior (z=0, deterministic inference), exactly what runs at
                      test time. The prediction is un-normalized back to radians and compared to the demo
                      action chunk. This is the "did ACT learn the mapping" number.
  2. l1_no_move    -- NAIVE baseline in the same raw space: predict "no movement" (action[i] = the current
                      joint state, broadcast across the chunk). Because joints barely move frame-to-frame,
                      this is a low bar that a real policy MUST beat. ACT is only meaningful if
                      l1_eval_raw < l1_no_move on held-out cans.
  3. loss_components -- TRAIN-mode `policy.forward` loss dict (l1_loss, kld_loss) via the same dataloader,
                      the generalization signal (small train->held-out l1 gap = ACT generalizes, not
                      memorizes). Train mode because ACT's VAE encoder + KL only run in train mode.

Deterministic (seeded), shuffle off, JSON written straight from the computed numbers (never hand-edited).

    python scripts/cluster/80_rigorous_eval.py --checkpoint <dir> \
        --train-root <lerobot train ds> --heldout-root <lerobot heldout ds> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SEED = 0


def _loader(root: str, repo_id: str, chunk: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = json.load(open(os.path.join(root, "meta", "info.json")))["fps"]
    dt = {"action": [i / fps for i in range(chunk)]}          # chunk the ACTION (ACT is single-obs-step)
    ds = LeRobotDataset(repo_id, root=root, delta_timestamps=dt)
    return DataLoader(ds, batch_size=32, num_workers=0, shuffle=False)


def _to(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def eval_l1_raw(policy, dl, dev) -> tuple[float, float, int]:
    """EVAL-mode predicted-action L1 vs the naive no-move baseline, both in raw joint space (radians)."""
    policy.eval()
    act_num = base_num = denom = 0.0
    for batch in dl:
        b = _to(batch, dev)
        target = b["action"]                                  # (B, chunk, A) raw demo action
        state = b["observation.state"]                        # (B, A) current joint state (raw)
        mask = (~b["action_is_pad"]).unsqueeze(-1).float()    # (B, chunk, 1) real (non-pad) frames
        with torch.no_grad():
            nb = policy.normalize_inputs(dict(b))             # normalize OBS the way training did
            nb.pop("action", None)                            # eval prediction must not see the target
            nb.pop("action_is_pad", None)
            actions_hat, _ = policy.model(nb)                 # eval mode -> z=prior (deterministic)
            pred_raw = policy.unnormalize_outputs({"action": actions_hat})["action"]   # back to radians
        no_move = state.unsqueeze(1).expand_as(target)        # predict the current state for every step
        act_num += (F.l1_loss(pred_raw, target, reduction="none") * mask).sum().item()
        base_num += (F.l1_loss(no_move, target, reduction="none") * mask).sum().item()
        denom += mask.sum().item() * target.shape[-1]
    return act_num / denom, base_num / denom, int(denom)


def loss_components(policy, dl, dev) -> tuple[dict, int]:
    """TRAIN-mode `policy.forward` loss components (l1_loss, kld_loss) for the generalization gap."""
    policy.train()                                            # VAE encoder + KL only run in train mode
    agg: dict[str, float] = defaultdict(float)
    n = 0
    for batch in dl:
        b = _to(batch, dev)
        with torch.no_grad():
            out = policy.forward(b)
        comp = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 and isinstance(out[1], dict) else {}
        bs = len(batch["action"])
        n += bs
        for k, v in comp.items():
            try:
                agg[k] += float(v) * bs
            except (TypeError, ValueError):
                pass
    return {k: v / n for k, v in agg.items()}, n


def split(policy, root, repo_id, chunk, dev) -> dict:
    l1_act, l1_base, nf = eval_l1_raw(policy, _loader(root, repo_id, chunk), dev)
    comp, _ = loss_components(policy, _loader(root, repo_id, chunk), dev)
    return {
        "l1_eval_raw": l1_act,          # ACT eval-mode prediction, raw joint L1 (radians)
        "l1_no_move": l1_base,          # naive baseline, same space
        "beats_no_move": bool(l1_act < l1_base),
        "loss_components": comp,        # train-mode l1_loss / kld_loss
        "n_frames": nf,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--heldout-root", required=True)
    ap.add_argument("--train-repo", default="harvest/act_sim_v1")
    ap.add_argument("--heldout-repo", default="harvest/act_sim_heldout")
    ap.add_argument("--out", default="experiments/act_rigorous_eval.json")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    from lerobot.policies.act.modeling_act import ACTPolicy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = ACTPolicy.from_pretrained(a.checkpoint).to(dev)
    chunk = pol.config.chunk_size

    tr = split(pol, a.train_root, a.train_repo, chunk, dev)
    ho = split(pol, a.heldout_root, a.heldout_repo, chunk, dev)
    m = {
        "seed": SEED,
        "train": tr,
        "heldout": ho,
        "l1_eval_gap": ho["l1_eval_raw"] - tr["l1_eval_raw"],
        "note": "l1_eval_raw = eval-mode (z=prior) predicted-action L1 in radians, scored through "
                "LeRobot's dataloader. ACT is meaningful iff l1_eval_raw < l1_no_move on held-out "
                "(beats the trivial no-move predictor); a small l1_eval_gap = generalizes to unseen cans.",
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    open(tmp, "w").write(json.dumps(m, indent=2))
    os.replace(tmp, a.out)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
