"""Rigorous, weld-independent ACT eval scored THROUGH LeRobot's own dataloader (cluster only).

This is the metric the red/blue audit asked for after the arm-slip fix made the sim demos teachable.
It answers one question honestly: did ACT learn the observation -> action mapping, and does it beat a
trivial predictor and generalize to unseen cans? Everything is scored through LeRobot's dataloader, so
the observations match the training pipeline exactly (the earlier hand-rolled obs pipeline did not, and
scored a converged policy WORSE than no-move, which is impossible). Three numbers, TRAIN vs HELD-OUT:

  1. l1_eval       -- EVAL-mode predicted-action L1 via `predict_action_chunk`. Policy in eval mode, so
                      the VAE latent is the prior (z=0, deterministic inference), exactly what runs at
                      test time. The prediction comes back in the dataset's own action space (ACT's
                      `forward` compares model output to `action` directly), compared to the demo action
                      chunk in that same space. This is the "did ACT learn the mapping" number.
  2. l1_no_move    -- NAIVE baseline in the same space: predict "no movement" (action[i] = the current
                      joint state, broadcast across the chunk; action[t] = state[t+1] here). Because
                      joints barely move frame-to-frame, this is a low bar that a real policy MUST beat.
                      ACT is only meaningful if l1_eval < l1_no_move on held-out cans.
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


def _as_tensor(x, dev):
    """The postprocessor may return a tensor or a numpy array; normalize to a tensor on `dev`."""
    return x.to(dev) if torch.is_tensor(x) else torch.as_tensor(x, device=dev, dtype=torch.float32)


def eval_l1(policy, pre, post, dl, dev) -> tuple[float, float, int]:
    """EVAL-mode predicted-action L1 vs the naive no-move baseline, in RAW joint space (radians).

    The critical detail: LeRobot normalizes obs/actions in a PROCESSOR pipeline OUTSIDE the model, so
    the model must be fed through `pre` (the saved normalizer) exactly as training and rollout do, or it
    sees out-of-distribution inputs and collapses. Here the batch is normalized by `pre`, ACT predicts
    the chunk in eval mode (VAE latent = prior, deterministic), and `post` (the saved unnormalizer) maps
    the prediction back to raw radians to compare against the raw demo action. Since `action[t] =
    state[t+1]` in this dataset, the no-move baseline (predict the current joint state for the whole
    chunk) is the trivial predictor ACT must beat."""
    policy.eval()
    act_num = base_num = denom = 0.0
    frames = 0
    for batch in dl:
        b = _to(batch, dev)
        target = b["action"]                                  # (B, chunk, A) raw demo action (radians)
        state = b["observation.state"]                        # (B, A) raw current joint state
        mask = (~b["action_is_pad"]).unsqueeze(-1).float()    # (B, chunk, 1) real (non-pad) frames
        with torch.no_grad():
            pred = policy.predict_action_chunk(pre(dict(b)))  # normalize obs -> eval predict (z=prior)
            pred = _as_tensor(post(pred), dev)                # unnormalize back to raw radians
        pred = pred[:, : target.shape[1]]                     # match the target chunk length
        no_move = state.unsqueeze(1).expand_as(target)        # predict the current state for every step
        act_num += (F.l1_loss(pred, target, reduction="none") * mask).sum().item()
        base_num += (F.l1_loss(no_move, target, reduction="none") * mask).sum().item()
        denom += mask.sum().item() * target.shape[-1]
        frames += int(mask.sum().item())
    return act_num / denom, base_num / denom, frames


def loss_components(policy, pre, dl, dev) -> tuple[dict, int]:
    """TRAIN-mode `policy.forward` loss components (l1_loss, kld_loss) for the generalization gap, in the
    normalized action space training used. The batch is normalized by `pre` first, exactly as the train
    loop does (`batch = preprocessor(batch)` before the forward)."""
    policy.train()                                            # VAE encoder + KL only run in train mode
    agg: dict[str, float] = defaultdict(float)
    n = 0
    for batch in dl:
        b = _to(batch, dev)
        with torch.no_grad():
            out = policy.forward(pre(dict(b)))
        comp = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 and isinstance(out[1], dict) else {}
        bs = len(batch["action"])
        n += bs
        for k, v in comp.items():
            try:
                agg[k] += float(v) * bs
            except (TypeError, ValueError):
                pass
    return {k: v / n for k, v in agg.items()}, n


def split(policy, pre, post, root, repo_id, chunk, dev) -> dict:
    l1_act, l1_base, nf = eval_l1(policy, pre, post, _loader(root, repo_id, chunk), dev)
    comp, _ = loss_components(policy, pre, _loader(root, repo_id, chunk), dev)
    return {
        "l1_eval": l1_act,              # ACT eval-mode (z=prior) predicted-action L1, raw radians
        "l1_no_move": l1_base,          # naive no-move baseline, same raw space
        "beats_no_move": bool(l1_act < l1_base),
        "loss_components": comp,        # train-mode l1_loss / kld_loss (normalized space)
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
    from lerobot.policies.factory import make_pre_post_processors
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = ACTPolicy.from_pretrained(a.checkpoint).to(dev)
    chunk = pol.config.chunk_size
    # The saved normalizer/unnormalizer processors (obs + action normalization lives OUTSIDE the model).
    pre, post = make_pre_post_processors(pol.config, pretrained_path=a.checkpoint)

    tr = split(pol, pre, post, a.train_root, a.train_repo, chunk, dev)
    ho = split(pol, pre, post, a.heldout_root, a.heldout_repo, chunk, dev)
    m = {
        "seed": SEED,
        "train": tr,
        "heldout": ho,
        "l1_eval_gap": ho["l1_eval"] - tr["l1_eval"],
        "note": "l1_eval = eval-mode (z=prior) predicted-action L1 via predict_action_chunk, scored "
                "through LeRobot's dataloader (obs match training). ACT is meaningful iff l1_eval < "
                "l1_no_move on held-out (beats the trivial no-move predictor); a small l1_eval_gap = "
                "generalizes to unseen cans.",
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    open(tmp, "w").write(json.dumps(m, indent=2))
    os.replace(tmp, a.out)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
