# ACT baseline on AICR (the 1.11 real training run)

The human-gated cluster run for the ACT baseline. Everything here is prepared locally. The sim
dataset materializes on the cluster (deterministic, so the Mac never holds the heavy streams) and
ACT trains + evaluates on the GPU nodes. Cluster reference is `/Cluster-Compute` (account
`p2026_0016_neu`, no `--qos`, `--gpus=N`, Blackwell torch `2.11.0+cu128`, code by git pull only).

**RE-RUN (2026-07-24), after the arm-slip fix.** The first run validated the whole pipeline but the
red/blue audit found the sim demos were UNTEACHABLE (the slip rotated the can while the arm stayed
frozen, so the failure was not in the action ACT imitates). `sim/reorient.py::_slip_roll` now rolls the
end-effector about the can's long axis via IK, so all seven arm joints carry the failure. This re-run
REGENERATES the dataset with the new slip, retrains, and rescores. The old closed-loop rollout eval
was WELD-INVALID and has been REMOVED; the dataloader-scored `80_rigorous_eval.py` is the valid eval.

**Status, UNTESTED until the cluster (no local torch/lerobot).** Follow the cluster convention,
integration-test ONE unit on the real GPU at each step before the full run. The LeRobot dataset and
policy API path names are marked `VERIFY` where they may differ by installed lerobot version.

## Order of operations

Vignan pushes the committed code first (`/Commit-Initiation` plan -> `git push`). Then:

```bash
# 0. Code onto the cluster (pull-only; first time is a clone).
ssh aicr 'git clone <REPO_URL> /home/kamarthi_v_neu/Harvest-Recovery' \
  || ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && git pull --ff-only'

# 1. One-time env on a GPU DEV node (torch cu128 + lerobot + mujoco). ~15 min.
ssh aicr 'salloc --partition=rtx-devel --gpus=1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
  bash /home/kamarthi_v_neu/Harvest-Recovery/scripts/cluster/00_setup_env.sh'

# 2. REGENERATE the 600-episode streams to /work (non-purging) with the NEW arm-slip (all 7 modalities,
#    deterministic). Use a FRESH output dir so no old can-roll streams survive. ~15-40 min.
#    (10_materialize.slurm runs scripts/cluster/gen_dataset_parallel.py.)
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && sbatch scripts/cluster/10_materialize.slurm'

# 3. Build the TRAIN LeRobotDataset. INTEGRATION TEST 1 episode first (rebuild without --limit after).
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && \
  python scripts/cluster/20_build_lerobot_dataset.py --split train \
    --streams /work/neu/p2026_0016_neu/harvest/streams_v1 \
    --out /work/neu/p2026_0016_neu/harvest/lerobot_v1 --repo-id harvest/act_sim_v1 --limit 1'
# verify the frame schema, then rebuild without --limit (fresh --out).

# 3b. Build the HELD-OUT LeRobotDataset (val+test cans) for scoring. Fresh --out.
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && \
  python scripts/cluster/20_build_lerobot_dataset.py --split heldout \
    --streams /work/neu/p2026_0016_neu/harvest/streams_v1 \
    --out /work/neu/p2026_0016_neu/harvest/lerobot_heldout --repo-id harvest/act_sim_heldout'

# 4. Train ACT. INTEGRATION TEST first: edit 30_train_act.slurm to rtx-batch + a small --steps,
#    confirm it steps, then submit the full B200 run (~35 min, 40k steps).
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && sbatch scripts/cluster/30_train_act.slurm'

# 5. RIGOROUS eval through LeRobot's dataloader (eval-mode L1 + no-move baseline + loss components),
#    TRAIN vs HELD-OUT. INTEGRATION TEST: the script iterates both loaders; run it directly on a dev
#    node first (it is short), read the JSON, then trust it. Weld-independent, replaces the rollout.
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && \
  python scripts/cluster/80_rigorous_eval.py \
    --checkpoint /work/neu/p2026_0016_neu/harvest/act_out_v1/checkpoints/last/pretrained_model \
    --train-root /work/neu/p2026_0016_neu/harvest/lerobot_v1 \
    --heldout-root /work/neu/p2026_0016_neu/harvest/lerobot_heldout \
    --out experiments/act_rigorous_eval.json'

# 6. Pull the small JSON back for the red/blue audit + GATE 3. (script-written, never hand-edited)
mkdir -p experiments
scp kamarthi_v_neu@login.aicr.ai:/home/kamarthi_v_neu/Harvest-Recovery/experiments/act_rigorous_eval.json experiments/

# monitor: ssh aicr 'squeue --me'   logs: ssh aicr 'tail -f /home/kamarthi_v_neu/Harvest-Recovery/logs/<job>.out'
```

## The data contract (what ACT trains on)

Each frame: `observation.state` = 7 arm joints (proprioception), `observation.images.overhead` +
`observation.images.wrist` = the two RGB views, `action` = next-step absolute joint target
(proprioception[t+1]). Only TRAIN-split cans (by-can, leak-free) build the train dataset. VAL/TEST cans
build the separate HELD-OUT dataset scored by `80_rigorous_eval.py`. Reported metric = eval-mode
predicted-action L1 (radians) vs a no-move baseline, TRAIN vs HELD-OUT. ACT is meaningful only if it
beats no-move on held-out cans, and a small train->held-out gap means it generalizes.

## Notes / cluster iteration points

- `20_build_lerobot_dataset.py` (`LeRobotDataset.create` / `add_frame` / `save_episode`) API path may
  differ by lerobot version. Test with `--limit 1` and read the resulting `meta/` schema. Build BOTH
  splits, `--split train` (repo `harvest/act_sim_v1`) and `--split heldout` (repo `harvest/act_sim_heldout`).
- `30_train_act.slurm`, the `lerobot-train` flag names (`--dataset.root`, `--policy.type`) VERIFY. A local
  dataset is addressed by `repo_id` + `--dataset.root`.
- `80_rigorous_eval.py` is the VALID eval. It scores through LeRobot's own dataloader (matching the
  training obs pipeline), reports eval-mode (z=prior) predicted-action L1 + a no-move baseline + the
  train-mode loss components, and writes JSON straight from the numbers. `policy.model(nb)` returning
  `(actions_hat, ...)` and `unnormalize_outputs` are the version-sensitive calls, integration-run the
  script on a dev node first and read the JSON before trusting it.
- The closed-loop rollout eval (formerly `40_eval_rollout.py` / `50_eval.slurm`) and the superseded
  `60_predict_eval.py` have been REMOVED: rollout is weld-invalid (a learned policy cannot reproduce the
  scripted search+weld presentation), and `80_rigorous_eval.py` replaces the prediction metric.
- This is the sim ACT baseline (a smoke test / prototype). The REPORTED ACT baseline is the physical
  dataset (Phase 3). GATE 3 (1.13) reviews this run + the tactile ablation before Part 2.
