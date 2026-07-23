# ACT baseline on AICR (the 1.11 real training run)

The human-gated cluster run for the ACT baseline. Everything here is prepared locally. The sim
dataset materializes on the cluster (deterministic, so the Mac never holds the heavy streams) and
ACT trains + evaluates on the GPU nodes. Cluster reference is `/Cluster-Compute` (account
`p2026_0016_neu`, no `--qos`, `--gpus=N`, Blackwell torch `2.11.0+cu128`, code by git pull only).

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

# 2. Materialize the 600-episode streams to /scratch (all 7 modalities, deterministic). ~15-40 min.
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && sbatch scripts/cluster/10_materialize.slurm'

# 3. Build the LeRobotDataset from the train split. INTEGRATION TEST 1 episode first:
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && \
  python scripts/cluster/20_build_lerobot_dataset.py \
    --streams /scratch/kamarthi_v_neu/harvest/streams_v1 \
    --out /scratch/kamarthi_v_neu/harvest/lerobot_v1 --limit 1'
# verify the frame schema, then rebuild without --limit (fresh --out).

# 4. Train ACT. INTEGRATION TEST first: edit 30_train_act.slurm to rtx-batch + a small --steps,
#    confirm it steps, then submit the full B200 run.
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && sbatch scripts/cluster/30_train_act.slurm'

# 5. Roll out on held-out val/test in sim, write per-condition metrics.
ssh aicr 'cd /home/kamarthi_v_neu/Harvest-Recovery && sbatch scripts/cluster/50_eval.slurm'

# 6. Pull the small metrics file back for GATE 3.
mkdir -p experiments
scp kamarthi_v_neu@login.aicr.ai:/home/kamarthi_v_neu/Harvest-Recovery/experiments/act_sim_v1_metrics.json experiments/

# monitor: ssh aicr 'squeue --me'   logs: ssh aicr 'tail -f /home/kamarthi_v_neu/Harvest-Recovery/logs/<job>.out'
```

## The data contract (what ACT trains on)

Each frame: `observation.state` = 7 arm joints (proprioception), `observation.images.overhead` +
`observation.images.wrist` = the two RGB views, `action` = next-step absolute joint target
(proprioception[t+1]). Only TRAIN-split cans (by-can, leak-free) build the dataset. VAL/TEST cans are
held out and only touched by the sim rollout eval. Reported metric = per-condition label-exposure
success, to compare against the StubTrainer floor (0.722 overall on this set) and Padir's criterion
(>=75% nominal, >=50% damaged on the physical data).

## Notes / cluster iteration points

- `20_build_lerobot_dataset.py`: `LeRobotDataset.create` / `add_frame` / `save_episode` API path may
  differ by lerobot version. Test with `--limit 1` and read the resulting `meta/` schema.
- `30_train_act.slurm`, the `lerobot-train` flag names (`--dataset.root`, `--policy.type`) VERIFY. A local
  dataset is addressed by `repo_id` + `--dataset.root`.
- `40_eval_rollout.py`: `ACTPolicy.from_pretrained` path and `select_action` batch keys VERIFY. The
  rollout applies ACT's predicted joints with the same weld the demonstrations used.
- This is the sim ACT baseline (a smoke test / prototype). The REPORTED ACT baseline is the physical
  dataset (Phase 3). GATE 3 (1.13) reviews this run + the tactile ablation before Part 2.
