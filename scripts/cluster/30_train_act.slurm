#!/bin/bash
# Train the ACT baseline on the built LeRobotDataset (train split). B200 for the real run; drop to
# rtx-batch for the 1-unit integration test (--steps small). Checkpoints to scratch (not Home).
#SBATCH --job-name=harvest-act-train
#SBATCH --partition=b200-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=p2026_0016_neu
set -euo pipefail
mkdir -p logs

DATASET_ROOT=/work/neu/p2026_0016_neu/harvest/lerobot_v1
OUTPUT=/work/neu/p2026_0016_neu/harvest/act_out_v1
REPO_ID=harvest/act_sim_v1

module load miniforge3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/kamarthi_v_neu/envs/harvest

CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_OK" != "True" ]; then echo "WARNING: CUDA not available (torch too old for Blackwell?)"; fi

# ACT on our sim dataset. A LOCAL dataset is addressed by repo_id + --dataset.root.
# INTEGRATION TEST FIRST: add --steps=200 and use rtx-batch to catch schema/loader bugs cheaply,
# then remove --steps for the full run. Exact flag names VERIFY against the installed lerobot version.
lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUTPUT" \
  --job_name=harvest_act_sim_v1 \
  --wandb.enable=false \
  --batch_size=64 \
  --steps=40000 \
  --save_freq=10000

echo "trained ACT -> $OUTPUT"
