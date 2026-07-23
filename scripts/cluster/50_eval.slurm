#!/bin/bash
# Roll out the trained ACT on the held-out val/test cans and write per-condition metrics.
# rtx-batch is enough (inference + sim, not training).
#SBATCH --job-name=harvest-act-eval
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=p2026_0016_neu
set -euo pipefail
mkdir -p logs

REPO=/home/kamarthi_v_neu/Harvest-Recovery
CKPT=/scratch/kamarthi_v_neu/harvest/act_out_v1/checkpoints/last/pretrained_model   # VERIFY path
STREAMS=/scratch/kamarthi_v_neu/harvest/streams_v1

module load miniforge3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/kamarthi_v_neu/envs/harvest

cd "$REPO"
export MUJOCO_GL=egl
python scripts/cluster/40_eval_rollout.py --checkpoint "$CKPT" --streams "$STREAMS" \
  --out "$REPO/experiments/act_sim_v1_metrics.json"
echo "wrote metrics -> $REPO/experiments/act_sim_v1_metrics.json"
