#!/bin/bash
# Materialize the 600-episode streams (the videos) on the cluster to /scratch. The Mac keeps only
# metadata; this reproduces the identical episodes (deterministic) with all 7 modalities saved.
# MuJoCo headless renders via EGL on the GPU node. rtx-batch is plenty (sim + render, no training).
#SBATCH --job-name=harvest-materialize
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=p2026_0016_neu
set -euo pipefail
mkdir -p logs

REPO=/home/kamarthi_v_neu/Harvest-Recovery
OUT=/scratch/kamarthi_v_neu/harvest/streams_v1     # heavy .npz streams live on scratch (30-day purge)

module load miniforge3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/kamarthi_v_neu/envs/harvest

cd "$REPO"
export MUJOCO_GL=egl                                # headless GL on the GPU node (NOT cgl, that is macOS)
# 16 deterministic shards, resumable (skips episodes whose .npz already exists). Records all 7
# modalities (DEFAULT_MODALITIES). --cans 40 --poses 3 = 600 episodes.
python scripts/cluster/gen_dataset_parallel.py local --n 16 --out "$OUT" --cans 40 --poses 3
echo "materialized streams at $OUT"
du -sh "$OUT"
