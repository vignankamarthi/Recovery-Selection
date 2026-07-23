#!/bin/bash
# One-time conda env for the ACT baseline on AICR (Blackwell). Run on a GPU DEV node, not login.
#   salloc --partition=rtx-devel --gpus=1 --cpus-per-task=8 --mem=32G --time=01:00:00
#   bash scripts/cluster/00_setup_env.sh
# Blackwell needs the cu128 torch wheel (torch==2.11.0+cu128 -> cuda.is_available()==True).
set -euo pipefail

ENV=/home/kamarthi_v_neu/envs/harvest
module load miniforge3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -p "$ENV" python=3.11 -y
conda activate "$ENV"

# Blackwell torch FIRST (bundles cuda 12.8), then lerobot + our sim deps.
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install lerobot                      # ACT policy + LeRobotDataset + lerobot-train CLI
pip install mujoco numpy rosbags         # our sim + io deps (schema/harvest are pure-python, run from src/)

# Verify Blackwell CUDA is live (must print True; a False means the torch build is too old).
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import lerobot; print('lerobot', getattr(lerobot,'__version__','?'))"
echo "env ready at $ENV"
