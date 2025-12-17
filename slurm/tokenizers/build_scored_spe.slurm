#!/bin/bash
#SBATCH --job-name=BuildTokenizer_ScoredSPE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=none
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH -A a131

export ENV_NAME=ml4science

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# HF cache folders
export HF_HOME="/iopsstor/scratch/cscs/$USER/cache/huggingface"
export HF_DATASETS_CACHE="/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

# Create needed folders
mkdir -p /iopsstor/scratch/cscs/$USER/logs
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface"
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

export PYTHONUNBUFFERED=1

echo "Running tokenizer build script..."
srun "$PYTHON_EXEC" -u main.py build-tokenizer \
    --config configs/tokenizers/scored_spe.yaml
EXIT_CODE=$?
