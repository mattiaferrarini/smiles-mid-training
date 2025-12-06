#!/bin/bash
#SBATCH --job-name=CountDataRows
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH -A a131

export ENV_NAME=ml4science

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

export HF_HOME="/iopsstor/scratch/cscs/$USER/hf_cache"
mkdir -p "$HF_HOME"

export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

echo "Running row counting script..."
srun "$PYTHON_EXEC" -u count_rows.py

echo "Job finished."