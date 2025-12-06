#!/bin/bash
#SBATCH --job-name=InspectDataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --nodes=1
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

echo "=================================================================="
echo "STARTING DATASET INSPECTION JOB at $(date)"
echo "=================================================================="

echo "Running inspection script..."
srun "$PYTHON_EXEC" inspect_dataset.py
EXIT_CODE=$?

echo "=================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Inspection completed successfully at $(date)"
    echo "Check dataset_head.txt for results."
else
    echo "FAILURE: Inspection failed with exit code $EXIT_CODE at $(date)"
fi
echo "=================================================================="

echo "Job finished."
