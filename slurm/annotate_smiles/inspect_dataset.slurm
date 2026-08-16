#!/bin/bash
#SBATCH --job-name=InspectDataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
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

echo "Checking and installing required libraries..."
pip install google-generativeai python-dotenv matplotlib

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
CONFIG_PATH="configs/utils/inspect_config.yaml"
srun "$PYTHON_EXEC" -u utils/annotate_smiles/inspect_dataset.py --config "$CONFIG_PATH"
EXIT_CODE=$?

echo "=================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Inspection completed successfully at $(date)"
    echo "Check output file for results."
else
    echo "FAILURE: Inspection failed with exit code $EXIT_CODE at $(date)"
fi
echo "=================================================================="

echo "Job finished."