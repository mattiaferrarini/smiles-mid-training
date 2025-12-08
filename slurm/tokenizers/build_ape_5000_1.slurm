#!/bin/bash
#SBATCH --job-name=BuildTokenizer
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

echo "=================================================================="
echo "STARTING TOKENIZER BUILD JOB at $(date)"
echo "=================================================================="

OUTPUT_DIR="/users/$USER/json/tokenizers"
mkdir -p "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1

echo "Running tokenizer build script..."
srun "$PYTHON_EXEC" -u custom_tokenizers/build_tokenizer.py \
    --config configs/ape5000_1.yaml \
    --output-dir "$OUTPUT_DIR"
EXIT_CODE=$?

echo "=================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Tokenizer build completed successfully at $(date)"
else
    echo "FAILURE: Tokenizer build failed with exit code $EXIT_CODE at $(date)"
fi
echo "=================================================================="

echo "Job finished."
