#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH -A a131

export ENV_NAME=ml4science

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh # or the path where you downloaded Anaconda (usually in your /home folder)

# Create or update conda environment
if conda env list | grep -q "^$ENV_NAME "; then \
 echo "Conda env exists: updating"; \
 conda run -n "$ENV_NAME" pip install -r requirements.txt; \
else \
 echo "Creating conda env"; \
 conda create -n "$ENV_NAME" --override-channels -c conda-forge python=3.12 -y && \
 conda run -n "$ENV_NAME" pip install -r requirements.txt; \
fi

# Activate conda environment
conda activate "$ENV_NAME"

# Distributed settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Memory Management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Silence standard Python warnings for cleaner logs
export PYTHONWARNINGS="ignore"

# HF cache folders
export HF_HOME="/iopsstor/scratch/cscs/$USER/cache/huggingface"
export HF_DATASETS_CACHE="/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

# Create needed folders
mkdir -p /iopsstor/scratch/cscs/$USER/logs
mkdir -p /iopsstor/scratch/cscs/$USER/jobreports
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface"
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

# Jobreport
export JOBREPORT=/users/$USER/jobreport
export PROFILING_FOLDER=/iopsstor/scratch/cscs/$USER/jobreports/jobreport_${SLURM_JOB_ID}

# Training settings
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR="/iopsstor/scratch/cscs/$USER/benchmarks/"
export CONFIG_PATH=configs/default_ddp.yaml
export MODEL_PATH="/users/monti/smiles-mid-training/results_gemma_strict/checkpoint-783/"

# Get master host address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Ask the OS for a free port for the master
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Get full path to python exec
export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

# Run benchmarks
echo "Running chembench..."
srun $JOBREPORT monitor -o "$PROFILING_FOLDER" -- "$PYTHON_EXEC" main.py run-chembench \
    --model-path "$MODEL_PATH" \
    --output-path "$OUTPUT_DIR"

# Print report
$JOBREPORT print "$PROFILING_FOLDER"
