#!/bin/bash
#SBATCH --job-name=training_hf_ape_10000
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH -A a131

export ENV_NAME=ml4science

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh # or the path where you downloaded Anaconda (usually in your /home folder)


# Activate conda environment
conda activate "$ENV_NAME"

# Ensure the GPU version of PyTorch is installed
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Update requirements with specific CUDA versions for cluster
# echo "Updating requirements.txt with current environment packages..."
# pip freeze > requirements.txt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Distributed settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Increase synchronization timeout
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=5400  # 90 minutes
export NCCL_TIMEOUT=5400 # 90 minutes

# Memory Management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Silence standard Python warnings for cleaner logs
export PYTHONWARNINGS="ignore"

# HF cache folders
export HF_HOME="/iopsstor/scratch/cscs/$USER/cache/huggingface"
export HF_DATASETS_CACHE="/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

# WandB settings (uncomment if using WandB)
# export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d '=' -f 2)
# export WANDB_PROJECT="slurm"

# Create needed folders
mkdir -p /iopsstor/scratch/cscs/$USER/logs
mkdir -p /iopsstor/scratch/cscs/$USER/jobreports
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface"
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

# Jobreport
export JOBREPORT=/users/$USER/jobreport
export PROFILING_FOLDER=/iopsstor/scratch/cscs/$USER/jobreports/jobreport_${SLURM_JOB_ID}

# Training settings
export JOB_NAME="base_finetuning"
export OUTPUT_DIR="/iopsstor/scratch/cscs/$USER/finetuning/${JOB_NAME}_${SLURM_JOB_ID}"
export CONFIG_PATH=configs/training/hf_ape_10000.yaml

# Get master host address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Ask the OS for a free port for the master
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Get full path to python exec
export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

# Run the training with jobreport
srun $JOBREPORT monitor -o "$PROFILING_FOLDER" -- "$PYTHON_EXEC" training/training_trl.py --config-path "$CONFIG_PATH" --output-dir "$OUTPUT_DIR"

# Print report
$JOBREPORT print "$PROFILING_FOLDER"
