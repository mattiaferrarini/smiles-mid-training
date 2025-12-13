#!/bin/bash
#SBATCH --job-name=fertility_eval
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
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

# Fertility evaluation settings
export OUTPUT_FOLDER="/iopsstor/scratch/cscs/$USER/fertility_eval"
export REGISTRY_PATH="evaluate/tokenizer_configs/registry.json"
export DATASET_PATH="/capstor/store/cscs/swissai/a131/ML4Science/datasets/CEMB_v1tags_HF_FineWeb-chemV1"

# Run fertility evaluation
python main.py fertility-eval \
    --registry-path "$REGISTRY_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-folder "$OUTPUT_FOLDER"

