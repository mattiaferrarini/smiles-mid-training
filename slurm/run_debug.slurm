#!/bin/bash
#SBATCH --job-name=debug_sft
#SBATCH --ntasks-per-node=1            
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00             
#SBATCH --partition=normal
#SBATCH --nodes=1                       
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH -A a131

export ENV_NAME=ml4science

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh 

# Create or update conda environment (Standard check)
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

export PYTHONPATH="/users/$USER/smiles-mid-training/smiles-mid-training/:$PYTHONPATH"

# Distributed settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"
export WANDB_DISABLED=true

# HF cache folders
export HF_HOME="/iopsstor/scratch/cscs/$USER/cache/huggingface"
export HF_DATASETS_CACHE="/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

# Create needed folders
mkdir -p /iopsstor/scratch/cscs/$USER/logs
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface"

# Get Python Path
export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

echo "--- STARTING DEBUG RUN ---"

srun "$PYTHON_EXEC" -m accelerate.commands.launch \
    --num_processes=4 \
    --multi_gpu \
    main.py instruction-tuning \
    --config configs/instruction.yaml

echo "--- DEBUG RUN FINISHED ---"