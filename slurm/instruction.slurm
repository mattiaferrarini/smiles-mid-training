#!/bin/bash
#SBATCH --job-name=instruction_tuning
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH -A a131

export ENV_NAME=ml4science

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=5400
export NCCL_TIMEOUT=5400

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"

export HF_HOME="/iopsstor/scratch/cscs/$USER/cache/huggingface"
export HF_DATASETS_CACHE="/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

mkdir -p /iopsstor/scratch/cscs/$USER/logs
mkdir -p /iopsstor/scratch/cscs/$USER/jobreports
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface"
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

export JOBREPORT=/users/$USER/jobreport
export PROFILING_FOLDER=/iopsstor/scratch/cscs/$USER/jobreports/jobreport_${SLURM_JOB_ID}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

export CONFIG_PATH="configs/instruction.yaml"
export MODEL_PATH="/iopsstor/scratch/cscs/$USER/artifacts/baseliens/gemma-3-1b-pt"

srun $JOBREPORT monitor -o "$PROFILING_FOLDER" -- "$PYTHON_EXEC" main.py instruction-tuning --config "$CONFIG_PATH" --model-path "$MODEL_PATH"
$JOBREPORT print "$PROFILING_FOLDER"
