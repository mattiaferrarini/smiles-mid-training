#!/bin/bash
#SBATCH --job-name=gemma_finetuning
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
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
  conda create -n "$ENV_NAME" python=3.12 -y && \
  conda run -n "$ENV_NAME" pip install -r requirements.txt; \
fi

# Activate conda environment
conda activate "$ENV_NAME"

# Ensure the GPU version of PyTorch is installed
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# HF cache folders
export HF_HOME="/iopsstor/scratch/cscs/$USER/cache/huggingface"
export HF_DATASETS_CACHE="/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

# Create needed folders
mkdir -p /iopsstor/scratch/cscs/$USER/logs
mkdir -p /iopsstor/scratch/cscs/$USER/jobreports
mkdir -p /iopsstor/scratch/cscs/$USER/finetuning
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface"
mkdir -p "/iopsstor/scratch/cscs/$USER/cache/huggingface/datasets"

# Jobreport
export JOBREPORT=/users/$USER/jobreport
export PROFILING_FOLDER=/iopsstor/scratch/cscs/$USER/jobreports/jobreport_${SLURM_JOB_ID}

# Training settings
export MODEL_NAME=google/gemma-3-4b-it
export DATA_FOLDER=/capstor/store/cscs/swissai/a131/ML4Science/datasets/CEMB_v1_HF_FineWeb-chemV1
export OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/finetuning
export TEXT_FIELD=text

# Get master host address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Ask the OS for a free port for the master
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Get full path to python exec
export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

# Run the training with jobreport
srun $JOBREPORT monitor -o "$PROFILING_FOLDER" -- "$PYTHON_EXEC" training/training.py --model-name "$MODEL_NAME" --data-folder "$DATA_FOLDER" --output-dir "$OUTPUT_DIR" --text-field "$TEXT_FIELD"

$JOBREPORT print "$PROFILING_FOLDER"
