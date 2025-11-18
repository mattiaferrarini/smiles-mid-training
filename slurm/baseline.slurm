#!/bin/bash
#SBATCH --job-name=gemma_chembench
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
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

# Jobreport
export JOBREPORT=/users/$USER/jobreport
export PROFILING_FOLDER=/iopsstor/scratch/cscs/$USER/jobreports/jobreport_${SLURM_JOB_ID}

# Benchmark settings
export MODEL_NAME=google/gemma-3-4b-it
export REPORT_FOLDER=/iopsstor/scratch/cscs/$USER/baseline

# Get full path to python exec
export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

# Run the benchmark with jobreport monitoring
srun $JOBREPORT monitor -o "$PROFILING_FOLDER" -- "$PYTHON_EXEC" baseline/baseline.py -m "$MODEL_NAME" -r "$REPORT_FOLDER" 

$JOBREPORT print "$PROFILING_FOLDER"
