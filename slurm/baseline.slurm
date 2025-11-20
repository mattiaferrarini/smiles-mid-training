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
  conda create -n "$ENV_NAME" --override-channels -c conda-forge python=3.12 -y && \
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

# Configuration
export CONFIG_PATH=configs/default.yaml
export BASELINES_DIR=/iopsstor/scratch/cscs/$USER/artifacts/baselines
export REPORT_DIR=/iopsstor/scratch/cscs/$USER/baseline_results

# Get full path to python exec
export PYTHON_EXEC=$(which python)
echo "Using Python at: $PYTHON_EXEC"

# 1. Download model(s) defined in config
echo "Downloading baselines..."
srun "$PYTHON_EXEC" main.py get-baseline \
    --config "$CONFIG_PATH" \
    --output-dir "$BASELINES_DIR"

# 2. Run benchmark
echo "Running benchmark..."
export CHEMIQ_COMMAND="$PYTHON_EXEC benchmarks/chemiq_runner.py --model-path {model_dir} --output-dir {output_dir}"

srun $JOBREPORT monitor -o "$PROFILING_FOLDER" -- "$PYTHON_EXEC" main.py benchmark-baselines \
    --config "$CONFIG_PATH" \
    --baselines-root "$BASELINES_DIR" \
    --output-dir "$REPORT_DIR" \
    --chemiq-command "\"$CHEMIQ_COMMAND\""

$JOBREPORT print "$PROFILING_FOLDER"
