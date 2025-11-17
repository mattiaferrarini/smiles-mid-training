#!/bin/bash
#SBATCH --job-name=gemma_finetuning
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/$USER/logs/jobname_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/$USER/logs/jobname_%j.log
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

# Jobreport
export JOBREPORT=/users/$USER/jobreport
export PROFILING_FOLDER=/iopsstor/scratch/cscs/$USER/jobreports/jobreport_${SLURM_JOB_ID}

# Training settings
export MODEL_NAME=google/gemma-3-4b-it
export DATA_FOLDER=/capstor/store/cscs/swissai/a131/ML4Science/datasets/CEMB_v1_HF_FineWeb-chemV1
export OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/finetuning
export TEXT_FIELD=text

# Run the benchmark with jobreport monitoring
srun $JOBREPORT monitor -o "$PROFILING_FOLDER" -- python training/training.py --model-name "$MODEL_NAME" --data-folder "$DATA_FOLDER" --output-dir "$OUTPUT_DIR" --text-field "$TEXT_FIELD"

$JOBREPORT print "$PROFILING_FOLDER"