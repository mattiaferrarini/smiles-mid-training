#!/bin/bash
#SBATCH --job-name=launch_experiments
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/%x_%j.log
#SBATCH -A a131

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

# List of chemical tokenizers to test (must match 'chem_type' options)
# Options: element, smiles_bpe, character, selfies, etc.
TOKENIZERS=("element" "smiles_bpe")

# List of embedding initialization strategies to test
# Options: random, average, elementwise
EMBEDDINGS=("random" "average")

# Base config file to use as a template
TEMPLATE_CONFIG="configs/default_trl.yaml"

# Directory where generated config files will be stored
CONFIG_DIR="configs/generated"
mkdir -p "$CONFIG_DIR"

# ==============================================================================
# LAUNCH LOOP
# ==============================================================================

echo "Starting experiment launcher..."
echo "Template config: $TEMPLATE_CONFIG"
echo "Output directory: $CONFIG_DIR"

for TOK in "${TOKENIZERS[@]}"; do
    for EMB in "${EMBEDDINGS[@]}"; do
        
        echo "--------------------------------------------------"
        echo "Preparing job for: Tokenizer=$TOK, Embedding=$EMB"
        
        # 1. Define new config filename
        CONFIG_NAME="${CONFIG_DIR}/train_${TOK}_${EMB}.yaml"
        
        # 2. Copy template to new config file
        cp "$TEMPLATE_CONFIG" "$CONFIG_NAME"
        
        # 3. Modify the config file using sed
        # We replace the values in the YAML file
        
        # Update chem_type
        # Matches "chem_type: <anything>" and replaces it
        sed -i "s/chem_type: .*/chem_type: $TOK/" "$CONFIG_NAME"
        
        # Update embedding_initialization
        # Matches "embedding_initialization: <anything>" and replaces it
        # We add quotes around the value just in case
        sed -i "s/embedding_initialization: .*/embedding_initialization: \"$EMB\"/" "$CONFIG_NAME"
        
        echo "Generated config: $CONFIG_NAME"
        
        # 4. Submit the training job
        # We pass the specific CONFIG_PATH to the training script via --export
        # This overrides the default in training_trl.slurm
        
        JOB_ID=$(sbatch --parsable --export=ALL,CONFIG_PATH="$CONFIG_NAME" slurm/training_trl.slurm)
        
        if [ $? -eq 0 ]; then
            echo "Submitted Training Job ID: $JOB_ID"
        else
            echo "FAILED to submit job for $TOK - $EMB"
        fi
        
    done
done

echo "--------------------------------------------------"
echo "All experiments submitted."
