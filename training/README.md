# Unified Training Pipeline

A flexible, config-driven training pipeline that supports both baseline and advanced training modes.

## Features

- **Dual Mode Operation**: Automatically switches between baseline and advanced modes based on config
- **Config-Driven**: All hyperparameters managed through YAML configuration files
- **Custom Tokenizers**: Support for element and hybrid tokenizers (advanced mode)
- **Embedding Strategies**: Multiple initialization strategies for new tokens (advanced mode)
- **Distributed Training**: Built-in support for multi-GPU training with Accelerate
- **Optional W&B Integration**: Automatic Weights & Biases logging when API key is provided
- **Cluster Support**: Jobreport integration for SLURM environments

## Training Modes

### Baseline Mode
Simple training with standard HuggingFace components. No tokenizer or embedding modifications.

**Use cases:**
- Quick benchmarking
- Standard fine-tuning
- Baseline comparisons

**Triggered when:** Config does NOT contain custom `tokenizer.type` or `embeddings.strategy`

### Advanced Mode
Custom tokenizers with chemical element support and embedding initialization strategies.

**Use cases:**
- Domain-specific tokenization (chemistry, math, etc.)
- Embedding initialization experiments
- Vocabulary expansion with proper initialization

**Triggered when:** Config contains `tokenizer.type` (element/hybrid) OR `embeddings.strategy`

## Quick Start

### 1. Baseline Training

```bash
python main.py train --config configs/baseline.yaml
```

### 2. Advanced Training

```bash
python main.py train --config configs/advanced.yaml
```

### 3. Override Config via CLI

```bash
python main.py train \
    --config configs/default.yaml \
    --model-name google/gemma-2b \
    --data-folder data/my_dataset \
    --output-dir artifacts/runs/my_experiment
```

### 4. Get Help

```bash
python main.py train --help
```

## Configuration

### Required Sections

```yaml
model:
  name: google/gemma-2b  # HuggingFace model ID

data:
  data_folder: data/processed  # Path to arrow files
  text_field: text             # Field to train on
```

### Training Parameters (Optional)

```yaml
training:
  output_dir: artifacts/runs/default
  epochs: 2
  batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  warmup_steps: 100
  weight_decay: 0.0
  save_steps: 1000
  logging_steps: 50
  bf16: true
  fp16: false
  gradient_checkpointing: false
  seed: 42
```

### Advanced Mode: Custom Tokenizers (Optional)

```yaml
tokenizer:
  type: element  # Options: 'element', 'hybrid'
  include_element_tokens: true
  extra_tokens: []
  
  # For hybrid tokenizer:
  hybrid_params:
    chem_start: "[CHEM]"
    chem_end: "[/CHEM]"
```

### Advanced Mode: Embedding Strategies (Optional)

```yaml
embeddings:
  strategy: mean_std  # Options: 'mean_std', 'average', 'normal', 'zero'
  params:
    min_std: 1.0e-5   # Strategy-specific parameters
    std_scale: 1.0
```

## Environment Setup

### Required Environment Variables

```bash
# .env file
HF_TOKEN=your_huggingface_token  # For authenticated model access
WANDB_API_KEY=your_wandb_key     # Optional: for W&B logging
```

### SLURM Environment

The pipeline automatically detects SLURM environments and configures distributed training:

```bash
# Automatic detection of:
# - SLURM_PROCID → RANK
# - SLURM_LOCALID → LOCAL_RANK  
# - SLURM_NTASKS → WORLD_SIZE
```

## Output Structure

Each training run creates a timestamped directory:

```
artifacts/runs/experiment_name/
├── 20250120_143022/              # Timestamp
│   ├── config.yaml               # Saved config for reproducibility
│   ├── wandb_run.json            # W&B run info (if enabled)
│   ├── checkpoint-1000/          # Training checkpoints
│   ├── final-model/              # Final saved model
│   └── jobreport/                # Cluster resource reports (if available)
│       ├── jobid-start.json
│       └── jobid-end.json
```

## Examples

### Example 1: Quick Baseline Benchmark

```yaml
# configs/quick_baseline.yaml
model:
  name: google/gemma-2b
data:
  data_folder: data/benchmark
  text_field: text
training:
  epochs: 1
  batch_size: 8
```

Run: `python main.py train -c configs/quick_baseline.yaml`

### Example 2: Advanced with Element Tokenizer

```yaml
# configs/element_exp.yaml
model:
  name: google/gemma-2b
data:
  data_folder: data/chemistry
  text_field: text
tokenizer:
  type: element
  include_element_tokens: true
embeddings:
  strategy: mean_std
training:
  epochs: 3
  gradient_checkpointing: true
```

Run: `python main.py train -c configs/element_exp.yaml`

## CLI Reference

```bash
python main.py train [OPTIONS]

Options:
  -c, --config PATH          Config file path (default: configs/default.yaml)
  -m, --model-name NAME      Model name (overrides config)
  -d, --data-folder PATH     Data folder (overrides config)
  -o, --output-dir PATH      Output directory (overrides config)
  --help                     Show help message and exit
```

The training command is integrated into the main CLI via Typer, providing a consistent interface with other commands like `tokenizer-train`, `embedding-evaluate`, etc.

## Dataset Format

The pipeline expects arrow format datasets:

```
data/processed/
├── train-00000.arrow
├── train-00001.arrow
└── ...
```

Each arrow file should contain a text field (configurable) with training examples.

## Logging

The pipeline uses Python's logging system. Logs include:

- Training mode detection (baseline/advanced)
- Model and tokenizer loading
- Embedding resizing (if applicable)
- Training progress
- W&B integration status
- Cluster resource information

Example log output:

```
INFO: ================================================================================
INFO: UNIFIED TRAINING PIPELINE
INFO: ================================================================================
INFO: Mode: ADVANCED
INFO: Model: google/gemma-2b
INFO: Data: data/chemistry
INFO: Config: configs/advanced.yaml
INFO: Distributed: DistributedType.MULTI_GPU
INFO: Process: 0/4
INFO: ================================================================================
INFO: Loading advanced tokenizer with custom configuration
INFO: Loading model google/gemma-2b
INFO: Initializing 118 new embeddings with strategy: mean_std
INFO: Resized embeddings from 256000 to 256118
INFO: Training for 3 epochs with batch size 2
```

## Migration from Old Training Scripts

If you have existing training scripts:

### From training2.py (Simple):
Replace hardcoded constants with config file. The pipeline provides the same functionality with more flexibility.

### From old training.py (Complex):
The new pipeline already includes all features:
- Config-based hyperparameters ✓
- W&B integration ✓
- Jobreport support ✓
- Custom tokenizers ✓
- Embedding strategies ✓

Simply use the new unified `training.py` with appropriate config files.

## Troubleshooting

### Issue: "Config must contain model.name"
**Solution:** Ensure your config has a `model.name` field.

### Issue: "Tokenizer type 'X' not found"
**Solution:** Use valid tokenizer types: 'element' or 'hybrid'

### Issue: W&B not logging
**Solution:** Set `WANDB_API_KEY` in your `.env` file

### Issue: Out of memory
**Solution:** Enable gradient checkpointing in config:
```yaml
training:
  gradient_checkpointing: true
```

Or reduce batch size / increase gradient accumulation steps.
