# Configs

This folder contains configuration files to build tokenizers and to train models. The configurations are specified as YAML and split into three categories: tokenizer construction, continued pre-training, instruction-tuning.

## Tokenizer Building Configurations

Files used to train and generate new tokenizers.

They are structured to contain:
- `job`: Name of the tokenizer building experiment.
- `data`: Source data used to learn the vocabulary.
    - `data_folder`: Path to the dataset directory.
    - `text_field`: The specific column in the data files to process.
- `tokenizer`: Tokenizer parameters.
    - `chem_type`: The specific tokenization method.
    - `output_dir` and `output_subdir_name`: Locations where the resulting JSON files are saved.
    - `params`: Tokenizer-specific parameters that will be used in initialization.

## Continued Pretraining Configurations

Files used to configure the training loop, environment, and hyperparameters.

They are structure to contain:
- `job`: Name of the training experiment.
- `model`: Base model path or HF ID.
- `data`: Training dataset configuration.
    - Includes `data_folder`, file patterns, and the `portion_of_data_used`.
- `data_mix`: Mixing of external datasets with the main dataset.
    - `probabilities`: Sampling ratio between the main and external datasets.
- `distributed`: Settings for parallel training strategies.
    - `strategy`: Supports `ddp` (Distributed Data Parallel) or `fsdp` (Fully Sharded Data Parallel).
- `training`: Training hyperparameters.
    - Includes `learning_rate`, `epochs`, `max_steps`, `warmup_ratio`, and `weight_decay`.
    - Controls precision (`bf16`, `fp16`), logging (`wandb`), and checkpointing frequency.
- `tokenizer`: Tokenizer specification.
    - `type`: Strategy type.
        - `base`: Base model tokenizer.
        - `hrybid`: Hybrid tokenizer with a chemical (sub-)tokenizer for SMILES strings.
        - `chem_only`: Only chemical tokenizer.
    - `chem_type`: Specific chemical (sub-)tokenizer to use.
    - `embedding_initialization`: Strategy to initialize embeddings for new tokens.
    - `special_tokens`: Dictionary of domain-specific control tokens.

## Instruction-Tuning Configurations

Files used to configure instruction-tuning.

They are structured to contain:
- `model`: Base model specification.
    - `name`: The Hugging Face ID or path to the base model.
- `data`: Instruction dataset configuration.
    - Includes paths or IDs for specific datasets (e.g., `chat_dataset`, `sciq_path`, `metamathqa_path`).
- `training`: Training hyperparameters and environment settings.
    - `output_dir`: Directory where the fine-tuned model and logs will be saved.
    - Includes `epochs`, `batch_size`, `learning_rate`, `warmup_steps`, `gradient_accumulation_steps`, and `max_length`.
    - Controls precision (`bf16`, `fp16`), logging (`logging_steps`), and checkpointing (`save_steps`).
- `peft`: Parameter-Efficient Fine-Tuning settings.
    - `use_lora`: Boolean flag to enable LoRA (Low-Rank Adaptation).
    - Includes LoRA-specific parameters: `lora_r`, `lora_alpha`, `lora_dropout`.
    - `lora_target_modules`: List of specific model layers to apply adaptation (e.g., `q_proj`, `k_proj`, etc.).