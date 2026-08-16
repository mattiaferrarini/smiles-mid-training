# Training

This folder contains the scripts for the continued pre-training of base models.

`training_trl.py` performs continued pre-training using the `trl` library:
- It supports distributed training via `accelerate`.
- It interleaves domain-specific datasets with general corpora (e.g., FineWeb).
- It handles hybrid tokenizers and resize/initialization of embeddings for new tokens.

`baselines.py` manages the retrieval of baseline model artifacts:
- It downloads model checkpoints and specific revisions directly from the Hugging Face Hub based on configuration.
- It parses the provided config dictionary to validate model names and revisions.
- It ensures the target directory exists and organizes downloaded artifacts into model-specific folders.