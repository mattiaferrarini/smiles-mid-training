# Training

This folder contains the script for the continued pre-training of base models.

`training_trl.py` performs continued pre-training using the `trl` library:
- It supports distributed training via `accelerate`.
- It interleaves domain-specific datasets with general corpora (e.g., FineWeb).
- It handles hybrid tokenizers and resize/initialization of embeddings for new tokens.