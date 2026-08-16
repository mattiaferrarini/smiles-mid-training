# Evaluation

This folder contains scripts to evaluate tokenizers and models.

- `benchmark.py`: Evaluates instruction-tuned models on ChemBench benchmark.
- `likelihood_eval.py`: Performs likelihood-based evaluation of base models and checkpoints on ChemBench multiple-choice questions.
- `embeddings.py`: Computes model embeddings for SMILES strings in COCONUT dataset and generates t-SNE visualizations to check semantic clustering.
- `fertility.py`: Analyzes tokenizer efficiency by calculating fertility statistics (tokens per SMILES).

`tokenizer_configs\registry.json` contains the tokenizers that will be evaluated by the `fertility.py` script.