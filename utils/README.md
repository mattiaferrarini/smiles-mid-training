# Utils

This folder contains utility scripts for data processing, tokenizer management, and configuration.

- `config.py`: Handles YAML configuration loading and Hugging Face authentication.
- `create_smiles.py`: Implements regex-based logic and heuristics to validate and wrap SMILES strings with special tags.
- `helpers.py`: Utilities for managing, building, and saving tokenizer vocabularies and configuration files.
- `inspect_dataset.py`: Loads, samples, and inspects the training dataset to verify the SMILES annotation logic.
- `logging.py`: Logging configuration and setup.