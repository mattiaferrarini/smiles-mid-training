# Instruction-Tuning

This folder contains the script to fine-tune base models on instruction-following datasets.

`instruction.py` performs Supervised Fine-Tuning (SFT) on the model using a mixture of datasets.
- It automatically downloads, processes, and interleaves three datasets:
    - `trl-lib/Capybara` (General chat/instruction following).
    - `allenai/sciq` (Scientific multiple-choice QA, reformatted).
    - `meta-math/MetaMathQA` (Mathematical reasoning, reformatted).
- It allows custom hybrid tokenizers or standard HuggingFace tokenizers.
- It Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.
- It uses multi-GPU setups (DDP).