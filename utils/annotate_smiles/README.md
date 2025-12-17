# Dataset Inspection & Annotation Scripts

This folder contains scripts to inspect text datasets, validate SMILES chemical representations using Regex and LLMs, and generate annotated datasets for training.

## Functionalities

1. **Annotate SMILES:** Scan large datasets and encapsule chemical strings between `[START_SMILES]` and `[END_SMILES]` tags.
2. **Inspect Performance:** Compare the Regex-based annotation against an existing Ground Truth to find False Positives/Negatives.
3. **Filter Noise:** Automatically exclude common English words (e.g., "FISH", "BOOKS") that resemble chemical formulas.
4. **Generate Datasets:** Save a clean, re-annotated version of the dataset for training.
5. **Annotation-Only Mode:** Run the annotation pipeline on raw text without requiring ground truth labels for comparison.

## Script Structure
* **`create_smiles.py`**: The core library containing regex patterns and chemical logic.
* **`inspect_dataset.py`**: The main script that runs the inspection/annotation.
* **`slurm/inspect_dataset.slurm`**: Job script for executing the inspection on the cluster.
* **`configs/utils/inspect_config.yaml`**: Configuration file for paths, flags, and model settings.

---

## How to Run

### 1. Configure
Create or edit the configuration file at `configs/utils/inspect_config.yaml`.
* Set `data_folder` to your dataset path.
* Set `whole_dataset: true` for full processing or `false` for a quick test.
* Set `save_new_dataset: true` if you want to save the output.
* Set `use_llm: true` if you want the choice of difficult strings to be done by a llm
* Set `compare_to_gt: true` if you want to compare your text with ground truth 

### 2. Run 
To run directly on a login node for testing:
```bash
python utils/inspect_dataset.py --config configs/utils/inspect_config.yaml
```

or, in the cluster
```bash
sbatch slurm/annotate_smiles/inspect_dataset.slurm
```