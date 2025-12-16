# Slurm Job Scripts

This folder contains the Slurm job scripts to execute experiments on the CSCS cluster. 

## Functionalities
The scripts make use of all the other code in the repository to:
1. Build tokenizers;
2. Perform continued pre-training of models;
3. Instruction-tune models;
4. Evaluate models on benchmarks;
5. Evaluate custom tokenizers;
6. Analyze the learned embeddings of the models.

## Script structure
All scripts have a similar structure:
1. A conda environment with the needed dependencies is activated;
2. Environment variables are set;
3. Some initialization operations are performed;
4. The core script is run;
5. Performance and resource utilization is logged using the Jobreport tool.

## Subfolders
- `tokenizers`: contains scripts to build custom tokenizers.
- `training`: contains scripts to train the models.
- `eval`: contains scripts to perform evaluation operations.

---

## How to Run Experiments

This section details the workflow for running the core experiments using the provided Slurm scripts. In general, you will need to modify specific environment variables within the `.slurm` file and/or the corresponding `.yaml` configuration file (in the folder `configs`) before submitting the job.

### Building Tokenizers
To generate a custom tokenizer vocabulary from your dataset:

1.  **Configure:** Navigate to `configs/tokenizers/` and edit the YAML file corresponding to the tokenizer you wish to build (e.g., `character.yaml`). Ensure the `data_folder` and `output_dir` paths are correct.
2.  **Run:** Submit the corresponding build script. There is a dedicated script for each tokenizer type. For example, to build the character tokenizer:
    ```bash
    sbatch slurm/tokenizers/build_character.slurm
    ```
    *Note: There are specific slurm scripts available for every tokenizer type following the same pattern.*

### Training
To perform continued pre-training (CPT) on the chemical dataset using the `trl` library:

1.  **Configure:** Edit `configs/default_trl.yaml`.
    * Set `tokenizer: type` and `chem_type` to match the tokenizer you built in step 1.
    * Adjust `data_mix` probabilities if you want to change the ratio of chemical data to general FineWeb data.
    * Set `distributed: strategy` (usually `ddp` or `fsdp`).
2.  **Run:** Point the Slurm script to your config and submit:
    ```bash
    # Inside training_trl.slurm, ensure CONFIG_PATH points to the right yaml
    sbatch slurm/training/training_trl.slurm
    ```
    

### Instruction Tuning
To perform Supervised Fine-Tuning (SFT) on the pre-trained models to align them for benchmarking:

1.  **Configure:** Edit `configs/instruction.yaml`.
    * Set `model: name` to the path of your checkpoint saved from the previous step (e.g., `/iopsstor/scratch/.../final_model`).
    * Verify `sciq_path` and `metamathqa_path` point to valid locations (the script will generate them if missing).
2.  **Run:**
    ```bash
    sbatch slurm/training/instruction.slurm
    ```

### Likelihood Evaluation (Pre-trained Models)
To evaluate a base or pre-trained model using the high-throughput likelihood method on ChemBench:

1.  **Configure:** Edit `slurm/eval/base_likelihood_eval.slurm`.
    * Set `MODEL_PATH` to the directory of the model you want to evaluate.
2.  **Run:**
    ```bash
    sbatch slurm/eval/base_likelihood_eval.slurm
    ```

### Benchmark Evaluation (Instruction-Tuned Models)
To perform the full ChemBench evaluation on an instruction-tuned model (generative evaluation):

1.  **Configure:** Edit `slurm/eval/benchmark.slurm`.
    * Set `MODEL_PATH` to the folder containing your instruction-tuned adapter/model.
    * Ensure `OUTPUT_DIR` is set to where you want the report JSONs to be saved.
2.  **Run:**
    ```bash
    sbatch slurm/eval/benchmark.slurm
    ```