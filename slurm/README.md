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
    * There is one script per tokenizer type.
- `training`: contains scripts to train the models.
    * There is one script to run continued pre-training with each tokenizer type.
    * `instruction.slurm` can be used to perform LoRA instruction-tuning.
- `eval`: contains scripts to perform evaluation operations.
    * `likelihood_eval.slurm` performs likelihood evaluation of pre-trained models on MCQ from ChemBench.
    * `benchmark.slurm` runs ChemBench evalution of an instruction-tuned model.
    * `fertility.slurm` evaluates the fertility of custom tokenizers.
    * `embeddings.slurm` evaluates the learned chemical embeddings of pre-trained models via clustering.
---

## How to Run Experiments

This section details the workflow for running the core experiments using the provided Slurm scripts. In general, you will need to modify specific environment variables within the `.slurm` file and/or the corresponding `.yaml` configuration file (in the folder `configs`) before submitting the job.

### Building Tokenizers
To generate a custom tokenizer vocabulary from your dataset:

1.  **Configure:** Navigate to `configs/tokenizers/` and edit the YAML file corresponding to the tokenizer you wish to build (e.g., `character.yaml`). Ensure the `data_folder` and `output_dir` paths are correct.
2.  **Run:** From the base folder of the repository, submit the corresponding build script. There is a dedicated script for each tokenizer type. For example, to build the character tokenizer:
    ```bash
    sbatch slurm/tokenizers/build_character.slurm
    ```
    *Note: There are specific slurm scripts available for every tokenizer type following the same pattern.*
3. **Built Tokenizers**: All tokenizer artifacts are saved in `output_dir` specified in the config file from step 1. 


### Training
To perform continued pre-training (CPT) on the chemical dataset using the `trl` library:

1.  **Configure:** Edit `configs/default_trl.yaml`.
    * Set `tokenizer: type` and `chem_type` to match the tokenizer you previously built.
    * Adjust `data_mix` probabilities if you want to change the ratio of chemical data to general FineWeb data.
    * Set `distributed: strategy` (usually `ddp` or `fsdp`).
2.  **Run:** Point the Slurm script to your config and submit:
    ```bash
    # Inside training_trl.slurm, ensure CONFIG_PATH points to the right yaml
    sbatch slurm/training/training_trl.slurm
    ```
3. **Final Model**: The final model and all checkpoints are saved in `OUTPUT_DIR`, which can be set inside the SLURM script.
    

### Instruction Tuning
To perform Supervised Fine-Tuning (SFT) on the pre-trained models to align them for benchmarking:

1.  **Configure:** Edit `configs/instruction.yaml`.
    * Set `model: name` to the path of your checkpoint saved from the previous step (e.g., `/iopsstor/scratch/.../final_model`).
    * Verify `sciq_path` and `metamathqa_path` point to valid locations (the script will generate them if missing).
2.  **Run:** From the base folder of the repository:
    ```bash
    sbatch slurm/training/instruction.slurm
    ```
3. **LoRA Adapters**: LoRA adapters are saved in `output_dir` specified in the config file from step 1.


### Likelihood Evaluation (Pre-trained Models)
To evaluate a base or pre-trained model using the high-throughput likelihood method on ChemBench:

1.  **Configure:** Edit `slurm/eval/likelihood_eval.slurm`.
    * Set `MODEL_PATH` to the directory of the model you want to evaluate.
2.  **Run:** From the base folder of the repository:
    ```bash
    sbatch slurm/eval/likelihood_eval.slurm
    ```
3. **Results**: Evaluation results are saved in `OUTPUT_DIR`, which can be set inside the SLURM script.


### Benchmark Evaluation (Instruction-Tuned Models)
To perform the full ChemBench evaluation on an instruction-tuned model (generative evaluation):

1.  **Configure:** Edit `slurm/eval/benchmark.slurm`.
    * Set `MODEL_PATH` to the folder containing your instruction-tuned adapter/model.
    * Ensure `OUTPUT_DIR` is set to where you want the report JSONs to be saved.
2.  **Run:** From the base folder of the repository:
    ```bash
    sbatch slurm/eval/benchmark.slurm
    ```
3. **Results**: Benchmarks results are saved in `OUTPUT_DIR`, which can be set inside the SLURM script.


### Fertility Evaluation
To evaluate the fertility of built tokenizers:

1. **Configure:** Edit `evaluate/tokenizer_configs/registry.json`.
    * Include configurations for the tokenizers you want to evaluate. 
    * Simply duplicate one of the entries and set the required fields following the same approach of `configs/tokenizers/`.
2. **Run:** From the base folder of the repository:
    ```bash
    sbatch slurm/eval/fertility.slurm
    ```
3. **Results**: Results are saved in `OUTPUT_FOLDER`, which can be set inside the SLURM script.


### Embedding Evaluation
To evaluate the chemical embeddings of a base model or a pre-trained one:

1. **Configure:** Edit `slurm/eval/embeddings`.
    * Set `CHECKPOINT_FOLDER` to the folder where the pre-trained model is saved or to a HuggingFace ID.
    * Set `DATASET_PATH` to point to the dataset you want to use for evaluation, and `SMILES_COL` and `LABEL_COL` to the names to the columns containing the SMILES strings and the corresponding classes.
    * Set `PLOT_TITLE` to the title you would like to have on the final plot.
2. **Run:** From the base folder of the repository:
    ```bash
    sbatch slurm/eval/embeddings.slurm
    ```
3. **Results**: Results are saved in `OUTPUT_PATH`, which can be set inside the SLURM script.