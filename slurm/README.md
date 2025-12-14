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