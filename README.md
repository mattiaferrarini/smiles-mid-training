# Project 2, Machine Learning (CS433), Fall 2025

## Authors 
- Mattia Ferrarini (407144)
- Francesco Monti (405682)
- Luca Prunotto (405496)

## Overview
Standard open-source LLMs often struggle with chemistry tasks. A primary reason is that they process chemical representations (such as SMILES strings) using general-purpose tokenizers, which frequently split molecules into semantically meaningless tokens

The goal of this project is to improve the chemical reasoning and understanding of LLMs through:
- Novel Tokenization Schemes: treating chemical entities such as atoms, functional groups, substructures... as distinct semantic units
- Mid-Stage Training: aligning the model's latent space with chemical concepts using large domain-specific datasets
- Smart Embedding Initialization: utilizing chemical priors like element properties to initialize new tokens

We validate these strategies by fine-tuning open-source models like gemma-3-1b and benchmarking them against industry standards like ChemBench for general chemistry

### Repository Structure
```text
smiles-mid-training/
    configs/                # YAML configurations for training, tokenizers, and evaluation
    custom_tokenizers/      # Implementations of different tokenizers
    embeddings/             # Logic for chemically-aware embedding initialization
    evaluate/               # Scripts for running ChemBench and likelihood-based evaluations
    json/                   # Vocabulary files, periodic table data, and tokenizer artifacts
    slurm/                  # SLURM scripts for distributed training on CSCS clusters
    training/               # Training scripts
    utils/                  # Helper functions for logging, config parsing, and SMILES processing
    main.py                 # Unified entry point - Typer CLI
    requirements.txt        # Project dependencies
```

### Tokenization

We implement specialized tokenizers to better represent molecular strings.For detailed implementation and usage, refer to the `custom_tokenizers/` folder.


The special tokenizers are applied exclusively to the smiles formula, that are between the tags [START_SMILES] and [END_SMILES]. The rest of the text is tokenized with Gemma 3 BaseTokenizer.

In order to use any tokenizer different from the base one to tokenize the smiles string, its vocabulary has to be built first. It is built either running `build_tokenizer.slurm` with the desired tokenizer in the field `['tokenizer']['chem_type']` of `configs/tokenizers/tokenizer` or running the dedicated slurm file. The script iterates on the whole dataset and builds a json with the tokenizer vocabulary. These vocabularies are already built and saved in `json/tokenizers`.

In order to choose the right tokenizer when the model is trained, the right config file has to be passed to the "training.py" file. The ready-to-use configs are in the folder `configs/tokenizers` and are passed to the file setting `['tokenizer']['chem_type']` as desired in the file `configs/training/defaul_trl.yaml`.


### Embeddings & Initialization

We implement three different embedding strategies for the embedding of the tokens created by the chemical tokenization.
For detailed implementation and usage, refer to the `embedding/` folder.
In order to choose the embedding strategy for the training, it has to be specified in `['tokenizer']['embedding_initialization']` in the file `configs/training/default_trl.yaml`.


### Fine-Tuning
We use the trl and accelerate libraries to perform continued pre-training and supervised fine-tuning 

TODO

### Instruction-Tuning

To improve the model's ability to answer question in a structured way, we perform Supervised Fine-Tuning (SFT) using **LoRA**. The `instruction.py` script manages the training on a dynamic mixture of:
- **General Instructions:** `trl-lib/Capybara` for conversational flow.
- **Scientific Reasoning:** `allenai/sciq` (reformatted for MCQA).
- **Math Reasoning:** `meta-math/MetaMathQA` (reformatted for numerical answers).

For detailed implementation and usage, refer to the `instruction/` folder.

### Evaluation & Benchmarking

To validate our model's performances, we employ a comprehensive evaluation suite of tools located in the `evaluate/` directory:
- **ChemBench**: We assess general chemistry knowledge using both generation-based (`benchmark.py`) and likelihood-based (`likelihood_eval.py`) scoring.
- **Tokenizer Efficiency**: The `fertility.py` script computes fertility statistics to measure how compactly different tokenizers represent SMILES strings.
- **Semantic Analysis**: We visualize the model's latent space using t-SNE (`embeddings.py`) to verify that chemically similar molecules cluster together.

For detailed implementation and usage, refer to the `evaluate/` folder.


## Project Setup and Usage Guide

#
Old Stuff below:
# smiles-mid-training

Create a virtual environment:

> ## Important
> You need a Hugging Face token to access Gemma. Get it on HF and create a ``.env`` with it following the structure of ``.env.example``.

--------------------------------------------

Create a virtual environment:
```
python3 -m venv .venv
```

Activate the virtual environemnt:
```
source .venv/bin/activate
```

Install requirements:
```
pip install -r requirements.txt
```

If you install new dependencies, do it **WITHIN** the environment and update the requirements file:
```
pip freeze > requirements.txt
```

---------------------------------------------

## Setting up cluster

### Cloning the repo

Create an ssh key on the cluster and add the public key to your [GitHub profile](https://github.com/settings/keys) to access the repo from the cluster.

Log into the cluster and clone the repo into your home folder:
```
git clone git@github.com:mattiaferrarini/smiles-mid-training.git
```

### Conda

To use environments and packages within a SLURM job, we need to use conda. I have found that the easiest way is to keep using venv locally and then use conda on the cluster.

You need to install (mini)conda into your home folder. You can follow the instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install) ensuring to install the version for **arch64**. Here's the summary:

1. Download the latest version of Miniconda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
```

2. (Optional) Anaconda recommends verifying the integrity of the installer after downloading it. Check you get the same sha from [here](https://repo.anaconda.com/miniconda/):
```
sha256sum Miniconda3-latest-Linux-aarch64.sh
```

3. Install Miniconda:
```
bash ~/Miniconda3-latest-Linux-aarch64.sh
```

4. Refresh the terminal:
```
source ~/.bashrc
```

### Conda and SLURM
My idea is to create and update conda environments directly within the SLURM jobs (see `slurm/baseline.slurm` for an example) as that seems the easiest way to run jobs without having to remember to do many things.

### Flash Attention

Since we are working on an `aarch64` architecture (Grace Hopper nodes), official pre-built wheels for Flash Attention are not available. We must build and install it from source. This allows the library to compile specifically for the cluster's GPU architecture.

**Warning:** This process involves compiling complex CUDA kernels and takes **approx. 40-60 minutes**. Do not run this on a login node; it will likely fail or get killed.

**Step-by-Step Build Instructions:**

1.  **Start an Interactive Session:**
    Request a GPU node for 1 hour to ensure the compilation has access to the GPU driver and sufficient resources.
    ```bash
    srun --partition=normal --nodes=1 --gres=gpu:1 --time=01:00:00 --pty /bin/bash
    ```

2.  **Prepare the Environment:**
    Activate your conda environment and install the necessary build tools. `ninja` is crucial to speed up compilation.
    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ml4science
    pip install packaging ninja setuptools
    ```

3.  **Verify Setup:**
    Ensure PyTorch is installed and `nvcc` (CUDA compiler) is accessible.
    ```bash
    python -c "import torch; print(torch.__version__, torch.version.cuda)"
    nvcc --version
    ```

4.  **Install Flash Attention:**
    Run the following command to compile and install the library directly. We use `--no-build-isolation` to force the builder to use the PyTorch version currently installed in your environment.
    ```bash
    # This command will take ~40 minutes. Do not interrupt it.
    pip install flash_attn --no-build-isolation
    ```

5.  **Verify Installation:**
    Run this Python one-liner to confirm the library can be imported and loaded correctly:
    ```bash
    python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} is successfully installed')"
    ```
    If this prints the version number without errors, the installation was successful.


### Jobreport
This needs to be installed in your home folder as well. Following [documentation](https://confluence.cscs.ch/spaces/KB/pages/862946109/Batch+Job+Summary+Report):

1. Get jobreport:
```
wget https://github.com/eth-cscs/alps-jobreport/releases/download/v0.1/jobreport
```

2. Make it executable:
```
chmod +x ./jobreport
```

---------------------------------------------

## Running SLURM jobs

> ### ! Important
> Remember to create your ``.env`` with your tokens as discussed above.

To run a slurm job:
```
sbatch slurm/baseline.slurm
```

To see the job queue:
```
squeue
```

To see an estimate start time of your submitted jobs that did not start (PD state = pending):
```
squeue --me --start
```

To see the state of jobs:
```
sacct
```

To cancel a job:
```
scancel [JOBID]
```

## How to write a SLURM job

You should take example from the baseline.slurm job (assuming it is somewhat correct).
Here I paste the main info from Jeremy:

### --job-name
To set!

### --partition
- **normal**: to run jobs 
- **debug**: to debug jobs **!!**

### --time 
I believe we need to change this to run long jobs.

### --nodes
To change to two for big jobs (eg training)

### --ntasks-per-node
Probably we need to set it = ``--gres=gpu`` to have one task per GPU.

### Jobreport
Just use it as in the examples we were given.
