# Novel Tokenization Schemes and Mid-Stage Training for LLM Chemical Knowledge - Machine Learning (CS433), Fall 2025

## Authors 
- Mattia Ferrarini (407144)
- Francesco Monti (405682)
- Luca Prunotto (405496)

## Overview
Standard open-source LLMs often struggle with chemistry tasks. Two of the primary reasons are lack of chemical knowledge in pre-training data and that they process chemical representations (such as SMILES strings) using general-purpose tokenizers, which frequently split molecules into semantically meaningless tokens

The goal of this project is to improve the chemical reasoning and understanding of LLMs through:
1.  **Chemical Tokenization**: Implementing novel tokenization schemes that treat chemical entities as distinct semantic units
2.  **Smart Embedding Initialization**: Bridging the gap between new tokens and the pre-trained model by initializing embeddings using chemical priors
3.  **Mid-Stage Training**: Continued pre-training on large, domain-specific datasets to align the model's latent space with these new chemical representations
4.  **Instruction Tuning**: Fine-tuning the adapted models on scientific QA and reasoning tasks 

We validate these strategies by fine-tuning open-source models like gemma-3-1b and benchmarking them against industry standards like ChemBench for general chemistry

## Repository Overview
```text
smiles-mid-training/
    configs/                # YAML configs for tokenizer building, training, instruction tuning, and evaluation
    custom_tokenizers/      # Implementations of different tokenizers
    data/                   # Sample datasets
    embeddings/             # Embedding initialization strategies
    evaluate/               # ChemBench, likelihood eval, fertility, embeddings viz...
    instructions/           # Instruction-Tuning (SFT/LoRA)
    json/                   # Tokenizer artifacts and chemistry resources
    slurm/                  # CSCS clusters scripts for all major tasks 
    training/               # CPT scripts
    utils/                  # Helper functions for logging, config parsing, and SMILES processing
    main.py                 # Unified entry point - Typer CLI
    requirements.txt        # Project dependencies
    .env.example            # Example of .env
```

See folder-specific README.md files for additional information. 

## Installation and Setup
All experiments in this repo are intended to be executed on the CSCS cluster 

### Clone on CSCS
Create an SSH key on the cluster and add the public key to your GitHub profile to access the report fom the cluster. Log into the cluster and clone the repo in your home folder:
```bash
git clone git@github.com:..."
cd smiles-mid-training
```

If you are not able to clone the repository on the cluster directly, simply clone it locally and then secure copy it to CSCS:

```bash
scp -r smiles-mid-training clariden:/desired/path/on/CSCS
```

### Environment Variables (`.env`)
Some API keys are needed in order to access the model on HuggingFace and log runs on Wandb. Create a `.env` file and add your keys.The expected content of the file is available in `.env.local` and shown below.

```
HF_TOKEN="your_huggingface_token"
WANDB_API_KEY="your_wandb_key_here"
WANDB_PROJECT="your_wandb_project_name_here"
```

### Conda Environment
The SLURM scripts assume an environment named `ml4science` with all the necessary packages installed

1. Install Miniconda (aarch64)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
sha256sum Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
source ~/.bashrc
```

2. Create the Environment and Install Requirements
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n ml4science --override-channels -c conda-forge python=3.12 -y
conda activate ml4science
pip install -r requirements.txt
```

#### Flash Attention
Since we are working on an aarch64 architecture (Grace Hopper nodes), official pre-built wheels for Flash Attention are not available. We must build and install it from source. This allows the library to compile specifically for the cluster's GPU architecture

**Warning:** This process involves compiling complex CUDA kernels and takes approximately 40-60 minutes

**Step-by-Step Build Instructions:**


```bash
srun --partition=normal --nodes=1 --gres=gpu:1 --time=01:00:00 --pty /bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml4science
pip install packaging ninja setuptools
pip install flash_attn --no-build-isolation
python -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} OK')"
```


**Step-by-Step Build Instructions:**
These are the steps that need to be executed. It is advised to add the commands to a SLURM job so that it can be installed using CSCS GPUs. Otherwise, on CPU, it will most likely fail.

1.  **Prepare the Environment:**
    Activate your conda environment and install the necessary build tools. `ninja` is crucial to speed up compilation.
    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ml4science
    pip install packaging ninja setuptools
    ```

2.  **Verify Setup:**
    Ensure PyTorch is installed and `nvcc` (CUDA compiler) is accessible.
    ```bash
    python -c "import torch; print(torch.__version__, torch.version.cuda)"
    nvcc --version
    ```

3.  **Install Flash Attention:**
    Run the following command to compile and install the library directly. We use `--no-build-isolation` to force the builder to use the PyTorch version currently installed in your environment.
    ```bash
    # This command will take ~40 minutes. Do not interrupt it.
    pip install flash_attn --no-build-isolation
    ```

4.  **Verify Installation:**
    Run this Python one-liner to confirm the library can be imported and loaded correctly:
    ```bash
    python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} is successfully installed')"
    ```
    If this prints the version number without errors, the installation was successful.

#### Jobreport
Install `jobreport` in your home directory. Following documentation from (https://confluence.cscs.ch/spaces/KB/pages/862946109/Batch+Job+Summary+Report):
```bash
wget https://github.com/eth-cscs/alps-jobreport/releases/download/v0.1/jobreport
chmod +x ./jobreport
```

## Some Useful SLURM Commands

1. To run a slurm job:
```
sbatch slurm/baseline.slurm
```

2. To see the job queue:
```
squeue
```

3. To see an estimate start time of your submitted jobs that did not start (PD state = pending):
```
squeue --me --start
```

4. To see the state of jobs:
```
sacct
```

5. To cancel a job:
```
scancel [JOBID]
```

## General Info

### Tokenization

We implement specialized tokenizers to better represent molecular strings. For detailed implementation and usage, refer to the `custom_tokenizers/` folder.


The special tokenizers are applied exclusively to the SMILES formula, that are between the tags [START_SMILES] and [END_SMILES]. The rest of the text is tokenized with Gemma 3 BaseTokenizer.

In order to use any tokenizer different from the base one to tokenize the SMILES string, its vocabulary has to be built first. It is built either running `build_tokenizer.slurm` with the desired tokenizer in the field `['tokenizer']['chem_type']` of `configs/tokenizers/tokenizer` or running the dedicated slurm file. The script iterates on the whole dataset and builds a json with the tokenizer vocabulary. These vocabularies are already built and saved in `json/tokenizers`.

In order to choose the right tokenizer when the model is trained, the right config file has to be passed to the "training.py" file. The ready-to-use configs are in the folder `configs/tokenizers` and are passed to the file setting `['tokenizer']['chem_type']` as desired in the file `configs/training/defaul_trl.yaml`.


### Embeddings & Initialization

We implement three different embedding strategies for the embedding of the tokens created by the chemical tokenization.
For detailed implementation and usage, refer to the `embedding/` folder.
In order to choose the embedding strategy for the training, it has to be specified in `['tokenizer']['embedding_initialization']` in the file `configs/training/default_trl.yaml`.


### Continued Pre-Training (CPT)

We perform Continued Pre-Training (CPT) to teach the base LLM (gemma-3-1b) our new chemical language. By training on a mix of specialized chemical data and general text, we help the model understand our new chemical tokens while preserving its existing knowledge.

We use the Hugging Face `trl` and `accelerate` libraries for efficient, distributed training on the CSCS cluster.

**Key Features:**
- **Dynamic Data Mixing:** The training script supports mixing chemical data with general text (FineWeb) to prevent catastrophic forgetting.
- **Custom Tokenizer Integration:** It swaps the standard tokenizer for our chemical-aware implementations specifically for content within `[START_SMILES]` tags.
- **Embedding Resizing:** It automatically resizes the model's embedding layer to accommodate the new vocabulary and initializes new tokens using the strategies defined in `embeddings/`.

**Configuration:**
The main configuration file is in `configs/training`. There is a config file per tokenizer Key parameters include:
- `tokenizer.chem_type`: Selects the tokenizer type matching the vocabulary built in the previous step.
- `data_mix`: Controls the ratio of chemical vs. general data.
- `distributed.strategy`: Sets up `ddp` or `fsdp` for multi-GPU training.

**Running CPT:**
We prepared a dedicated script for training every tokenizer they are in `slurm/training`
To launch the training job on the cluster, use the desired slurm file:
```bash
sbatch slurm/training/training_bpe.slurm
```

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
