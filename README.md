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

## Benchmarks
To be able to run ChemIQ benchmarks, make sure to clone the repository inside `smiles-mid-training` with:

`git clone https://github.com/oxpig/ChemIQ.git`
