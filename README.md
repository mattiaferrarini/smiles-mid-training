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
