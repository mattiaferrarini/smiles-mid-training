# Data

This folder contains data used in the experiments.

- `sample_dataset` contains a small sample of the training dataset.
- `instruction-tuning` contains datatsets used for instruction-tuning.
- `sample_coconut` contains a small sample of the coconut dataset used for the evaluation of the models' embeddings.

If you would like to run the embedding evaluation, you can download the Coconut dataset with:
```
wget https://coconut.s3.uni-jena.de/prod/downloads/2025-12/coconut_csv-12-2025.zip
```

Then unzip it and move it to a folder name `coconut`.