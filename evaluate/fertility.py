import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_tokenizers.assemble_tokenizer import assemble_tokenizer
import re
from datasets import load_dataset
from dotenv import load_dotenv
from utils.logging import get_logger
import random

LOGGER = get_logger(__name__)

TEXT_FIELD = "text_annotated_v1tags"


def get_smiles_list_from_dataset(dataset_path, text_field, target=1_000_000):
    """
    Extracts SMILES strings from a dataset stored in Arrow format.
    Args:
        dataset_path (str): Path to the dataset directory.
        text_field (str): Name of the text field containing annotated SMILES.
    Returns:
        list: A list of extracted SMILES strings.
    """
    dataset = load_dataset(
        "arrow",
        data_dir=dataset_path,
        data_files="**/*.arrow",
        split="train",
    )

    def extract_smiles_batch(batch):
        extracted_smiles = []
        pattern = r"\[START_SMILES\](.*?)\[END_SMILES\]"

        for text in batch[text_field]:
            if text:
                matches = re.findall(pattern, text, re.DOTALL)
                valid_matches = [m.strip() for m in matches if m.strip()]
                extracted_smiles.extend(valid_matches)

        return {text_field: extracted_smiles}

    chem_dataset = dataset.map(
        extract_smiles_batch,
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
        num_proc=8,
        load_from_cache_file=False,
        desc="Extracting SMILES",
    )

    if len(chem_dataset) > target:
        LOGGER.info(f"Shuffling and selecting {target} samples.")
        sampled_dataset = chem_dataset.shuffle(seed=42).select(range(target))
        smiles = sampled_dataset[TEXT_FIELD]
    else:
        smiles = chem_dataset[TEXT_FIELD]

    return smiles


def evaluate_tokenizer(tokenizer, smiles_list):
    """
    Evaluates the tokenizer's performance on a list of SMILES strings.
    
    Args:
        tokenizer: The tokenizer to evaluate.
        smiles_list (list): A list of SMILES strings.
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    total_smiles = len(smiles_list)
    token_counts = []
    for smiles in smiles_list:
        tokens = tokenizer.tokenize(smiles)
        if len(tokens) == 0:
            LOGGER.info(tokens)
            LOGGER.info(smiles)
            raise ValueError(f"{tokens}: {smiles}")
        token_counts.append(len(tokens))

    average_tokens = sum(token_counts) / total_smiles if total_smiles > 0 else 0
    std = (sum((x - average_tokens) ** 2 for x in token_counts) / total_smiles) ** 0.5 if total_smiles > 0 else 0
    median_tokens = sorted(token_counts)[total_smiles // 2] if total_smiles > 0 else 0
    percentile_25 = sorted(token_counts)[int(0.25 * total_smiles)] if total_smiles > 0 else 0
    percentile_75 = sorted(token_counts)[int(0.75 * total_smiles)] if total_smiles > 0 else 0
    max_tokens = max(token_counts) if total_smiles > 0 else 0
    min_tokens = min(token_counts) if total_smiles > 0 else 0

    LOGGER.info(f"Total SMILES evaluated: {total_smiles}")
    LOGGER.info(f"Total tokens counted: {sum(token_counts)}")
    LOGGER.info(f"Standard deviation of tokens per SMILES: {std:.2f}")
    LOGGER.info(f"Average number of tokens per SMILES: {average_tokens:.2f}")
    LOGGER.info(f"Median number of tokens per SMILES: {median_tokens}")
    LOGGER.info(f"25th percentile of tokens per SMILES: {percentile_25}")
    LOGGER.info(f"75th percentile of tokens per SMILES: {percentile_75}")
    LOGGER.info(f"Max tokens in a SMILES: {max_tokens}")
    LOGGER.info(f"Min tokens in a SMILES: {min_tokens}")

    return {
        "total_smiles": total_smiles,
        "total_tokens": sum(token_counts),
        "average_tokens": average_tokens,
        "std_tokens": std,
        "median_tokens": median_tokens,
        "percentile_25": percentile_25,
        "percentile_75": percentile_75,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
    }


def evaluate_tokenizers_fertility(registry_path, tokenizers_folder, dataset_path, output_folder, target=1_000_000):
    """
    Evaluates the fertility of multiple tokenizers defined in a registry file on a dataset of SMILES strings.
    Args:
        registry_path (str): Path to the tokenizer registry JSON file.
        tokenizers_folder (str): Folder where tokenizers are stored.
        dataset_path (str): Path to the dataset containing SMILES strings.
        output_folder (str): Folder to save the evaluation results.
    Returns:
        None
    """
    load_dotenv()
    with open(registry_path, "r") as f:
        configs = json.load(f)

    LOGGER.info(f"Loaded {len(configs)} tokenizer configurations from registry.")

    os.makedirs(output_folder, exist_ok=True)
    smiles = get_smiles_list_from_dataset(dataset_path, TEXT_FIELD, target=1_000_000)
    LOGGER.info(f"Extracted {len(smiles)} SMILES entries from dataset.")
    
    all_results = {}

    for config in configs:
        LOGGER.info(f"Evaluating fertility for tokenizer config: {config['name']}")
        config["tokenizer"]["output_dir"] = tokenizers_folder
        tokenizer = assemble_tokenizer(config)
        results = evaluate_tokenizer(tokenizer, smiles)
        all_results[config['name']] = results
    
    results_path = os.path.join(output_folder, "tokenizer_fertility_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    LOGGER.info(f"Fertility evaluation results saved to {results_path}")

