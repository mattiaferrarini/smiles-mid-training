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

LOGGER = get_logger(__name__)

TEXT_FIELD = "text_annotated_v1tags"


def get_smiles_list_from_dataset(dataset_path, text_field):
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
    )

    smiles_list = []
    for i in range(len(chem_dataset)):
        smiles_list.append(chem_dataset[i][text_field])

    return smiles_list


def evaluate_tokenizer(tokenizer, smiles_list):
    total_smiles = len(smiles_list)
    token_counts = []
    for smiles in smiles_list:
        tokens = tokenizer.tokenize(smiles)
        token_counts.append(len(tokens))

    average_tokens = sum(token_counts) / total_smiles if total_smiles > 0 else 0
    std = (sum((x - average_tokens) ** 2 for x in token_counts) / total_smiles) ** 0.5 if total_smiles > 0 else 0
    median_tokens = sorted(token_counts)[total_smiles // 2] if total_smiles > 0 else 0
    percentile_25 = sorted(token_counts)[int(0.25 * total_smiles)] if total_smiles > 0 else 0
    percentile_75 = sorted(token_counts)[int(0.75 * total_smiles)] if total_smiles > 0 else 0

    LOGGER.info(f"Total SMILES evaluated: {total_smiles}")
    LOGGER.info(f"Average number of tokens per SMILES: {average_tokens:.2f}")
    LOGGER.info(f"Median number of tokens per SMILES: {median_tokens}")
    LOGGER.info(f"25th percentile of tokens per SMILES: {percentile_25}")
    LOGGER.info(f"75th percentile of tokens per SMILES: {percentile_75}")

    return {
        "total_smiles": total_smiles,
        "average_tokens": average_tokens,
        "std_tokens": std,
        "median_tokens": median_tokens,
        "percentile_25": percentile_25,
        "percentile_75": percentile_75,
    }


def evaluate_tokenizers_fertility(registry_path, tokenizers_dir, dataset_path, output_folder):
    load_dotenv()
    with open(registry_path, "r") as f:
        configs = json.load(f)

    os.makedirs(output_folder, exist_ok=True)
    smiles = get_smiles_list_from_dataset(dataset_path, TEXT_FIELD)
    all_results = {}

    for config in configs:
        LOGGER.info(f"Evaluating fertility for tokenizer config: {config['name']}")
        config["tokenizer"]["output_dir"] = tokenizers_dir
        tokenizer = assemble_tokenizer(config)
        results = evaluate_tokenizer(tokenizer, smiles)
        all_results[config['name']] = results
    
    results_path = os.path.join(output_folder, "tokenizer_fertility_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    LOGGER.info(f"Fertility evaluation results saved to {results_path}")

