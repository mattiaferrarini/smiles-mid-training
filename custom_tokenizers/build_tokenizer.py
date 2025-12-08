import argparse
from pathlib import Path
from datasets import load_dataset

# 1. Importa tutte le classi tokenizer
from character_tokenizer import CharacterTokenizer 
from element_tokenizer import ElementTokenizer
from elementallparenthesis_tokenizer import ElementAllParenthesisTokenizer
from elementaromatics_tokenizer import ElementAromaticsTokenizer
from elementnoparenthesis_tokenizer import ElementNoParenthesisTokenizer
from elementrings_tokenizer import ElementRingsTokenizer
from selfies_tokenizer import SelfiesTokenizer
from smiles_bpe_tokenizer import SmilesBpeTokenizer
from ape_tokenizer import APETokenizer
from parallel_ape_tokenizer import ParallelAPETokenizer

from utils.helpers import build_and_save_tokenizer
from utils.config import load_config


CONFIG_PATH = "configs/tokenizer.yaml" 

# Parse command line arguments
parser = argparse.ArgumentParser(description="Build tokenizer")
parser.add_argument("--output-dir", type=str, required=True, help="Directory where to save the tokenizer")
parser.add_argument("--config", type=str, default="configs/tokenizer.yaml", help="Path to config file")

args = parser.parse_args()

config = load_config(args.config)

dataset = load_dataset(
    "arrow", 
    data_dir=config["data"]["data_folder"],
    # FIX: Use the pattern from config, default to **/*.arrow
    data_files=config["data"].get("data_files_pattern", "**/*.arrow"),
    split="train",
)


# Optionally use only a portion of the data for building the tokenizer
portion_of_data = config["data"].get("portion_of_data", 1.0)
if portion_of_data < 1.0:
    total_size = len(dataset)
    new_size = int(total_size * portion_of_data)
    print(f"[INFO] Using only {portion_of_data*100}% of the data: {new_size} out of {total_size} rows.")
    dataset = dataset.select(range(new_size))


# DEBUG: Limit to 100 rows for testing
#print("!!! DEBUG MODE: Using only first 100 rows !!!")
#dataset = dataset.select(range(100))

text_field = config["data"]["text_field"]
count_field = config["data"]["count_field"]
base_output_dir = args.output_dir
tokenizer_type = config["tokenizer"]["type"] 

# Handle case where type is 'base' or 'hybrid' but we want to build the chemical tokenizer defined in 'chem_type'
if (tokenizer_type == "base" or tokenizer_type == "hybrid") and "chem_type" in config["tokenizer"]:
    print(f"[INFO] Tokenizer type is '{tokenizer_type}'. Switching to build chemical tokenizer defined in 'chem_type': {config['tokenizer']['chem_type']}")
    tokenizer_type = config["tokenizer"]["chem_type"]

Path(base_output_dir).mkdir(parents=True, exist_ok=True)

tokenizerclass = None
output_subdir_name = config["tokenizer"].get("output_subdir_name", f"{tokenizer_type}_tokenizer")

tokenizer_classes = {
    "character": CharacterTokenizer,
    "element": ElementTokenizer,
    "elementallparenthesis": ElementAllParenthesisTokenizer,
    "elementaromatics": ElementAromaticsTokenizer,
    "elementnoparenthesis": ElementNoParenthesisTokenizer,
    "elementrings": ElementRingsTokenizer,
    "selfies": SelfiesTokenizer,
    "smiles_bpe": SmilesBpeTokenizer,
    "ape": APETokenizer,
    "parallel_ape": ParallelAPETokenizer,
}

if tokenizer_type in tokenizer_classes:
    tokenizerclass = tokenizer_classes[tokenizer_type]
else:
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


import re

if tokenizerclass:
    print(f"--- Starting build for tokenizer type: {tokenizer_type} ---")
    
    print("Filtering dataset to extract chemical segments (flattening to one SMILES per row)...")
    
    def extract_smiles_batch(batch):
        extracted_smiles = []
        pattern = r"\[START_SMILES\](.*?)\[END_SMILES\]"
        
        for text in batch[text_field]:
            if text:
                matches = re.findall(pattern, text, re.DOTALL)
                # Filter empty strings and strip whitespace
                valid_matches = [m.strip() for m in matches if m.strip()]
                extracted_smiles.extend(valid_matches)
        
        return {text_field: extracted_smiles}

    # Debug print of original
    print("\n[DEBUG] First 3 original examples:")
    for i in range(min(3, len(dataset))):
        print(f"Example {i}: {dataset[i][text_field]}...")

    # Apply transformation
    # We remove all columns from original dataset to avoid mismatch in lengths
    chem_dataset = dataset.map(
        extract_smiles_batch, 
        batched=True, 
        batch_size=10000,
        remove_columns=dataset.column_names,
        num_proc=64
    )

    print(f"\n[DEBUG] New dataset size: {len(chem_dataset)} rows.")
    print("[DEBUG] First 5 extracted SMILES:")
    for i in range(min(5, len(chem_dataset))):
        print(f"Row {i}: {chem_dataset[i][text_field]}")
    
    if len(chem_dataset) == 0:
        print("\n[WARNING] !!! NO CHEMICAL DATA FOUND !!!")
        print("The regex didn't match anything.")
    else:
        print(f"\n[INFO] Successfully extracted {len(chem_dataset)} SMILES segments.")

    build_and_save_tokenizer(
        TokenizerClass=tokenizerclass,
        dataset=chem_dataset, 
        text_field=text_field, 
        output_dir=f"{base_output_dir}/{output_subdir_name}",
        config=config
    )

    print("\nTokenizer built and saved.")