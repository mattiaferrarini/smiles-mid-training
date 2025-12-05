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

# DEBUG: Limit to 100 rows for testing
print("!!! DEBUG MODE: Using only first 100 rows !!!")
dataset = dataset.select(range(100))

text_field = config["data"]["text_field"]
count_field = config["data"]["count_field"]
base_output_dir = args.output_dir
tokenizer_type = config["tokenizer"]["type"] 

# Handle case where type is 'base' but we want to build the chemical tokenizer defined in 'chem_type'
if tokenizer_type == "base" and "chem_type" in config["tokenizer"]:
    print(f"[INFO] Tokenizer type is 'base'. Switching to build chemical tokenizer defined in 'chem_type': {config['tokenizer']['chem_type']}")
    tokenizer_type = config["tokenizer"]["chem_type"]

Path(base_output_dir).mkdir(parents=True, exist_ok=True)



tokenizerclass = None
output_subdir_name = f"{tokenizer_type}_tokenizer"

if tokenizer_type == "character":
    tokenizerclass = CharacterTokenizer
elif tokenizer_type == "element":
    tokenizerclass = ElementTokenizer
elif tokenizer_type == "elementallparenthesis":
    tokenizerclass = ElementAllParenthesisTokenizer
elif tokenizer_type == "elementaromatics":
    tokenizerclass = ElementAromaticsTokenizer
elif tokenizer_type == "elementnoparenthesis":
    tokenizerclass = ElementNoParenthesisTokenizer
elif tokenizer_type == "elementrings":
    tokenizerclass = ElementRingsTokenizer
elif tokenizer_type == "selfies":
    tokenizerclass = SelfiesTokenizer
elif tokenizer_type == "smiles_bpe":
    tokenizerclass = SmilesBpeTokenizer
else:
    raise ValueError(f"Tipo di tokenizer non supportato nel file di configurazione: {tokenizer_type}")


import re

def extract_chem_segments(text):
    """Estrae solo il contenuto tra [CHEM] e [/CHEM]"""
    if text is None:
        return ""
    pattern = r"\[START_SMILES\](.*?)\[END_SMILES\]"
    matches = re.findall(pattern, text, re.DOTALL)
    return " ".join(matches)

if tokenizerclass:
    print(f"--- Starting build for tokenizer type: {tokenizer_type} ---")
    
    print("Filtering dataset to extract chemical segments...")
    
    def filter_chem_text(example):
        return {text_field: extract_chem_segments(example[text_field])}
    
    print("\n[DEBUG] First 3 original examples:")
    for i in range(min(3, len(dataset))):
        print(f"Example {i}: {dataset[i][text_field]}...")

    print("\n[DEBUG] First 3 filtered examples (what the tokenizer will see):")
    non_empty_count = 0
    chem_dataset = dataset.map(filter_chem_text, num_proc=4)
    num_chem_row = dataset[count_field]
    num_chem_row_sum = 0
    for i in range(min(100, len(chem_dataset))): # Controlla i primi 100
        text = chem_dataset[i][text_field]
        num_chem_row_sum += num_chem_row[i]
        if text.strip():
            non_empty_count += 1
            if non_empty_count <= 3:
                print(f"Filtered {i}: {text}...")
    
    if non_empty_count == 0:
        print(f"Number of chemical segments we should have found: {num_chem_row_sum}")
        print("\n[WARNING] !!! NO CHEMICAL DATA FOUND !!!")
        print("The regex didn't match anything in the first 100 examples.")
        print("Check if your dataset actually contains [START_SMILES] and [END_SMILES] tags.")
    else:
        print(f"\n[INFO] We found data in {non_empty_count}% of the first 100 examples checked.")

    
    
    build_and_save_tokenizer(
        TokenizerClass=tokenizerclass,
        dataset=chem_dataset, 
        text_field=text_field, 
        output_dir=f"{base_output_dir}/{output_subdir_name}" 
    )

    print("\nTokenizer built and saved.")