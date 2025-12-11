import argparse
import re
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
from apewordpiece_tokenizer import APEWordPieceTokenizer

from utils.helpers import build_and_save_tokenizer
from utils.config import load_config

TOKENIZER_CLASSES = {
    "character": CharacterTokenizer,
    "element": ElementTokenizer,
    "elementallparenthesis": ElementAllParenthesisTokenizer,
    "elementaromatics": ElementAromaticsTokenizer,
    "elementnoparenthesis": ElementNoParenthesisTokenizer,
    "elementrings": ElementRingsTokenizer,
    "selfies": SelfiesTokenizer,
    "smiles_bpe": SmilesBpeTokenizer,
    "ape_wordpiece": APEWordPieceTokenizer
}

CONFIG_PATH = "configs/tokenizer.yaml" 

# Parse command line arguments
parser = argparse.ArgumentParser(description="Build tokenizer")
parser.add_argument("--output-dir", type=str, required=True, help="Directory where to save the tokenizer")
parser.add_argument("--config", type=str, default="configs/tokenizer.yaml", help="Path to config file")
parser.add_argument("--all", action="store_true", help="Build all available tokenizers")

args = parser.parse_args()

config = load_config(args.config)

print("Loading dataset...")
dataset = load_dataset(
    "arrow", 
    data_dir=config["data"]["data_folder"],
    # FIX: Use the pattern from config, default to **/*.arrow
    data_files=config["data"].get("data_files_pattern", "**/*.arrow"),
    split="train",
)

# DEBUG: Limit to 100 rows for testing
#print("!!! DEBUG MODE: Using only first 100 rows !!!")
#dataset = dataset.select(range(100))

text_field = config["data"]["text_field"]
base_output_dir = args.output_dir

Path(base_output_dir).mkdir(parents=True, exist_ok=True)

# --- Pre-processing: Extract SMILES ---
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
    batch_size=1000,
    remove_columns=dataset.column_names,
    num_proc=4
)

print(f"\n[DEBUG] New dataset size: {len(chem_dataset)} rows.")
print("[DEBUG] First 5 extracted SMILES:")
for i in range(min(5, len(chem_dataset))):
    print(f"Row {i}: {chem_dataset[i][text_field]}")

if len(chem_dataset) == 0:
    print("\n[WARNING] !!! NO CHEMICAL DATA FOUND !!!")
    print("The regex didn't match anything. Tokenizers might be empty.")
else:
    print(f"\n[INFO] Successfully extracted {len(chem_dataset)} SMILES segments.")


# --- Determine which tokenizers to build ---
tokenizer_types_to_build = []

if args.all:
    print("[INFO] '--all' flag provided. Building ALL tokenizers.")
    tokenizer_types_to_build = list(TOKENIZER_CLASSES.keys())
else:
    tokenizer_type = config["tokenizer"]["type"] 
    # Handle case where type is 'base' or 'hybrid' but we want to build the chemical tokenizer defined in 'chem_type'
    if (tokenizer_type == "base" or tokenizer_type == "hybrid") and "chem_type" in config["tokenizer"]:
        print(f"[INFO] Tokenizer type is '{tokenizer_type}'. Switching to build chemical tokenizer defined in 'chem_type': {config['tokenizer']['chem_type']}")
        tokenizer_type = config["tokenizer"]["chem_type"]
    
    tokenizer_types_to_build = [tokenizer_type]

print(f"Tokenizers to build: {tokenizer_types_to_build}")

# --- Build Loop ---
for t_type in tokenizer_types_to_build:
    print(f"\n==================================================================")
    print(f"BUILDING TOKENIZER: {t_type}")
    print(f"==================================================================")
    
    if t_type not in TOKENIZER_CLASSES:
        print(f"[ERROR] Unknown tokenizer type: {t_type}. Skipping.")
        continue
        
    TokenizerClass = TOKENIZER_CLASSES[t_type]
    output_subdir_name = f"{t_type}_tokenizer"
    
    try:
        build_and_save_tokenizer(
            TokenizerClass=TokenizerClass,
            dataset=chem_dataset, 
            text_field=text_field, 
            output_dir=f"{base_output_dir}/{output_subdir_name}" 
        )
        print(f"SUCCESS: {t_type} tokenizer built and saved.")
    except Exception as e:
        print(f"FAILURE: Failed to build {t_type} tokenizer.")
        print(f"Error: {e}")

print("\nAll requested tokenizers processed.")