import argparse
import os  # Added to check cpu count safely
from pathlib import Path
from datasets import load_dataset

# 1. Import all tokenizer classes
from character_tokenizer import CharacterTokenizer 
from element_tokenizer import ElementTokenizer
from elementallparenthesis_tokenizer import ElementAllParenthesisTokenizer
from elementaromatics_tokenizer import ElementAromaticsTokenizer
from elementnoparenthesis_tokenizer import ElementNoParenthesisTokenizer
from elementrings_tokenizer import ElementRingsTokenizer
from selfies_tokenizer import SelfiesTokenizer
from smiles_bpe_tokenizer import SmilesBpeTokenizer
from ape_tokenizer import APETokenizer
from ape_hf_tokenizer import APEHFTokenizer
from ape_wp_hf_tokenizer import APEWPHFTokenizer
from chem_ape import ChemAPETokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import build_and_save_tokenizer
from utils.config import load_config
import re

def main():
    CONFIG_PATH = "configs/tokenizer.yaml" 

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build tokenizer")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where to save the tokenizer")
    parser.add_argument("--config", type=str, default="configs/tokenizer.yaml", help="Path to config file")

    args = parser.parse_args()

    config = load_config(args.config)

    print("Configuration Loaded:")
    print(config)

    print("Loading dataset...")
    dataset = load_dataset(
        "arrow", 
        data_dir=config["data"]["data_folder"],
        data_files=config["data"].get("data_files_pattern", "**/*.arrow"),
        split="train",
    )

    text_field = config["data"]["text_field"]
    base_output_dir = args.output_dir
    tokenizer_type = config["tokenizer"]["type"] 

    # Handle case where type is 'base' or 'hybrid'
    if (tokenizer_type == "base" or tokenizer_type == "hybrid") and "chem_type" in config["tokenizer"]:
        print(f"[INFO] Tokenizer type is '{tokenizer_type}'. Switching to build chemical tokenizer defined in 'chem_type': {config['tokenizer']['chem_type']}")
        tokenizer_type = config["tokenizer"]["chem_type"]

    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    output_subdir_name = config["tokenizer"].get("output_subdir_name", f"{tokenizer_type}_tokenizer")
    print(f"Output directory for tokenizer: {base_output_dir}/{output_subdir_name}")

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
        "ape_hf": APEHFTokenizer,
        "ape_wp_hf": APEWPHFTokenizer,
        "chem_ape": ChemAPETokenizer,
    }

    if tokenizer_type in tokenizer_classes:
        tokenizerclass = tokenizer_classes[tokenizer_type]
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    if tokenizerclass:
        print(f"--- Starting build for tokenizer type: {tokenizer_type} ---")
        
        print("Filtering dataset to extract chemical segments (flattening to one SMILES per row)...")
        
        def extract_smiles_batch(batch):
            extracted_smiles = []
            pattern = r"\[START_SMILES\](.*?)\[END_SMILES\]"
            
            for text in batch[text_field]:
                if text:
                    matches = re.findall(pattern, text, re.DOTALL)
                    valid_matches = [m.strip() for m in matches if m.strip()]
                    extracted_smiles.extend(valid_matches)
            
            return {text_field: extracted_smiles}

        print("\n[DEBUG] First 3 original examples:")
        for i in range(min(3, len(dataset))):
            print(f"Example {i}: {dataset[i][text_field]}...")

        # FIX: Lower num_proc significantly to prevent OOM
        # Using a safer number like 8, or os.cpu_count() if you have enough RAM
        safe_num_proc = min(8, os.cpu_count() or 1)
        print(f"[INFO] Using num_proc={safe_num_proc} for processing")

        chem_dataset = dataset.map(
            extract_smiles_batch, 
            batched=True, 
            batch_size=10000,
            remove_columns=dataset.column_names,
            num_proc=safe_num_proc  # CHANGED FROM 64
        )

        print(f"\n[DEBUG] New dataset size: {len(chem_dataset)} rows.")
        print("[DEBUG] First 5 extracted SMILES:")
        for i in range(min(5, len(chem_dataset))):
            print(f"Row {i}: {chem_dataset[i][text_field]}")
        
        if len(chem_dataset) == 0:
            print("\n[WARNING] !!! NO CHEMICAL DATA FOUND !!!")
        else:
            print(f"\n[INFO] Successfully extracted {len(chem_dataset)} SMILES segments.")

        if "portion_of_data" in config["data"]:
            portion = config["data"]["portion_of_data"]
            num_rows = int(len(chem_dataset) * portion)
            chem_dataset = chem_dataset.select(range(num_rows))
            print(f"\n[INFO] Using portion_of_data={portion}. Reduced dataset to {num_rows} rows.")

        print(f"\nBuilding and saving tokenizer to {base_output_dir}/{output_subdir_name} ...")

        build_and_save_tokenizer(
            TokenizerClass=tokenizerclass,
            dataset=chem_dataset, 
            text_field=text_field, 
            output_dir=f"{base_output_dir}/{output_subdir_name}",
            config=config
        )

        print("\nTokenizer built and saved.")

if __name__ == "__main__":
    main()

