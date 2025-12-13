import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import re

from datasets import load_dataset
from utils.config import load_config
from utils.logging import get_logger
from utils.helpers import build_and_save_tokenizer

from .character_tokenizer import CharacterTokenizer
from .element_tokenizer import ElementTokenizer
from .elementallparenthesis_tokenizer import ElementAllParenthesisTokenizer
from .elementaromatics_tokenizer import ElementAromaticsTokenizer
from .elementnoparenthesis_tokenizer import ElementNoParenthesisTokenizer
from .elementrings_tokenizer import ElementRingsTokenizer
from .smiles_bpe_tokenizer import SmilesBpeTokenizer
from .ape_tokenizer import APETokenizer
from .ape_hf_tokenizer import APEHFTokenizer
from .ape_wp_hf_tokenizer import APEWPHFTokenizer
from .chem_ape import ChemAPETokenizer
from .kmer_tokenizer import KmerTokenizer
from .ape_wordpiece import APEWordPieceTokenizer

TOKENIZER_CLASSES = {
    "character": CharacterTokenizer,
    "element": ElementTokenizer,
    "elementallparenthesis": ElementAllParenthesisTokenizer,
    "elementaromatics": ElementAromaticsTokenizer,
    "elementnoparenthesis": ElementNoParenthesisTokenizer,
    "elementrings": ElementRingsTokenizer,
    "smiles_bpe": SmilesBpeTokenizer,
    "ape": APETokenizer,
    "ape_hf": APEHFTokenizer,
    "ape_wp_hf": APEWPHFTokenizer,
    "chem_ape": ChemAPETokenizer,
    "kmer": KmerTokenizer,
    "ape_wordpiece": APEWordPieceTokenizer,
}

LOGGER = get_logger(__name__)


def build_tokenizer(output_dir, config_path):
    """
    Builds and saves a tokenizer based on the provided configuration

    Args:
        output_dir (str): Directory where to save the tokenizer
        config_path (str): Path to the configuration file
    """
    config = load_config(config_path)

    LOGGER.info("Configuration loaded")
    LOGGER.debug(f"Config: {config}")

    LOGGER.info("Loading dataset...")
    dataset = load_dataset(
        "arrow",
        data_dir=config["data"]["data_folder"],
        data_files=config["data"].get("data_files_pattern", "**/*.arrow"),
        split="train",
    )

    base_output_dir = output_dir
    text_field = config["data"]["text_field"]
    tokenizer_type = config["tokenizer"]["type"]

    # Handle case where type is 'base' or 'hybrid'
    if (
        tokenizer_type == "base" or tokenizer_type == "hybrid"
    ) and "chem_type" in config["tokenizer"]:
        LOGGER.info(
            f"Tokenizer type is '{tokenizer_type}'. Switching to build chemical tokenizer defined in 'chem_type': {config['tokenizer']['chem_type']}"
        )
        tokenizer_type = config["tokenizer"]["chem_type"]

    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    output_subdir_name = config["tokenizer"].get(
        "output_subdir_name", f"{tokenizer_type}_tokenizer"
    )
    LOGGER.info(
        f"Output directory for tokenizer: {base_output_dir}/{output_subdir_name}"
    )

    if tokenizer_type in TOKENIZER_CLASSES:
        tokenizerclass = TOKENIZER_CLASSES[tokenizer_type]
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    if tokenizerclass:
        LOGGER.info(f"Starting build for tokenizer type: {tokenizer_type}")

        LOGGER.info(
            "Filtering dataset to extract chemical segments (flattening to one SMILES per row)..."
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

        LOGGER.debug("First 3 original examples:")
        for i in range(min(3, len(dataset))):
            LOGGER.debug(f"Example {i}: {dataset[i][text_field]}...")

        safe_num_proc = min(8, os.cpu_count() or 1)
        LOGGER.info(f"Using num_proc={safe_num_proc} for processing")

        chem_dataset = dataset.map(
            extract_smiles_batch,
            batched=True,
            batch_size=10000,
            remove_columns=dataset.column_names,
            num_proc=safe_num_proc,
        )

        LOGGER.debug(f"New dataset size: {len(chem_dataset)} rows.")
        LOGGER.debug("First 5 extracted SMILES:")
        for i in range(min(5, len(chem_dataset))):
            LOGGER.debug(f"Row {i}: {chem_dataset[i][text_field]}")

        if len(chem_dataset) == 0:
            LOGGER.warning("No chemical data found")
        else:
            LOGGER.info(f"Successfully extracted {len(chem_dataset)} SMILES segments.")

        if "portion_of_data" in config["data"]:
            portion = config["data"]["portion_of_data"]
            num_rows = int(len(chem_dataset) * portion)
            chem_dataset = chem_dataset.select(range(num_rows))
            LOGGER.info(
                f"Using portion_of_data={portion}. Reduced dataset to {num_rows} rows."
            )

        LOGGER.info(
            f"Building and saving tokenizer to {base_output_dir}/{output_subdir_name} ..."
        )

        build_and_save_tokenizer(
            TokenizerClass=tokenizerclass,
            dataset=chem_dataset,
            text_field=text_field,
            output_dir=f"{base_output_dir}/{output_subdir_name}",
            config=config,
        )

        LOGGER.info("Tokenizer built and saved")
