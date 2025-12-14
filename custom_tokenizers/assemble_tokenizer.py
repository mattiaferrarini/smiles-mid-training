import os
from utils.logging import get_logger
from transformers import AutoTokenizer

from .hybrid_tokenizer import HybridTokenizer
from .registry import TOKENIZER_CLASSES

LOGGER = get_logger(__name__)


def assemble_tokenizer(config):
    """
    Assembles a tokenizer based on the provided configuration.

    Args:
        config (dict): A dictionary containing tokenizer configuration parameters.

    Returns:
        PreTrainedTokenizerBase: The assembled tokenizer instance.
    """
    # Get tokenizer type and special tokens
    tokenizer_type = config["tokenizer"]["type"]
    special_tokens = config["tokenizer"].get("special_tokens", {})
    START_SMILES = special_tokens.get("start_smiles", "[START_SMILES]")
    END_SMILES = special_tokens.get("end_smiles", "[END_SMILES]")

    LOGGER.info(f"Assembling tokenizer of type: {tokenizer_type}")

    if tokenizer_type == "base":
        base_tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        return base_tokenizer
    elif tokenizer_type == "base_special":
        LOGGER.info("Including special SMILES tokens")
        base_tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        base_tokenizer.add_special_tokens(
            {"additional_special_tokens": [START_SMILES, END_SMILES]}
        )
        return base_tokenizer
    elif tokenizer_type == "chem_only":
        chem_tokenizer = assemble_chem_tokenizer(config)
        return chem_tokenizer
    elif tokenizer_type == "hybrid":
        chem_tokenizer = assemble_chem_tokenizer(config)
        base_tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

        # Assemble HybridTokenizer
        hybrid_tokenizer = HybridTokenizer(
            base_tokenizer=base_tokenizer,
            chem_tokenizer=chem_tokenizer,
            chem_start=START_SMILES,
            chem_end=END_SMILES,
        )
        return hybrid_tokenizer
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")


def assemble_chem_tokenizer(config):
    """
    Assembles a chemical tokenizer based on the provided configuration.
    Args:
        config (dict): A dictionary containing tokenizer configuration parameters.
    Returns:
        PreTrainedTokenizerBase: The assembled chemical tokenizer instance.
    """

    chem_type = config["tokenizer"].get("chem_type", "element")
    LOGGER.info(f"Assembling Hybrid Tokenizer with chem_type: {chem_type}")

    base_output_dir = config["tokenizer"]["output_dir"]
    output_subdir_name = config["tokenizer"].get(
        "output_subdir_name", f"{chem_type}_tokenizer"
    )
    tokenizer_dir = os.path.join(base_output_dir, output_subdir_name)
    LOGGER.info(f"Tokenizer dir: {tokenizer_dir}")

    if chem_type not in TOKENIZER_CLASSES:
        raise ValueError(f"Unknown chem_type: {chem_type}")
    else:
        tokenizer_class = TOKENIZER_CLASSES[chem_type]
        LOGGER.info(f"Using tokenizer class: {tokenizer_class.__name__}")
        chem_tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
        return chem_tokenizer
    