import os
from utils.logging import get_logger
from transformers import AutoTokenizer

from .element_tokenizer import ElementTokenizer
from .elementallparenthesis_tokenizer import ElementAllParenthesisTokenizer
from .elementaromatics_tokenizer import ElementAromaticsTokenizer
from .elementnoparenthesis_tokenizer import ElementNoParenthesisTokenizer
from .elementrings_tokenizer import ElementRingsTokenizer
from .smiles_bpe_tokenizer import SmilesBpeTokenizer
from .smiles_wp_tokenizer import SmilesWPTokenizer
from .ape_tokenizer import APETokenizer
from .ape_hf_tokenizer import APEHFTokenizer
from .ape_wp_hf_tokenizer import APEWPHFTokenizer
from .chem_ape import ChemAPETokenizer
from .hybrid_tokenizer import HybridTokenizer
from .kmer_tokenizer import KmerTokenizer
from .ape_wordpiece import APEWordPieceTokenizer
from .character_tokenizer import CharacterTokenizer

LOGGER = get_logger(__name__)


def assemble_tokenizer(config):
    """
    Assembles a tokenizer based on the provided configuration.

    Args:
        config (dict): A dictionary containing tokenizer configuration parameters.

    Returns:
        PreTrainedTokenizerBase: The assembled tokenizer instance.
    """
    # Recupera i valori dal config
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
    chem_type = config["tokenizer"].get("chem_type", "element")
    LOGGER.info(f"Assembling Hybrid Tokenizer with chem_type: {chem_type}")

    base_output_dir = config["tokenizer"]["output_dir"]
    output_subdir_name = config["tokenizer"].get(
        "output_subdir_name", f"{chem_type}_tokenizer"
    )
    tokenizer_dir = os.path.join(base_output_dir, output_subdir_name)
    LOGGER.info(f"Tokenizer dir: {tokenizer_dir}")

    chem_tokenizer = None

    # Handle BPE and WordPiece tokenizers separately
    if chem_type in ["smiles_bpe", "ape_hf", "ape_wp_hf", "smiles_wp"]:
        tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.exists(tokenizer_file):
            LOGGER.info(f"Loading {chem_type} tokenizer from {tokenizer_file}")
            if chem_type == "smiles_bpe":
                chem_tokenizer = SmilesBpeTokenizer(tokenizer_file=tokenizer_file)
            elif chem_type == "smiles_wp":
                chem_tokenizer = SmilesWPTokenizer(tokenizer_file=tokenizer_file)
            elif chem_type == "ape_hf":
                chem_tokenizer = APEHFTokenizer.from_pretrained(tokenizer_dir)
            elif chem_type == "ape_wp_hf":
                chem_tokenizer = APEWPHFTokenizer.from_pretrained(tokenizer_dir)
        else:
            raise FileNotFoundError(
                f"Tokenizer file not found at {tokenizer_file}. Please run build_tokenizer.py first."
            )
    else:
        # Map type to class
        tokenizer_classes = {
            "character": CharacterTokenizer,
            "element": ElementTokenizer,
            "elementallparenthesis": ElementAllParenthesisTokenizer,
            "elementaromatics": ElementAromaticsTokenizer,
            "elementnoparenthesis": ElementNoParenthesisTokenizer,
            "elementrings": ElementRingsTokenizer,
            "ape": APETokenizer,
            "ape_hf": APEHFTokenizer,
            "ape_wp_hf": APEWPHFTokenizer,
            "chem_ape": ChemAPETokenizer,
            "ape_wordpiece": APEWordPieceTokenizer,
            "kmer": KmerTokenizer,
        }

        if chem_type not in tokenizer_classes:
            raise ValueError(f"Unknown chem_type: {chem_type}")

        TokenizerClass = tokenizer_classes[chem_type]
        vocab_file = os.path.join(tokenizer_dir, "vocab.json")

        if os.path.exists(vocab_file):
            LOGGER.info(f"Loading {chem_type} tokenizer vocab from {vocab_file}")
            chem_tokenizer = TokenizerClass(vocab_file=vocab_file)
        else:
            LOGGER.warning(
                f"Pre-built vocab not found at {vocab_file}. Initializing default {chem_type} tokenizer."
            )
            chem_tokenizer = TokenizerClass()
        
    return chem_tokenizer
    