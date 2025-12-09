from .character_tokenizer import CharacterTokenizer 
from .element_tokenizer import ElementTokenizer
from .elementallparenthesis_tokenizer import ElementAllParenthesisTokenizer
from .elementaromatics_tokenizer import ElementAromaticsTokenizer
from .elementnoparenthesis_tokenizer import ElementNoParenthesisTokenizer
from .elementrings_tokenizer import ElementRingsTokenizer
from .selfies_tokenizer import SelfiesTokenizer
from .kmer_tokenizer import KmerTokenizer
from .hybrid_tokenizer import HybridTokenizer

from transformers import AutoTokenizer

def assemble_tokenizer(config):
    tokenizer = None
    tokenizer_type = config["tokenizer"]["type"]

    tokenizer_params = config["tokenizer"].get("params", {})
    tokenizer_path = config["tokenizer"].get("path")

    START_SMILES = config["tokenizer"].get("special_tokens", {}).get("start_smiles", "[START_SMILES]")
    END_SMILES = config["tokenizer"].get("special_tokens", {}).get("end_smiles", "[END_SMILES]")
    base_tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    if tokenizer_type == "base":
        tokenizer = base_tokenizer
    elif tokenizer_type == "base_special":
        print("Including special SMILES tokens")
        base_tokenizer.add_special_tokens({'additional_special_tokens': [START_SMILES, END_SMILES]})
        tokenizer = base_tokenizer
    elif tokenizer_type == "character":
        # JUST FOR TEST, TO BE DELETED LATER
        tokenizer = CharacterTokenizer()
    elif tokenizer_type == "kmer":
        vocab_file = f"{tokenizer_path}/vocab.json"
        print(f"Loading k-mer vocab from {vocab_file}")
        chem_tokenizer = KmerTokenizer(vocab_file=vocab_file, **tokenizer_params)
        
        print("Assembling HybridTokenizer (Gemma + Kmer)...")
        tokenizer = HybridTokenizer(
            base_tokenizer=base_tokenizer,
            chem_tokenizer=chem_tokenizer,
            chem_start=START_SMILES,
            chem_end=END_SMILES
        )
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

    return tokenizer 