from .character_tokenizer import CharacterTokenizer 
from .element_tokenizer import ElementTokenizer
from .elementallparenthesis_tokenizer import ElementAllParenthesisTokenizer
from .elementaromatics_tokenizer import ElementAromaticsTokenizer
from .elementnoparenthesis_tokenizer import ElementNoParenthesisTokenizer
from .elementrings_tokenizer import ElementRingsTokenizer
from .selfies_tokenizer import SelfiesTokenizer
from .smiles_bpe_tokenizer import SmilesBpeTokenizer
from .ape_tokenizer import APETokenizer
from .parallel_ape_tokenizer import ParallelAPETokenizer
from .hybrid_tokenizer import HybridTokenizer

from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os


def assemble_tokenizer(config):
    tokenizer = None
    tokenizer_type = config["tokenizer"]["type"]

    START_SMILES = config["tokenizer"]["special_tokens"]["start_smiles"]
    END_SMILES = config["tokenizer"]["special_tokens"]["end_smiles"]
    base_tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    if tokenizer_type == "base":
        tokenizer = base_tokenizer
    elif tokenizer_type == "base_special":
        print("Including special SMILES tokens")
        base_tokenizer.add_special_tokens({'additional_special_tokens': [START_SMILES, END_SMILES]})
        tokenizer = base_tokenizer
    elif tokenizer_type == "hybrid":
        chem_type = config["tokenizer"].get("chem_type", "element")
        print(f"Assembling Hybrid Tokenizer with chem_type: {chem_type}")
        
        # Determine path to pre-built tokenizer
        base_output_dir = config["tokenizer"]["output_dir"]
        output_subdir_name = config["tokenizer"].get("output_subdir_name", f"{chem_type}_tokenizer")
        tokenizer_dir = os.path.join(base_output_dir, output_subdir_name)
        
        chem_tokenizer = None
        
        if chem_type == "smiles_bpe":
            tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
            if os.path.exists(tokenizer_file):
                print(f"Loading BPE tokenizer from {tokenizer_file}")
                chem_tokenizer = SmilesBpeTokenizer(tokenizer_file=tokenizer_file)
            else:
                raise FileNotFoundError(f"BPE tokenizer file not found at {tokenizer_file}. Please run build_tokenizer.py first.")
        
        else:
            # Map type to class
            tokenizer_classes = {
                "character": CharacterTokenizer,
                "element": ElementTokenizer,
                "elementallparenthesis": ElementAllParenthesisTokenizer,
                "elementaromatics": ElementAromaticsTokenizer,
                "elementnoparenthesis": ElementNoParenthesisTokenizer,
                "elementrings": ElementRingsTokenizer,
                "selfies": SelfiesTokenizer,
                "ape": APETokenizer,
                "parallel_ape": ParallelAPETokenizer,
            }
            
            if chem_type not in tokenizer_classes:
                 raise ValueError(f"Unknown chem_type: {chem_type}")
            
            TokenizerClass = tokenizer_classes[chem_type]
            vocab_file = os.path.join(tokenizer_dir, "vocab.json")
            
            if os.path.exists(vocab_file):
                print(f"Loading {chem_type} tokenizer vocab from {vocab_file}")
                chem_tokenizer = TokenizerClass(vocab_file=vocab_file)
            else:
                print(f"Warning: Pre-built vocab not found at {vocab_file}. Initializing default {chem_type} tokenizer.")
                chem_tokenizer = TokenizerClass()
            
        tokenizer = HybridTokenizer(
            base_tokenizer=base_tokenizer,
            chem_tokenizer=chem_tokenizer,
            chem_start=START_SMILES,
            chem_end=END_SMILES
        )
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

    return tokenizer 