from .character_tokenizer import CharacterTokenizer 
from .element_tokenizer import ElementTokenizer
from .elementallparenthesis_tokenizer import ElementAllParenthesisTokenizer
from .elementaromatics_tokenizer import ElementAromaticsTokenizer
from .elementnoparenthesis_tokenizer import ElementNoParenthesisTokenizer
from .elementrings_tokenizer import ElementRingsTokenizer
from .selfies_tokenizer import SelfiesTokenizer

from transformers import AutoTokenizer


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
    elif tokenizer_type == "character":
        # JUST FOR TEST, TO BE DELETED LATER
        tokenizer = CharacterTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

    return tokenizer 