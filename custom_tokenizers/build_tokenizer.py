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

from build_tokenizer_utils import build_and_save_tokenizer 
from utils.config import load_config


CONFIG_PATH = "configs/tokenizer.yaml" 

config = load_config(CONFIG_PATH)


dataset = load_dataset(
    "arrow", 
    data_dir=config["data"]["data_folder"], 
    split="train"
) 
text_field = config["data"]["text_field"]
base_output_dir = config["tokenizer"]["output_dir"]
tokenizer_type = config["tokenizer"]["type"] 

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
else:
    raise ValueError(f"Tipo di tokenizer non supportato nel file di configurazione: {tokenizer_type}")


if tokenizerclass:
    build_and_save_tokenizer(
        TokenizerClass=tokenizerclass,
        dataset=dataset, 
        text_field=text_field, 
        output_dir=f"{base_output_dir}/{output_subdir_name}" 
    )

    print("\nTokenizer built and saved.")