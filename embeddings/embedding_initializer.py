import re
import sys
import json
import torch
from pathlib import Path
from utils.logging import get_logger

LOGGER = get_logger(__name__)


def initialize_default_embeddings(model, tokenizer):
    """
    Resizes the given model's token embeddings to match the given tokenizer's vocabulary size
    For new tokens, it uses the default random initialization provided by the model configuration

    Args:
        model (torch.nn.Module): The transformer model instance
        tokenizer (PreTrainedTokenizer): The tokenizer containing the full vocabulary

    Returns:
        torch.nn.Module: The model with resized and initialized embeddings
    """
    model.resize_token_embeddings(len(tokenizer))
    return model


def initialize_average_embeddings(model, tokenizer):
    """
    Resizes the given model's token embeddings to match the given tokenizer's vocabulary size
    For new tokens, it initializes their embeddings to the average of the existing embeddings

    Args:
        model (torch.nn.Module): The transformer model instance
        tokenizer (PreTrainedTokenizer): The tokenizer containing the full vocabulary

    Returns:
        torch.nn.Module: The model with resized and initialized embeddings
    """
    base_embeddings = model.get_input_embeddings().weight
    num_base_tokens = base_embeddings.shape[0]
    num_new_tokens = len(tokenizer) - num_base_tokens

    if num_new_tokens <= 0:
        return model

    mean_embedding = base_embeddings.mean(dim=0, keepdim=True)
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        for i in range(num_new_tokens):
            model.get_input_embeddings().weight[num_base_tokens + i] = mean_embedding
    return model


def initialize_elementwise_embeddings(model, hybrid_tokenizer):
    """
    Resizes the given model's token embeddings to match the given hybrid tokenizer's vocabulary size
    For new chemical tokens, it initializes their embeddings based on the embeddings of their constituent elements

    It decomposes chemical tokens (e.g. "NaCl) into elements (e.g. "Sodium", "Chlorine"),
    fetches the embeddings of the full elements from the base model, averages them, and assign the result

    Args
        model (torch.nn.Module): The transformer model instance
        hybrid_tokenizer (HybridTokenizer): The hybrid tokenizer containing both base and chemical vocabularies

    Returns:
        torch.nn.Module: The model with resized and initialized embeddings
    """
    if not hasattr(hybrid_tokenizer, "base_tokenizer") or not hasattr(
        hybrid_tokenizer, "get_chem_vocab"
    ):
        LOGGER.warning(
            f"Found non-hybrid tokenizer ({type(hybrid_tokenizer)}), using default initialization..."
        )
        return initialize_default_embeddings(model, hybrid_tokenizer)

    symbol_to_name = {}
    json_path = Path(__file__).resolve().parent.parent / "json" / "periodic_table.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            periodic_table = json.load(f)
            for entry in periodic_table:
                sym = entry.get("Symbol", "").strip()
                name = entry.get("Element", "").strip()
                if sym and name:
                    symbol_to_name[sym] = name
    except Exception:
        pass

    elements_list = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
    ]
    try:
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from custom_tokenizers.element_tokenizer import ElementTokenizer

        elements_list = ElementTokenizer.ELEMENTS
    except ImportError:
        pass

    pattern = "|".join(sorted(elements_list, key=lambda x: -len(x)))

    base_tokenizer = hybrid_tokenizer.base_tokenizer
    chem_vocab = hybrid_tokenizer.get_chem_vocab()
    chem_ids_map = hybrid_tokenizer.get_chem_ids_map()

    model.resize_token_embeddings(len(hybrid_tokenizer))
    embeddings = model.get_input_embeddings().weight

    LOGGER.info(
        f"Initializing {len(chem_ids_map)} chemical tokens using element names..."
    )

    with torch.no_grad():
        for token, chem_id in chem_vocab.items():
            if chem_id not in chem_ids_map:
                continue

            hybrid_id = chem_ids_map[chem_id]

            found_elements = re.findall(pattern, token)
            target_embeddings = []

            if found_elements:
                for el in found_elements:
                    el_name = symbol_to_name.get(el, el)
                    base_ids = base_tokenizer(el_name, add_special_tokens=False)[
                        "input_ids"
                    ]
                    if base_ids:
                        el_emb = embeddings[base_ids].mean(dim=0)
                        target_embeddings.append(el_emb)
            else:
                base_ids = base_tokenizer(token, add_special_tokens=False)["input_ids"]
                if base_ids:
                    token_emb = embeddings[base_ids].mean(dim=0)
                    target_embeddings.append(token_emb)

            if target_embeddings:
                final_embedding = torch.stack(target_embeddings).mean(dim=0)
                embeddings[hybrid_id] = final_embedding

    return model


def initialize_embeddings(model, tokenizer, strategy="default"):
    """
    Initializes the model's token embeddings based on the given strategy

    Args:
        model (torch.nn.Module): The transformer model instance
        tokenizer (PreTrainedTokenizer): The tokenizer containing the full vocabulary
        strategy (str): The initialization strategy to use ("default", "average", "elementwise")

    Returns:
        torch.nn.Module: The model with resized and initialized embeddings
    """

    LOGGER.info(f"Base embeddings shape: {model.get_input_embeddings().weight.shape}")
    LOGGER.info(f"Initializing embeddings with strategy: {strategy}")

    if strategy == "average":
        model = initialize_average_embeddings(model, tokenizer)
    elif strategy == "elementwise":
        model = initialize_elementwise_embeddings(model, tokenizer)
    else:
        model = initialize_default_embeddings(model, tokenizer)

    LOGGER.info(
        f"Resized embeddings shape: {model.get_input_embeddings().weight.shape}"
    )

    return model
