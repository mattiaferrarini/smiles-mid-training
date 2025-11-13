import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer
from custom_tokenizers.hybrid_tokenizer import HybridTokenizer
from custom_tokenizers.element_tokenizer import ElementTokenizer
from transformers import AutoModelForCausalLM
import json

def initialize_default_embeddings(model, hybrid_tokenizer):
    model.resize_token_embeddings(len(hybrid_tokenizer))
    return model

def initialize_average_embeddings(model, hybrid_tokenizer):
    base_embeddings = model.get_input_embeddings().weight

    num_base_tokens = base_embeddings.shape[0]
    num_new_tokens = len(hybrid_tokenizer.get_chem_vocab())

    mean_embedding = base_embeddings.mean(dim=0, keepdim=True) 
    model.resize_token_embeddings(len(hybrid_tokenizer))

    with torch.no_grad():
        for i in range(num_new_tokens):
            model.get_input_embeddings().weight[num_base_tokens + i] = mean_embedding
    return model

def initialize_elementwise_embeddings(model, hybrid_tokenizer):
    symbol_name_map = {}
    with open("json/periodic_table.json", "r") as f:
        periodic_table = json.load(f)
        for element in periodic_table:
            symbol_name_map[element["Symbol"]] = element["Element"]
    reverse_chem_vocab = {v: k for k, v in hybrid_tokenizer.get_chem_vocab().items()}
    print(reverse_chem_vocab)

    num_base_tokens = model.get_input_embeddings().weight.shape[0]
    num_new_tokens = len(hybrid_tokenizer.get_chem_vocab())

    model.resize_token_embeddings(len(hybrid_tokenizer))

    for id in range(num_base_tokens + 2, num_base_tokens + 2 + num_new_tokens):
        token = reverse_chem_vocab[id]
        if token in symbol_name_map:
            element_name = symbol_name_map[token]
        else:
            element_name = reverse_chem_vocab[id]

        print("Initializing embedding for token:", token, "Element name:", element_name, "ID:", id)

        ids = hybrid_tokenizer(element_name, add_special_tokens=False)["input_ids"]
        print("Mapped to base token IDs:", ids)
        embeddings = model.get_input_embeddings().weight[ids]
        avg_embedding = embeddings.mean(dim=0)
        with torch.no_grad():
            model.get_input_embeddings().weight[id] = avg_embedding

    return model


def initialize_embeddings(model, hybrid_tokenizer, strategy="default"):
    base_shape = model.get_input_embeddings().weight.shape 
    print("Base embeddings shape: ", base_shape)

    match strategy:
        case "default":
            model = initialize_default_embeddings(model, hybrid_tokenizer)
        case "average":
            model = initialize_average_embeddings(model, hybrid_tokenizer)
        case "elementwise":
            model = initialize_elementwise_embeddings(model, hybrid_tokenizer)

    new_shape = model.get_input_embeddings().weight.shape
    print("Resized embeddings shape: ", new_shape)

    assert new_shape[0] == len(hybrid_tokenizer), "Embedding layer size does not match tokenizer size."
    assert new_shape[0] == base_shape[0] + len(hybrid_tokenizer.get_chem_vocab()) + 2, "Number of embeddings does not match expected size."
    assert new_shape[1] == base_shape[1], "Embedding dimension size has changed."

if __name__ == "__main__":
    load_dotenv()

    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hybrid_tokenizer = HybridTokenizer(base_tokenizer, ElementTokenizer(), "[CHEM]", "[/CHEM]")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    print("Initializing embedding layer...")
    model = initialize_embeddings(model, hybrid_tokenizer, strategy="elementwise")
    print("Model embedding layer initialized.")