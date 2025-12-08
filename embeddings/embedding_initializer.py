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
    import re
    import json
    from custom_tokenizers.element_tokenizer import ElementTokenizer
    
    # Load element symbol to name mapping
    symbol_to_name = {}
    json_path = os.path.join(os.path.dirname(__file__), '..', 'json', 'periodic_table.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            periodic_table = json.load(f)
            for entry in periodic_table:
                sym = entry.get('Symbol', '').strip()
                name = entry.get('Element', '').strip()
                if sym and name:
                    symbol_to_name[sym] = name
    except FileNotFoundError:
        print(f"Warning: Periodic Table JSON not found at {json_path}. Fallback to symbol tokenization.")
    except Exception as e:
        print(f"Warning: Error loading Periodic Table JSON: {e}")
    
    # Use the ELEMENTS list from ElementTokenizer for regex matching
    elements_list = ElementTokenizer.ELEMENTS
    # Ensure sorted by length to match longest first
    elements_list = sorted(elements_list, key=lambda x: -len(x))
    pattern = "|".join(elements_list)
    
    base_tokenizer = hybrid_tokenizer.base_tokenizer
    chem_vocab = hybrid_tokenizer.get_chem_vocab()
    chem_ids_map = hybrid_tokenizer.get_chem_ids_map()
    
    # Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(hybrid_tokenizer))
    embeddings = model.get_input_embeddings().weight
    
    print(f"Initializing {len(chem_ids_map)} chemical tokens using element names...")
    
    with torch.no_grad():
        for token, chem_id in chem_vocab.items():
            if chem_id not in chem_ids_map:
                continue
            
            hybrid_id = chem_ids_map[chem_id]
            
            # Find all elements contained in the token
            found_elements = re.findall(pattern, token)
            
            target_embeddings = []
            
            if found_elements:
                for el in found_elements:
                    # Get the full name of the element, fallback to symbol if not found
                    el_name = symbol_to_name.get(el, el)
                    
                    # Tokenize the element name with the base tokenizer
                    base_ids = base_tokenizer(el_name, add_special_tokens=False)["input_ids"]
                    if base_ids:
                        # Average the embeddings of the tokens making up this element name
                        el_emb = embeddings[base_ids].mean(dim=0)
                        target_embeddings.append(el_emb)
            else:
                # Fallback: No elements found (e.g. punctuation, numbers, bonds)
                # Tokenize the raw token string with base tokenizer
                base_ids = base_tokenizer(token, add_special_tokens=False)["input_ids"]
                if base_ids:
                    token_emb = embeddings[base_ids].mean(dim=0)
                    target_embeddings.append(token_emb)
            
            if target_embeddings:
                # Average all component embeddings
                final_embedding = torch.stack(target_embeddings).mean(dim=0)
                embeddings[hybrid_id] = final_embedding
            else:
                print(f"Warning: Could not initialize embedding for token '{token}' (no base tokens found)")

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
    
    return model

if __name__ == "__main__":
    load_dotenv()

    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hybrid_tokenizer = HybridTokenizer(base_tokenizer, ElementTokenizer(), "[CHEM]", "[/CHEM]")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    print("Initializing embedding layer...")
    model = initialize_embeddings(model, hybrid_tokenizer, strategy="elementwise")
    print("Model embedding layer initialized.")