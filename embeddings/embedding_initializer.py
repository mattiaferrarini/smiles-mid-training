import re
import sys
import json
import torch
from pathlib import Path

def initialize_default_embeddings(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    return model

def initialize_average_embeddings(model, tokenizer):
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
    Inizializza gli embedding chimici usando la composizione degli elementi.
    Richiede un HybridTokenizer.
    """
    # 1. FALLBACK DI SICUREZZA SE IL TOKENIZER NON È IBRIDO
    if not hasattr(hybrid_tokenizer, "base_tokenizer") or not hasattr(hybrid_tokenizer, "get_chem_vocab"):
        print(f"WARNING: Tokenizer non ibrido rilevato ({type(hybrid_tokenizer)}). Uso inizializzazione default.")
        return initialize_default_embeddings(model, hybrid_tokenizer)

    # 2. Caricamento tavola periodica (Gestione percorsi robusta)
    symbol_to_name = {}
    json_path = Path(__file__).resolve().parent.parent / 'json' / 'periodic_table.json'
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            periodic_table = json.load(f)
            for entry in periodic_table:
                sym = entry.get('Symbol', '').strip()
                name = entry.get('Element', '').strip()
                if sym and name:
                    symbol_to_name[sym] = name
    except Exception:
        # Se fallisce, useremo solo i simboli come fallback
        pass 

    # 3. Regex Elementi (Hardcoded fallback se l'import fallisce)
    elements_list = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca"]
    try:
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from custom_tokenizers.element_tokenizer import ElementTokenizer
        elements_list = ElementTokenizer.ELEMENTS
    except ImportError:
        pass
    
    pattern = "|".join(sorted(elements_list, key=lambda x: -len(x)))
    
    # 4. Logica di Inizializzazione
    base_tokenizer = hybrid_tokenizer.base_tokenizer
    chem_vocab = hybrid_tokenizer.get_chem_vocab()
    chem_ids_map = hybrid_tokenizer.get_chem_ids_map()
    
    model.resize_token_embeddings(len(hybrid_tokenizer))
    embeddings = model.get_input_embeddings().weight
    
    print(f"Initializing {len(chem_ids_map)} chemical tokens using element names...")
    
    with torch.no_grad():
        for token, chem_id in chem_vocab.items():
            if chem_id not in chem_ids_map:
                continue
            
            hybrid_id = chem_ids_map[chem_id]
            
            # Scomposizione token chimico in elementi
            found_elements = re.findall(pattern, token)
            target_embeddings = []
            
            if found_elements:
                for el in found_elements:
                    # Ottieni nome esteso (es. "Na" -> "Sodium")
                    el_name = symbol_to_name.get(el, el)
                    # Tokenizza il nome con il tokenizer LLM base
                    base_ids = base_tokenizer(el_name, add_special_tokens=False)["input_ids"]
                    if base_ids:
                        el_emb = embeddings[base_ids].mean(dim=0)
                        target_embeddings.append(el_emb)
            else:
                # Fallback per numeri o punteggiatura
                base_ids = base_tokenizer(token, add_special_tokens=False)["input_ids"]
                if base_ids:
                    token_emb = embeddings[base_ids].mean(dim=0)
                    target_embeddings.append(token_emb)
            
            if target_embeddings:
                final_embedding = torch.stack(target_embeddings).mean(dim=0)
                embeddings[hybrid_id] = final_embedding

    return model

# --- FIRMA CORRETTA: Accetta 'strategy' esplicitamente ---
def initialize_embeddings(model, tokenizer, strategy="default"):
    
    print(f"Base embeddings shape: {model.get_input_embeddings().weight.shape}")
    print(f"Initializing embeddings with strategy: {strategy}")

    if strategy == "average":
        model = initialize_average_embeddings(model, tokenizer)
    elif strategy == "elementwise":
        model = initialize_elementwise_embeddings(model, tokenizer)
    else:
        # Default, random o qualsiasi altra stringa
        model = initialize_default_embeddings(model, tokenizer)
    
    print(f"Resized embeddings shape: {model.get_input_embeddings().weight.shape}")
    
    return model