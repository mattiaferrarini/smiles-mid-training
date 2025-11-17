import json
from pathlib import Path

def _load_vocab_from_json(path, append_to_existing_vocabulary=False, self_vocab=None):
    p = Path(path)
    if not p.exists():
        print("Vocabulary file does not exist at path:", path)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # normalize types
    vocab = {str(k): int(v) for k, v in raw.items()}
    if append_to_existing_vocabulary:
        for token in self_vocab.keys():
            if token not in vocab:
                vocab[token] = self_vocab[token]
        return vocab
    else:
        # Ensure "[UNK]": 0 exists; if 0 is taken shift existing ids +1
        if "[UNK]" in vocab:
            if vocab["[UNK]"] != 0:
                # shift all ids by +1 and set [UNK]=0
                vocab = {k: (v + 1) if k != "[UNK]" else 0 for k, v in vocab.items()}
                vocab["[UNK]"] = 0
        else:
            if 0 in vocab.values():
                vocab = {k: (v + 1) for k, v in vocab.items()}
            vocab["[UNK]"] = 0
        return vocab
    
def reset_vocabulary():
    return {'[UNK]': 0}

def load_vocabulary(vocab_path="../json/vocab_symbol_to_number.json"):
    try:
        vocab = _load_vocab_from_json(vocab_path)
        return vocab
    except Exception as e:
        print("Json file not found or could not be loaded:", e)
        return reset_vocabulary()
    
def create_vocabulary( text, _tokenize, append_to_existing_vocabulary=False, vocab_path="../json/vocab_symbol_to_number.json", save_vocabulary=False, vocab = {'[UNK]': 0}):
    if append_to_existing_vocabulary:
        try:
            vocab_file = _load_vocab_from_json(vocab_path)
            for token in vocab_file.keys():
                if token not in vocab:
                    vocab[token] = vocab_file[token]
        except Exception as e:
            print("Json file not found or could not be loaded:", e)
    
    tokens = _tokenize(text)
    max_id = max(vocab.values()) if vocab else -1
    next_id = max_id + 1        
    for token in tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
    # Save updated vocabulary to JSON
    if save_vocabulary:
        with open(vocab_path, 'w', encoding='utf-8') as fh:
            import json
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
    return vocab


    
