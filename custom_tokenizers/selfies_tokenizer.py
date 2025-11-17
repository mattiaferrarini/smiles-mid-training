from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re
import json
from pathlib import Path

class SelfiesTokenizer(PreTrainedTokenizerBase):

    SELFIES_GROUP_PATTERN = r"(\[[^\[\]]+\])"
    CHEM_TOKEN_PATTERN = r"([A-Z][a-z]?|\d+|[=#+-]|[()·−])"

    def __init__(self):
        super().__init__()
        self.vocab = {'[UNK]': 0}   
        
    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, **kwargs):
        tokens = self._tokenize(text)
        return {"input_ids": [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]}
    
    def decode(self, token_ids):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join([reverse_vocab.get(tid, '[UNK]') for tid in token_ids]) 
    
    def _tokenize_selfies_style(self, text):
        tokens = []
        parts = re.split(self.SELFIES_GROUP_PATTERN, text)
        
        for part in parts:
            if not part:
                continue
            
            if part.startswith('[') and part.endswith(']'):
                inner = part[1:-1]
                inner_tokens = re.findall(self.CHEM_TOKEN_PATTERN, inner)
                
                tokens.append('[')
                tokens.extend(inner_tokens)
                tokens.append(']')
            else:
                tokens.extend(re.findall(self.CHEM_TOKEN_PATTERN, part))
        
        return tokens
    

    def _tokenize(self, text):
        
        tokens = self._tokenize_selfies_style(text)
        
        return tokens


    def _load_vocab_from_json(self, path, append_to_existing_vocabulary=False):
        p = Path(path)
        if not p.exists():
            print("Vocabulary file does not exist at path:", path)
            return {}
        with p.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        # normalize types
        vocab = {str(k): int(v) for k, v in raw.items()}
        if append_to_existing_vocabulary:
            for token in self.vocab.keys():
                if token not in vocab:
                    vocab[token] = self.vocab[token]
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
    
    def reset_vocabulary(self):
        self.vocab = {'[UNK]': 0}

    def load_vocabulary(self, vocab_path="../json/vocab_symbol_to_number.json"):
        try:
            self.vocab = self._load_vocab_from_json(vocab_path)
            return self.vocab
        except Exception as e:
            print("Json file not found or could not be loaded:", e)
            return {}

    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path="../json/vocab_symbol_to_number.json", save_vocabulary=False):
        if append_to_existing_vocabulary:
            try:
                vocab_file = self._load_vocab_from_json(vocab_path)
                for token in vocab_file.keys():
                    if token not in self.vocab:
                        self.vocab[token] = vocab_file[token]
            except Exception as e:
                print("Json file not found or could not be loaded:", e)
        
        tokens = self._tokenize(text)
        max_id = max(self.vocab.values()) if self.vocab else -1
        next_id = max_id + 1        
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
        # Save updated vocabulary to JSON
        if save_vocabulary:
            with open(vocab_path, 'w', encoding='utf-8') as fh:
                import json
                json.dump(self.vocab, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # small smoke test
    tk = SelfiesTokenizer()
    print("Initial vocab size:", len(tk))
    s = "CNaC(=O)Oc1ccccc1C(=O)O"
    print("Tokens:", tk._tokenize(s))
    tk.create_vocabulary(s, append_to_existing_vocabulary=False)
    print("Vocab size after create:", len(tk))
    print("Encoded:", tk(s))