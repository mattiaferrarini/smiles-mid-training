from transformers import PreTrainedTokenizer
import re
import json
import os
from typing import Optional, List, Dict, Any
import utils.helpers as helpers

class SelfiesTokenizer(PreTrainedTokenizer):

    SELFIES_GROUP_PATTERN = r"(\[[^\[\]]+\])"
    CHEM_TOKEN_PATTERN = r"([A-Z][a-z]?|\d+|[=#+-]|[()·−])"

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self, 
        vocab_file=None, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        bos_token="[BOS]", 
        eos_token="[EOS]", 
        config=None,
        **kwargs
    ):
        self.vocab = {}
        self.decoder = {}
        
        if vocab_file:
            with open(vocab_file, encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {unk_token: 0, pad_token: 1, bos_token: 2, eos_token: 3}
            
        self.decoder = {v: k for k, v in self.vocab.items()}
        
        super().__init__(
            unk_token=unk_token, 
            pad_token=pad_token, 
            bos_token=bos_token, 
            eos_token=eos_token, 
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

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

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        if filename_prefix:
            vocab_file = f"{filename_prefix}-vocab.json"
        else:
            vocab_file = "vocab.json"
        path = os.path.join(save_directory, vocab_file)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)
    
    def get_vocab(self):
        return self.vocab

    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path="../json/vocab_symbol_to_number.json", save_vocabulary=False):
        self.vocab = helpers.create_vocabulary(text, self._tokenize, append_to_existing_vocabulary, vocab_path, save_vocabulary, self.vocab)
        self.decoder = {v: k for k, v in self.vocab.items()}
        return self.vocab

    def _load_vocab_from_json(self, path, append_to_existing_vocabulary=False):
        return helpers._load_vocab_from_json(path, append_to_existing_vocabulary, self.vocab)

    def reset_vocabulary(self):
        self.vocab = {self.unk_token: 0, self.pad_token: 1, self.bos_token: 2, self.eos_token: 3}
        self.decoder = {v: k for k, v in self.vocab.items()}

    def load_vocabulary(self, vocab_path="../json/vocab_symbol_to_number.json"):
        self.vocab = helpers.load_vocabulary(vocab_path)
        self.decoder = {v: k for k, v in self.vocab.items()}


if __name__ == "__main__":
    # small smoke test
    tk = SelfiesTokenizer()
    print("Initial vocab size:", len(tk))
    s = "CNaC(=O)Oc1ccccc1C(=O)O"
    print("Tokens:", tk._tokenize(s))
    tk.create_vocabulary(s, append_to_existing_vocabulary=False)
    print("Vocab size after create:", len(tk))
    print("Encoded:", tk(s))