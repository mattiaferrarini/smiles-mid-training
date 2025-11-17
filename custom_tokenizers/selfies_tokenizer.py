from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re
import json
from pathlib import Path
import utils.helpers as helpers


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
        return helpers._load_vocab_from_json(path, append_to_existing_vocabulary, self.vocab)

    def reset_vocabulary(self):
        self.vocab = helpers.reset_vocabulary()

    def load_vocabulary(self, vocab_path="../json/vocab_symbol_to_number.json"):
        self.vocab = helpers.load_vocabulary(vocab_path)


    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path="../json/vocab_symbol_to_number.json", save_vocabulary=False):
        self.vocab = helpers.create_vocabulary( text, self._tokenize, append_to_existing_vocabulary, vocab_path, save_vocabulary, self.vocab)


if __name__ == "__main__":
    # small smoke test
    tk = SelfiesTokenizer()
    print("Initial vocab size:", len(tk))
    s = "CNaC(=O)Oc1ccccc1C(=O)O"
    print("Tokens:", tk._tokenize(s))
    tk.create_vocabulary(s, append_to_existing_vocabulary=False)
    print("Vocab size after create:", len(tk))
    print("Encoded:", tk(s))