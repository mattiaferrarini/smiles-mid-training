from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import utils.helpers as helpers
class CharacterTokenizer(PreTrainedTokenizer):

    CHAR_LEVEL_PATTERN = r"." 

    def __init__(self, 
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
        self.current_pattern = self.CHAR_LEVEL_PATTERN
        super().__init__(unk_token=unk_token, 
            pad_token=pad_token, 
            bos_token=bos_token, 
            eos_token=eos_token, 
            **kwargs
        )
        
        
    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.vocab)

    @property # for Huggingface compatibility
    def vocab_size(self):
        return len(self.vocab)
    
    def save_pretrained(self, save_directory, **kwargs):
        save_directory = Path(save_directory)
        
        config_dict = {
            "max_len": getattr(self, "model_max_length", 1024), 
            "unk_token": "[UNK]", 
            "model_type": "character_tokenizer" 
            # Aggiungi qui gli altri token speciali usati (pad_token, eos_token, ecc.)
            # Esempio: "pad_token": self.pad_token,
        }

        vocab_file, config_file = helpers.save_tokenizer_files(
            save_directory=save_directory,
            vocab_dict=self.vocab,
            config_dict=config_dict
        )
        
        return vocab_file, config_file
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = cls() # Crea una nuova istanza
        vocab_path = Path(pretrained_model_name_or_path) / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)

        # special tokens and configs
        return tokenizer


    def _encode_plus(
        self, 
        text, 
        text_pair=None, 
        add_special_tokens=True, # Tipicamente True
        padding_strategy="do_not_pad", 
        truncation_strategy="do_not_truncate", 
        max_length=None, 
        is_split_into_words=False, 
        **kwargs
    ):
        
        tokens = self._tokenize(text)
        
        input_ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids) 
        }

    def __call__(self, text, **kwargs):

        return self._encode_plus(text, **kwargs)
    
    def decode(self, token_ids):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join([reverse_vocab.get(tid, '[UNK]') for tid in token_ids]) 
    
    def _tokenize(self, text):
        
        tokens = re.findall(self.current_pattern, text)
        
        return tokens
    
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)
    
    def _load_vocab_from_json(self, path, append_to_existing_vocabulary=False):
        return helpers._load_vocab_from_json(path, append_to_existing_vocabulary, self.vocab)
    
    def reset_vocabulary(self):
        self.vocab = helpers.reset_vocabulary()

    def save_vocabulary(self, save_directory, filename_prefix=None):
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

    def load_vocabulary(self, vocab_path="../json/vocab_symbol_to_number.json"):
        self.vocab = helpers.load_vocabulary(vocab_path)


    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path="../json/vocab_symbol_to_number.json", save_vocabulary=False):
        self.vocab = helpers.create_vocabulary( text, self._tokenize, append_to_existing_vocabulary, vocab_path, save_vocabulary, self.vocab)
        self.decoder = {v: k for k, v in self.vocab.items()}
        return self.vocab

if __name__ == "__main__":
    print("--- Testing CharacterTokenizer ---")
    tk = CharacterTokenizer()
    print("Initial vocab:", tk.vocab)
    
    s = "CNaC(=O)Oc1ccccc1C(=O)O"
    print(f"Tokenizing string: {s}")
    
    tk.create_vocabulary(s)
    print("Vocab size after learning:", len(tk))
    
    encoded = tk(s) 
    print("Standard Encode Output (input_ids):", encoded['input_ids'])
    
    decoded = tk.decode(encoded['input_ids'])
    print("Decoded:", decoded)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"Saving tokenizer to {tmpdirname}...")
        tk.save_pretrained(tmpdirname)
        
        print("Loading tokenizer from saved files...")
        loaded_tk = CharacterTokenizer.from_pretrained(tmpdirname)
        
        print("Loaded vocab size:", len(loaded_tk))
        print("Loaded tokenizer encode check:", loaded_tk.encode(s))
        
        assert len(tk) == len(loaded_tk)
        assert tk.encode(s) == loaded_tk.encode(s)
        print("SUCCESS: Tokenizer saved and loaded# filepath: c:/Users/luca_/OneDrive/Desktop/Unpoditutto/EPFL/ML/P02/smiles-mid-training/custom_tokenizers/character_tokenizer.py correctly!")