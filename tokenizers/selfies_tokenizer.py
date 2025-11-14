from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re

class SelfiesTokenizer(PreTrainedTokenizerBase):

    SELFIES_GROUP_PATTERN = r"(\[[^\[\]]+\])"
    CHEM_TOKEN_PATTERN = r"([A-Z][a-z]?|\d+|[=#+-]|[()·−])"

    def __init__(self):
        super().__init__()
        try:
            # TODO create the vocabulary JSON for SELFIES
            self.vocab = self._load_vocab_from_json("../json/vocab_selfies.json")
        except Exception as e:
            print("Json file not found or could not be loaded:", e)
            return -1        
        
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


if __name__ == "__main__":
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"

