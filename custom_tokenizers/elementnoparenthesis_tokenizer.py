from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re

class ElementNoParenthesisTokenizer(PreTrainedTokenizerBase):

    ATOM_LEVEL_PATTERN = r"(\[|\]|Br?|Cl?|[A-Z][a-z]?|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"

    def __init__(self):
        super().__init__()
        try:
            self.vocab = self._load_vocab_from_json("../json/vocab_symbol_to_number.json")
        except Exception as e:
            print("Json file not found or could not be loaded:", e)
            return -1    
        self.current_pattern = self.ATOM_LEVEL_PATTERN    
        
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

    def _tokenize(self, text):
        
        tokens = re.findall(self.current_pattern, text)
        
        return tokens


if __name__ == "__main__":
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"

