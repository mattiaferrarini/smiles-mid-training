from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re

class ElementAllParenthesisTokenizer(PreTrainedTokenizerBase):

    ATOM_LEVEL_PATTERN = r"(\[[^\]]+]|Br?|Cl?|[A-Z][a-z]?|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"

    def __init__(self):
        super().__init__()
        try:
            # TODO implement a new json improved with the new tokens
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
    def extract_parenthesis_groups(smiles):
        tokens = []
        i = 0
        n = len(smiles)

        while i < n:
            if smiles[i] == '(':
                start = i
                depth = 1
                i += 1
                while i < n and depth > 0:
                    if smiles[i] == '(':
                        depth += 1
                    elif smiles[i] == ')':
                        depth -= 1
                    i += 1
                tokens.append(smiles[start:i])
            else:
                tokens.append(smiles[i])
                i += 1

        return tokens

    def _tokenize(self, text):
        
        tokens = re.findall(self.current_pattern, text)

        parenthesis_groups = self.extract_parenthesis_groups(text)

        tokens = []
        for item in parenthesis_groups:
            if item.startswith("(") and item.endswith(")"):
                tokens.append(item)
            else:
                tokens.extend(re.findall(self.current_pattern, item))
        
        return tokens


if __name__ == "__main__":
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"

