from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re

class ElementRingsTokenizer(PreTrainedTokenizerBase):

    ATOM_LEVEL_PATTERN = r"(\[|\]|Br?|Cl?|[A-Z][a-z]?|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"

    def __init__(self):
        super().__init__()
        try:
            # TODO implement a new json improved with the new tokens (just add rings)
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
        all_tokens = re.findall(self.current_pattern, text)
        tokens = []
        in_brackets = False

        for t in all_tokens:
            if t == '[':
                in_brackets = True
                tokens.append(t)
            elif t == ']':
                in_brackets = False
                tokens.append(t)
            elif not in_brackets:
                # outside brackets
                if re.fullmatch(r'%[0-9]{2}', t):
                    # multi-digit ring closure
                    tokens.append(f'RING{t[1:]}')
                elif t.isdigit():
                    tokens.append(f'RING{t}')
                else:
                    tokens.append(t)
            else:
                # inside brackets
                tokens.append(t)
        return tokens


if __name__ == "__main__":
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"

