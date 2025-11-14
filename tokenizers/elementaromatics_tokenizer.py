from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re

class ElementAromaticsTokenizer(PreTrainedTokenizerBase):

    ATOM_LEVEL_PATTERN = r"(\[[^\]]+]|Br?|Cl?|[A-Z][a-z]?|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"

    def __init__(self):
        super().__init__()
        try:
            # TODO implement a new json improved with the new tokens (just add aromatic rings)
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
        i = 0
        while i < len(all_tokens):
            t = all_tokens[i]
            if t.startswith('[') and t.endswith(']'):
                tokens.append(t)
                i += 1
                continue

            # Detect aromatic rings (lowercase atoms with matching ring numbers)
            if t.islower():
                ring_tokens = [t]
                j = i + 1
                while j < len(all_tokens):
                    if all_tokens[j].islower() or all_tokens[j].isdigit() or re.fullmatch(r'%[0-9]{2}', all_tokens[j]):
                        ring_tokens.append(all_tokens[j])
                        j += 1
                    else:
                        break
                # collapse aromatic ring into one token
                tokens.append("AROM_RING:" + "".join(ring_tokens))
                i = j
                continue

            if t.isdigit():
                tokens.append(f'RING{t}')
            elif re.fullmatch(r'%[0-9]{2}', t):
                tokens.append(f'RING{t[1:]}')
            else:
                tokens.append(t)
            i += 1

        return tokens


if __name__ == "__main__":
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"

