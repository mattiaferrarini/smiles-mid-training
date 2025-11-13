from transformers import PreTrainedTokenizerBase

class ElementTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        super().__init__()
        self.vocab = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "+": 11} # TODO

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.vocab.get(token, 0) for token in text]} # TODO
    
    def decode(self, token_ids):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join([reverse_vocab.get(tid, '[UNK]') for tid in token_ids]) # TODO


'''
Below is a an almost working tokenizer for SELFIES and chemical formulas.
The class above should use something similar to tokenize chemical strings.
However, it also needs a vocabulary to match tokens (strings) to ids (ints).
The vocabulary should include all chemical elements, numbers, and special tokens used in SELFIES.
The list of chemical elements can be found online in csv format.
'''

if __name__ == "__main__":
    import re

    # Regex to detect groups
    SELFIES_GROUP_PATTERN = r"(\[[^\[\]]+\])"

    # Regex for chemical tokens
    CHEM_TOKEN_PATTERN = r"([A-Z][a-z]?|\d+|[=#+-]|[()·−])"

    def tokenize_selfies_and_chem(text):
        tokens = []
        
        # Split text into bracketed groups and plain parts
        parts = re.split(SELFIES_GROUP_PATTERN, text)
        
        for part in parts:
            if not part:
                continue
            
            if part.startswith('[') and part.endswith(']'):
                # Tokenize inside brackets using the chemical regex
                inner = part[1:-1]
                inner_tokens = re.findall(CHEM_TOKEN_PATTERN, inner)
                
                # Add brackets as tokens around inner parts
                tokens.append('[')
                tokens.extend(inner_tokens)
                tokens.append(']')
            else:
                # Tokenize normal text outside brackets
                tokens.extend(re.findall(CHEM_TOKEN_PATTERN, part))
        
        return tokens


    examples = [
        "[C][O][=O]",
        "[C][C][O][C]",
        "H2O",
        "NaCl",
        "Fe2(SO4)3[C][N][O]"
    ]

    for ex in examples:
        print(f"{ex}: {tokenize_selfies_and_chem(ex)}")

