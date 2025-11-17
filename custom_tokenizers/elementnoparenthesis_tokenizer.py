from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re
import json
from pathlib import Path
import utils.helpers as helpers

class ElementNoParenthesisTokenizer(PreTrainedTokenizerBase):

    ELEMENTS = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
            "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
            "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
            "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
            "Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
            "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
            "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]

    # sort by length descending to match longer first
    ELEMENTS = sorted(ELEMENTS, key=lambda x: -len(x))
    ELEMENT_PATTERN = "|".join(ELEMENTS)

    # corrected regex
    ATOM_LEVEL_PATTERN = r"(\[|\]|" + ELEMENT_PATTERN + r"|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"

        
    #ATOM_LEVEL_PATTERN = r"(\[|\]|Br?|Cl?|[A-Z][a-z]?|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"

    def __init__(self):
        super().__init__()
        self.vocab = {'[UNK]': 0}
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
    tk = ElementNoParenthesisTokenizer()
    print("Initial vocab size:", len(tk))
    s = "CNaC(=O)Oc1ccccc1C(=O)O"
    print("Tokens:", tk._tokenize(s))
    tk.create_vocabulary(s, append_to_existing_vocabulary=False)
    print("Vocab size after create:", len(tk))
    print("Encoded:", tk(s))