from transformers import PreTrainedTokenizer
import re
import json
import os
from typing import Optional, List, Dict, Any
import utils.helpers as helpers

class ElementRingsTokenizer(PreTrainedTokenizer):

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

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self, 
        vocab_file=None, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        bos_token="[BOS]", 
        eos_token="[EOS]", 
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
        self.current_pattern = self.ATOM_LEVEL_PATTERN
        
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
    tk = ElementRingsTokenizer()
    print("Initial vocab size:", len(tk))
    s = "CNaC(=O)Oc1cc[cc]c1C(=O)O"
    print("Tokens:", tk._tokenize(s))
    tk.create_vocabulary(s, append_to_existing_vocabulary=False)
    print("Vocab size after create:", len(tk))
    print("Encoded:", tk(s))