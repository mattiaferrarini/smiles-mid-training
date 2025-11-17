from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer
import re
import json
from pathlib import Path

class ElementAllParenthesisTokenizer(PreTrainedTokenizerBase):

    ELEMENTS = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
            "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
            "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
            "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
            "Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
            "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
            "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]
    
    ELEMENTS = sorted(ELEMENTS, key=lambda x: -len(x))  # longest first

    ELEMENT_PATTERN = "|".join(ELEMENTS)  # NO parentheses
    ATOM_LEVEL_PATTERN = r"\[[^\]]+]|" + ELEMENT_PATTERN + r"|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9]"
    #ATOM_LEVEL_PATTERN = r"(\[[^\]]+]|Br?|Cl?|[A-Z][a-z]?|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"

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

    def _load_vocab_from_json(self, path, append_to_existing_vocabulary=False):
        p = Path(path)
        if not p.exists():
            print("Vocabulary file does not exist at path:", path)
            return {}
        with p.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        # normalize types
        vocab = {str(k): int(v) for k, v in raw.items()}
        if append_to_existing_vocabulary:
            for token in self.vocab.keys():
                if token not in vocab:
                    vocab[token] = self.vocab[token]
        else:
            # Ensure "[UNK]": 0 exists; if 0 is taken shift existing ids +1
            if "[UNK]" in vocab:
                if vocab["[UNK]"] != 0:
                    # shift all ids by +1 and set [UNK]=0
                    vocab = {k: (v + 1) if k != "[UNK]" else 0 for k, v in vocab.items()}
                    vocab["[UNK]"] = 0
            else:
                if 0 in vocab.values():
                    vocab = {k: (v + 1) for k, v in vocab.items()}
                vocab["[UNK]"] = 0
            return vocab
    
    def reset_vocabulary(self):
        self.vocab = {'[UNK]': 0}

    def load_vocabulary(self, vocab_path="../json/vocab_symbol_to_number.json"):
        try:
            self.vocab = self._load_vocab_from_json(vocab_path)
            return self.vocab
        except Exception as e:
            print("Json file not found or could not be loaded:", e)
            return {}

    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path="../json/vocab_symbol_to_number.json", save_vocabulary=False):
        if append_to_existing_vocabulary:
            try:
                vocab_file = self._load_vocab_from_json(vocab_path)
                for token in vocab_file.keys():
                    if token not in self.vocab:
                        self.vocab[token] = vocab_file[token]
            except Exception as e:
                print("Json file not found or could not be loaded:", e)
        
        tokens = self._tokenize(text)
        max_id = max(self.vocab.values()) if self.vocab else -1
        next_id = max_id + 1        
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
        # Save updated vocabulary to JSON
        if save_vocabulary:
            with open(vocab_path, 'w', encoding='utf-8') as fh:
                import json
                json.dump(self.vocab, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # small smoke test
    tk = ElementAllParenthesisTokenizer()
    print("Initial vocab size:", len(tk))
    s = "CNaC(=O)Oc1ccccc1C(=O)O"
    print("Tokens:", tk._tokenize(s))
    tk.create_vocabulary(s, append_to_existing_vocabulary=False)
    print("Vocab size after create:", len(tk))
    print("Encoded:", tk(s))