from transformers import PreTrainedTokenizerBase
import re

class ElementTokenizer(PreTrainedTokenizerBase):

    CHAR_LEVEL_PATTERN = r"." 
    ATOM_LEVEL_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])" 
    SELFIES_GROUP_PATTERN = r"(\[[^\[\]]+\])"
    CHEM_TOKEN_PATTERN = r"([A-Z][a-z]?|\d+|[=#+-]|[()·−])"

    def __init__(self, tokenization_mode="atom_level", use_bpe=False, bpe_merges=None):
        super().__init__()
        self.use_bpe = use_bpe
        self.bpe_merges = bpe_merges
        self.tokenization_mode = tokenization_mode
        # TEMPORARY. TODO: Load from file where it adds to these vocab the symbol extracted from a list of smiles using the method provided
        self.vocab = {
            "[UNK]": 0,
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
            "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
            "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
            "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
            "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
            "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
            "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
            "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
            "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
            "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
            "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
            "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
            "[": 119, 
            "]": 120,
            "(": 121, 
            ")": 122,
            ".": 123, 
            "=": 124, 
            "#": 125,
            "-": 126,  
            "+": 127,
            "\\": 128,
            "/": 129,
            "@": 130, 
            "1": 131,
            "2": 132,
            "3": 133,
            "4": 134,
            "5": 135,
            "6": 136,
            "7": 137,
            "8": 138,
            "9": 139
        }
        #self.vocab = {"[UNK]": 0, "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10} 
        
        if self.tokenization_mode == "char_level":
            self.current_pattern = self.CHAR_LEVEL_PATTERN
        elif self.tokenization_mode == "atom_level":
            self.current_pattern = self.ATOM_LEVEL_PATTERN
        elif self.tokenization_mode == "selfies_level":
            # It uses tokenize_selfies_style
            pass 
        else:
            raise ValueError(f"Modalità '{tokenization_mode}' non supportata.")

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
    
    def _apply_bpe_merges(self, tokens):
        
        # It creates a dictionary where each key is a tuple representing a pair of tokens to be merged,
        merges_rank = {tuple(m.split()): i for i, m in enumerate(self.bpe_merges)}
        
        # Merging tokens based on the highest priority defined in merges_rank
        while True:
            best_pair = None
            best_pair_index = -1
            best_rank = float('inf')
            
            # It looks for the best pair to merge in the current token list
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i+1])
                
                # Check if the pair is in the merges dictionary and has a better priority
                if pair in merges_rank:
                    current_rank = merges_rank[pair]
                    
                    if current_rank < best_rank:
                        # Found a merge with higher priority (lower rank)
                        best_rank = current_rank
                        best_pair = pair
                        best_pair_index = i
                
                i += 1
            
            if best_pair is None:
                break
            
            # Merge
            
            new_tokens = []
            new_token = "".join(best_pair)
            
            j = 0
            while j < len(tokens):
                if j == best_pair_index:
                    new_tokens.append(new_token)
                    j += 2
                elif j == best_pair_index + 1:
                    j += 1
                else:
                    # Add the token normally
                    new_tokens.append(tokens[j])
                    j += 1
        
            tokens = new_tokens

        return tokens

    def _tokenize(self, text):
        
        if self.tokenization_mode == "selfies_level":
            tokens = self._tokenize_selfies_style(text)
        else:
            tokens = re.findall(self.current_pattern, text)

        if self.use_bpe and self.bpe_merges:
             if self.bpe_merges is None:
                return tokens
             
             return self._apply_bpe_merges(tokens)
        
        return tokens


if __name__ == "__main__":
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    bpe_merges_list = ["C C", "C =", "c 1"]

    # --- Test 1: Atom-Level (Base SMILES) - NO BPE ---
    atom_tokenizer_base = ElementTokenizer(tokenization_mode="atom_level", use_bpe=False)
    print("\n--- 1. Base: Atom-Level (SMILES optimized) ---")
    print(f"Tokens: {atom_tokenizer_base._tokenize(test_smiles)}")
    
    # --- Test 2: Atom-Level (Base SMILES) - CON BPE ---
    atom_tokenizer_bpe = ElementTokenizer(tokenization_mode="atom_level", use_bpe=True, bpe_merges=bpe_merges_list)
    print("\n--- 2. Experiment: Atom-Level + BPE ---")
    print(f"Tokens base (prima della fusione): {atom_tokenizer_bpe._tokenize(test_smiles)}")
    print("NB: Si applica BPE alla base Atom-Level.")

    # --- Test 3: Character-Level (Base Rigida) - CON BPE ---
    char_tokenizer_bpe = ElementTokenizer(tokenization_mode="char_level", use_bpe=True, bpe_merges=bpe_merges_list)
    print("\n--- 3. Experiment: Character-Level + BPE ---")
    print(f"Tokens base (prima della fusione): {char_tokenizer_bpe._tokenize(test_smiles)}")
    print("NB: Si applica BPE alla base Character-Level.")
