from transformers import PreTrainedTokenizerBase
import re

class ElementTokenizer(PreTrainedTokenizerBase):

    CHAR_LEVEL_PATTERN = r"." #Character-Level
    ATOM_LEVEL_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])" #Chemical unit encoding
    SELFIES_GROUP_PATTERN = r"(\[[^\[\]]+\])"
    CHEM_TOKEN_PATTERN = r"([A-Z][a-z]?|\d+|[=#+-]|[()·−])"

    def __init__(self, tokenization_mode="atom_level", bpe_merges=None):
        super().__init__()
        self.bpe_merges = bpe_merges
        self.tokenization_mode = tokenization_mode
        self.vocab = {"[UNK]": 0, "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10} # TODO
        if self.tokenization_mode == "char_level":
            self.current_pattern = self.CHAR_LEVEL_PATTERN
        elif self.tokenization_mode == "atom_level":
            self.current_pattern = self.ATOM_LEVEL_PATTERN
        elif self.tokenization_mode == "bpe_level":
            # BPE/APE parte da ATOM_LEVEL per la pre-tokenizzazione
            self.current_pattern = self.ATOM_LEVEL_PATTERN
        elif self.tokenization_mode == "selfies_level":
            pass # La logica è nel _tokenize_selfies_style
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
        return ' '.join([reverse_vocab.get(tid, '[UNK]') for tid in token_ids]) # TODO
    
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
        if self.tokenization_mode == "bpe_level" and self.bpe_merges:
            base_tokens = re.findall(self.current_pattern, text)
            # Qui si applicherebbe la logica BPE/APE usando self.bpe_merges
            # Per l'esempio, ritorna i token atomici non fusi
            return base_tokens 
            
        elif self.tokenization_mode == "selfies_level":
            return self._tokenize_selfies_style(text)
        
        else:
            # char_level o atom_level (Chemical Unit Encoding)
            return re.findall(self.current_pattern, text)


'''
Below is a an almost working tokenizer for SELFIES and chemical formulas.
The class above should use something similar to tokenize chemical strings.
However, it also needs a vocabulary to match tokens (strings) to ids (ints).
The vocabulary should include all chemical elements, numbers, and special tokens used in SELFIES.
The list of chemical elements can be found online in csv format.
'''

if __name__ == "__main__":
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    test_selfies = "[C][O][=O]"

    # --- Test Metodo 1: Character-Level ---
    char_tokenizer = ElementTokenizer(tokenization_mode="char_level")
    print("\n--- 1. Character-Level ---")
    tokens_char = char_tokenizer._tokenize(test_smiles)
    print(f"SMILES: {test_smiles}\nTokens: {tokens_char}")
    tokens_selfies_char = char_tokenizer._tokenize(test_selfies)
    print(f"SELFIES: {test_selfies}\nTokens: {tokens_selfies_char}")
    
    # --- Test Metodo 2: Atom-Level (Chemical Unit Encoding) ---
    atom_tokenizer = ElementTokenizer(tokenization_mode="atom_level")
    print("\n--- 2. Atom-Level (SMILES optimized) ---")
    tokens_atom = atom_tokenizer._tokenize(test_smiles)
    print(f"SMILES: {test_smiles}\nTokens: {tokens_atom}")
    
    # Esempio di token ID generati
    ids_atom = atom_tokenizer(test_smiles)["input_ids"]
    print(f"Input IDs: {ids_atom}")
    print(f"Decoded: {atom_tokenizer.decode(ids_atom)}")


    # --- Test Metodo 3: SELFIES-Level (Nuova logica) ---
    selfies_tokenizer = ElementTokenizer(tokenization_mode="selfies_level")
    print("\n--- 3. SELFIES-Level ---")
    tokens_selfies = selfies_tokenizer._tokenize(test_selfies)
    print(f"SELFIES: {test_selfies}\nTokens: {tokens_selfies}")
    # Come si comporta con le SMILES
    tokens_smiles_selfies = selfies_tokenizer._tokenize(test_smiles)
    print(f"SMILES: {test_smiles}\nTokens: {tokens_smiles_selfies}")
    
    # --- Test Metodo 4: BPE-Level (simulato) ---
    print("\n--- 4. BPE-Level (Simulated) ---")
    bpe_tokenizer = ElementTokenizer(tokenization_mode="bpe_level", bpe_merges=["C C", "C =", "c 1"])
    tokens_bpe = bpe_tokenizer._tokenize(test_smiles)
    print(f"SMILES: {test_smiles}\nBase Tokens (non fusi): {tokens_bpe}")
    print("NB: La logica di fusione BPE deve ancora essere implementata.")


