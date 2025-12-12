import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import json
import utils.helpers as helpers
from transformers import PreTrainedTokenizer

class ElementRingsTokenizer(PreTrainedTokenizer):
    """
    Element tokenizer that handles ring numbers as special tokens.
    """

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
        config=None,
        **kwargs
    ):
        """
        Initializes the ElementRingsTokenizer.

        Args:
            vocab_file (str, optional): Path to a JSON file containing the vocabulary mapping.
            unk_token (str): The unknown token. Defaults to "[UNK]".
            pad_token (str): The padding token. Defaults to "[PAD]".
            bos_token (str): The beginning of sequence token. Defaults to "[BOS]".
            eos_token (str): The end of sequence token. Defaults to "[EOS]".
            config (dict, optional): A dictionary containing tokenizer configuration parameters.
            **kwargs: Additional keyword arguments passed to `PreTrainedTokenizer`.
        """
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
    def vocab_size(self):
        """
        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vocab)

    def _tokenize(self, text):
        """
        Splits text into tokens, handling ring numbers as special tokens.

        Args:
            text (str): The input text.

        Returns:
            list: A list of tokens.
        """
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
        """
        Converts a token to its corresponding ID.

        Args:
            token (str): The token to convert.

        Returns:
            int: The token ID.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Converts an ID to its corresponding token.

        Args:
            index (int): The token ID.

        Returns:
            str: The token.
        """
        return self.decoder.get(index, self.unk_token)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Saves the vocabulary to a file.

        Args:
            save_directory (str): The directory to save the vocabulary.
            filename_prefix (str, optional): Prefix for the vocabulary file name.

        Returns:
            tuple: Path to the saved vocabulary file.
        """
        if filename_prefix:
            vocab_file = f"{filename_prefix}-vocab.json"
        else:
            vocab_file = "vocab.json"
        path = os.path.join(save_directory, vocab_file)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)
    
    def get_vocab(self):
        """
        Returns the vocabulary dictionary.

        Returns:
            dict: The vocabulary mapping tokens to IDs.
        """
        return self.vocab

    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path="../json/vocab_symbol_to_number.json", save_vocabulary=False):
        """
        Creates a vocabulary from the provided text.

        Args:
            text (str or iterator): The input text or iterator over text.
            append_to_existing_vocabulary (bool): Whether to append to an existing vocabulary. Defaults to False.
            vocab_path (str, optional): Path to load/save the vocabulary.
            save_vocabulary (bool): Whether to save the vocabulary. Defaults to False.

        Returns:
            dict: The created vocabulary.
        """
        self.vocab = helpers.create_vocabulary(text, self._tokenize, append_to_existing_vocabulary, vocab_path, save_vocabulary, self.vocab)
        self.decoder = {v: k for k, v in self.vocab.items()}
        return self.vocab

    def _load_vocab_from_json(self, path, append_to_existing_vocabulary=False):
        """
        Loads a vocabulary from a JSON file.

        Args:
            path (str): Path to the JSON vocabulary file.

        Returns:
            dict: The loaded vocabulary.
        """
        return helpers._load_vocab_from_json(path, append_to_existing_vocabulary, self.vocab)

    def reset_vocabulary(self):
        """
        Resets the vocabulary to an empty state.
        """
        self.vocab = {self.unk_token: 0, self.pad_token: 1, self.bos_token: 2, self.eos_token: 3}
        self.decoder = {v: k for k, v in self.vocab.items()}

    def load_vocabulary(self, vocab_path="../json/vocab_symbol_to_number.json"):
        """
        Loads a vocabulary from the specified JSON file.

        Args:
            vocab_path (str): Path to the JSON vocabulary file.
        """
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