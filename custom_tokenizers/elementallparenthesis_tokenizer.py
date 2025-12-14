import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import json
import utils.helpers as helpers
from transformers import PreTrainedTokenizer


class ElementAllParenthesisTokenizer(PreTrainedTokenizer):
    """
    Element tokenizer that keeps parenthesis groups together.
    """

    ELEMENTS = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]

    ELEMENTS = sorted(ELEMENTS, key=lambda x: -len(x))  # longest first

    ELEMENT_PATTERN = "|".join(ELEMENTS)  # NO parentheses
    ATOM_LEVEL_PATTERN = (
        r"\[[^\]]+]|"
        + ELEMENT_PATTERN
        + r"|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9]"
    )

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
        **kwargs,
    ):
        """
        Initializes the ElementAllParenthesisTokenizer.

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
            **kwargs,
        )

    @property
    def vocab_size(self):
        """
        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vocab)

    def extract_parenthesis_groups(self, smiles):
        """
        Extracts groups enclosed in parentheses from the SMILES string.

        Args:
            smiles (str): The SMILES string.

        Returns:
            list: A list of strings, where parenthesis groups are kept as single items.
        """
        tokens = []
        i = 0
        n = len(smiles)

        while i < n:
            if smiles[i] == "(":
                start = i
                depth = 1
                i += 1
                while i < n and depth > 0:
                    if smiles[i] == "(":
                        depth += 1
                    elif smiles[i] == ")":
                        depth -= 1
                    i += 1
                tokens.append(smiles[start:i])
            else:
                tokens.append(smiles[i])
                i += 1

        return tokens

    def _tokenize(self, text):
        """
        Splits text into tokens, keeping parenthesis groups together.

        Args:
            text (str): The input text.

        Returns:
            list: A list of tokens.
        """
        parenthesis_groups = self.extract_parenthesis_groups(text)
        tokens = []
        for item in parenthesis_groups:
            if item.startswith("(") and item.endswith(")"):
                tokens.append(item)
            else:
                tokens.extend(re.findall(self.current_pattern, item))
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)

    def get_vocab(self):
        """
        Returns the vocabulary dictionary.

        Returns:
            dict: The vocabulary mapping tokens to IDs.
        """
        return self.vocab

    def _add_tokens(self, new_tokens, special_tokens=False):
        """
        Adds new tokens to the vocabulary.

        Args:
            new_tokens (list): A list of tokens to add.
            special_tokens (bool): Whether the tokens are special tokens. Defaults to False.

        Returns:
            int: The number of tokens added.
        """
        if not new_tokens:
            return 0
        
        added = 0
        for token in new_tokens:
            token_str = str(token)
            if token_str not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[token_str] = new_id
                self.decoder[new_id] = token_str
                added += 1
        return added

    def create_vocabulary(
        self,
        text,
        append_to_existing_vocabulary=False,
        vocab_path="../json/vocab_symbol_to_number.json",
        save_vocabulary=False,
    ):
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
        self.vocab = helpers.create_vocabulary(
            text,
            self._tokenize,
            append_to_existing_vocabulary,
            vocab_path,
            save_vocabulary,
            self.vocab,
        )
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
        return helpers._load_vocab_from_json(
            path, append_to_existing_vocabulary, self.vocab
        )

    def reset_vocabulary(self):
        """
        Resets the vocabulary to an empty state.
        """
        self.vocab = {
            self.unk_token: 0,
            self.pad_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.decoder = {v: k for k, v in self.vocab.items()}

    def load_vocabulary(self, vocab_path="../json/vocab_symbol_to_number.json"):
        """
        Loads a vocabulary from the specified JSON file.

        Args:
            vocab_path (str): Path to the JSON vocabulary file.
        """
        self.vocab = helpers.load_vocabulary(vocab_path)
        self.decoder = {v: k for k, v in self.vocab.items()}
