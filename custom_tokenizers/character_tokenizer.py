import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import json
import utils.helpers as helpers
from transformers import PreTrainedTokenizer


class CharacterTokenizer(PreTrainedTokenizer):
    """
    Tokenizer that splits text into individual characters.
    """

    CHAR_LEVEL_PATTERN = r"."

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
        Initializes the CharacterTokenizer.

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
        self.current_pattern = self.CHAR_LEVEL_PATTERN
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    def get_vocab(self):
        """
        Returns the vocabulary dictionary.

        Returns:
            dict: The vocabulary mapping tokens to IDs.
        """
        return self.vocab

    def __len__(self):
        """
        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vocab)

    @property  # for Huggingface compatibility
    def vocab_size(self):
        """
        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vocab)

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves the tokenizer vocabulary and configuration to the specified directory.

        Args:
            save_directory (str): The directory to save the tokenizer files.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: Paths to the saved files.
        """
        save_directory = Path(save_directory)

        config_dict = {
            "max_len": getattr(self, "model_max_length", 1024),
            "unk_token": "[UNK]",
            "model_type": "character_tokenizer",
        }

        vocab_file, config_file = helpers.save_tokenizer_files(
            save_directory=save_directory,
            vocab_dict=self.vocab,
            config_dict=config_dict,
        )

        return vocab_file, config_file

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Loads a tokenizer from a pretrained model directory.

        Args:
            pretrained_model_name_or_path (str): The directory containing the pretrained tokenizer files.
            **kwargs: Additional keyword arguments.

        Returns:
            CharacterTokenizer: The loaded tokenizer instance.
        """
        tokenizer = cls() 
        vocab_path = Path(pretrained_model_name_or_path) / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)

        return tokenizer

    def _encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,  
        padding_strategy="do_not_pad",
        truncation_strategy="do_not_truncate",
        max_length=None,
        is_split_into_words=False,
        **kwargs,
    ):
        """
        Tokenizes and encodes the input text.

        Args:
            text (str): The input text to encode.
            text_pair (str, optional): A second input text for sequence pairs.
            add_special_tokens (bool): Whether to add special tokens. Defaults to True.
            padding_strategy (str): The padding strategy. Defaults to "do_not_pad".
            truncation_strategy (str): The truncation strategy. Defaults to "do_not_truncate".
            max_length (int, optional): The maximum length of the sequence.
            is_split_into_words (bool): Whether the input is already split into words. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            BatchEncoding: The encoded output containing input_ids and attention_mask.
        """

        tokens = self._tokenize(text)

        input_ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def __call__(self, text, **kwargs):
        """
        Tokenizes and encodes the input text.
        """
        return self._encode_plus(text, **kwargs)

    def decode(self, token_ids):
        """
        Decodes a sequence of token IDs back into a string.

        Args:
            token_ids (list): A list of token IDs.

        Returns:
            str: The decoded string.
        """
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([reverse_vocab.get(tid, "[UNK]") for tid in token_ids])

    def _tokenize(self, text):
        """
        Splits text into characters.

        Args:
            text (str): The input text.

        Returns:
            list: A list of character tokens.
        """
        tokens = re.findall(self.current_pattern, text)

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
        self.vocab = helpers.reset_vocabulary()

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

    def load_vocabulary(self, vocab_path="../json/vocab_symbol_to_number.json"):
        """
        Loads a vocabulary from the specified JSON file.

        Args:
            vocab_path (str): Path to the JSON vocabulary file.
        """
        self.vocab = helpers.load_vocabulary(vocab_path)

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
