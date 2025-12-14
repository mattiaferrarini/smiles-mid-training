import os
import json
import collections
import utils.helpers as helpers

from tqdm import tqdm
from utils.logging import get_logger
from transformers import PreTrainedTokenizer
from SmilesPE.pretokenizer import kmer_tokenizer

LOGGER = get_logger(__name__)


class KmerTokenizer(PreTrainedTokenizer):
    """
    Tokenizer that splits text into k-mers (n-grams).
    """

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
        Initializes the KmerTokenizer.

        Args:
            vocab_file (str, optional): Path to a JSON file containing the vocabulary mapping.
            unk_token (str): The unknown token. Defaults to "[UNK]".
            pad_token (str): The padding token. Defaults to "[PAD]".
            bos_token (str): The beginning of sequence token. Defaults to "[BOS]".
            eos_token (str): The end of sequence token. Defaults to "[EOS]".
            ngram (int): The size of the k-mer. Defaults to 4.
            stride (int): The stride for sliding window. Defaults to 1.
            max_vocab_size (int, optional): Maximum vocabulary size.
            **kwargs: Additional keyword arguments passed to `PreTrainedTokenizer`.
        """
        self.vocab = {}
        self.decoder = {}
        
        # Default values
        ngram = 4
        stride = 1
        max_vocab_size = None

        # Load from config if available
        if config and isinstance(config, dict) and "tokenizer" in config and "params" in config["tokenizer"]:
            params = config["tokenizer"]["params"]
            ngram = params.get("ngram", ngram)
            stride = params.get("stride", stride)
            max_vocab_size = params.get("max_vocab_size", max_vocab_size)
        
        # Override with kwargs (e.g. from from_pretrained)
        self.ngram = kwargs.pop("ngram", ngram)
        self.stride = kwargs.pop("stride", stride)
        self.max_vocab_size = kwargs.pop("max_vocab_size", max_vocab_size)

        if vocab_file:
            with open(vocab_file, encoding="utf-8") as f:
                self.vocab = json.load(f)
            print(f"Loaded vocabulary from {vocab_file} with size {len(self.vocab)}")
        else:
            self.vocab = {unk_token: 0, pad_token: 1, bos_token: 2, eos_token: 3}

        self.decoder = {v: k for k, v in self.vocab.items()}

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            ngram=ngram,
            stride=stride,
            max_vocab_size=max_vocab_size,
            **kwargs,
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
        Splits text into k-mers.

        Args:
            text (str): The input text.

        Returns:
            list: A list of k-mer tokens.
        """
        tokens = kmer_tokenizer(text, ngram=self.ngram, stride=self.stride)
       
        # Fallbacks for short SMILES 
        if not tokens and text:
            tokens = kmer_tokenizer(text, ngram=1, stride=self.stride)
        if not tokens:
            tokens = [""]
            
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

    def create_vocabulary(
        self,
        text_iterator,
        append_to_existing_vocabulary=False,
        vocab_path="../json/vocab_symbol_to_number.json",
        save_vocabulary=False,
    ):
        """
        Creates a vocabulary from the provided text iterator.

        Args:
            text_iterator (iterator): An iterator over the training data.
            append_to_existing_vocabulary (bool): Whether to append to an existing vocabulary. Defaults to False.
            vocab_path (str, optional): Path to load/save the vocabulary.
            save_vocabulary (bool): Whether to save the vocabulary. Defaults to False.

        Returns:
            dict: The created vocabulary.
        """
        LOGGER.info(
            f"Counting k-mers (ngram={self.ngram}, limit={self.max_vocab_size})..."
        )
        counter = collections.Counter()

        for i, item in enumerate(tqdm(text_iterator)):
            text = item["text"] if isinstance(item, dict) else item
            tokens = self._tokenize(text)
            counter.update(tokens)

        LOGGER.info(f"Total unique k-mers found: {len(counter)}")

        new_vocab = {
            self.unk_token: 0,
            self.pad_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        if self.max_vocab_size:
            limit = self.max_vocab_size - len(new_vocab)
            most_common = counter.most_common(limit)
        else:
            most_common = counter.most_common()

        for token, count in most_common:
            if token not in new_vocab:
                new_vocab[token] = len(new_vocab)

        self.vocab = new_vocab
        self.decoder = {v: k for k, v in self.vocab.items()}

        LOGGER.info(f"Final vocab size: {len(self.vocab)}")

        if save_vocabulary and vocab_path:
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)

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
