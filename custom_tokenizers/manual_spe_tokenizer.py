import os
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from utils.logging import get_logger
from transformers import PreTrainedTokenizer

LOGGER = get_logger(__name__)


class ManualSPETokenizer(PreTrainedTokenizer):
    """
    Non-optimized implementation of SPE to allow custom merge scoring.
    Adapted and simplified from both https://github.com/mikemayuare/apetokenizer
    and https://github.com/XinhaoLi74/SmilesPE.
    """

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
        Initializes the tokenizer.

        Args:
            vocab_file (str, optional): Path to a JSON file containing the vocabulary mapping.
            unk_token (str): The unknown token. Defaults to "[UNK]".
            pad_token (str): The padding token. Defaults to "[PAD]".
            bos_token (str): The beginning of sequence token. Defaults to "[BOS]".
            eos_token (str): The end of sequence token. Defaults to "[EOS]".
            config (dict, optional): A dictionary containing tokenizer configuration parameters
            **kwargs: Additional keyword arguments passed to `PreTrainedTokenizer`.
        """

        self.max_vocab_size = (
            config["tokenizer"]["params"].get("max_vocab_size", 20000)
            if config
            else 20000
        )
        self.min_freq_for_merge = (
            config["tokenizer"]["params"].get("min_freq_for_merge", 0) if config else 0
        )
        self.vocabulary_frequency = defaultdict(int)
        self.pair_counts = defaultdict(int)

        # Initialize vocabulary
        self.vocab = {}
        self.decoder = {}

        if vocab_file:
            with open(vocab_file, encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {unk_token: 0, pad_token: 1, bos_token: 2, eos_token: 3}

        self.decoder = {v: k for k, v in self.vocab.items()}

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

        # Store special tokens for compatibility
        self.special_tokens = {
            unk_token: self.vocab[unk_token],
            pad_token: self.vocab[pad_token],
            bos_token: self.vocab[bos_token],
            eos_token: self.vocab[eos_token],
        }

        # Alias vocabulary to vocab for compatibility
        self.vocabulary = self.vocab

    @property
    def vocab_size(self):
        """
        Required by PreTrainedTokenizerBase

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vocab)

    def _add_tokens(self, new_tokens, special_tokens=False):
        """
        Adds new tokens to the vocabulary.
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

    @property
    def unk_token_id(self):
        """
        Returns:
            int: The ID of the unknown token.
        """
        return self.vocab.get(self.unk_token, 0)

    def __len__(self):
        """
        Returns:
            int: The length of the vocabulary
        """
        return len(self.vocab)

    def _tokenize(self, text):
        """
        Tokenizes the input text into merged tokens using the learned vocabulary.
        Implements a greedy longest-match strategy.

        Args:
            text (str): The input text to tokenize.
        Returns:
            List[str]: A list of string tokens.
        """
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            match = None
            # Scan for the longest substring starting at i that exists in the vocabulary
            for j in range(n, i, -1):
                possible_match = text[i:j]
                if possible_match in self.vocab:
                    match = possible_match
                    break

            if match:
                tokens.append(match)
                i += len(match)
            else:
                # If no match found, use the unknown token
                tokens.append(self.unk_token)
                i += 1
        return tokens

    def _convert_token_to_id(self, token):
        """
        Converts a token string to its integer ID.
        Required by PreTrainedTokenizerBase

        Args:
            token (str): The token string to convert.
        Returns:
            int: The integer ID of the token.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index):
        """
        Converts a token string to its integer ID.
        Required by PreTrainedTokenizerBase

        Args:
            index (int): The integer ID to convert.

        Returns:
            str: The token string corresponding to the ID.
        """
        return self.decoder.get(index, self.unk_token)

    def pre_tokenize(self, molecule):
        """
        Pre-tokenizes a SMILES string into initial tokens based on atom-level patterns.
        Used during training.

        Args:
            molecule (str): The SMILES string to pre-tokenize.

        Returns:
            List[str]: A list of pre-tokenized string tokens.
        """
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        words = re.findall(pattern, molecule)
        return words

    def get_most_common_pair(self, words):
        """
        Identifies the most common adjacent token pair in the list of words.

        Args:
            words (List[str]): A list of string tokens.

        Returns:
            Tuple[Tuple[str, str], int]: The most common token pair and its frequency.
        """
        for i in range(len(words) - 1):
            pair = (words[i], words[i + 1])
            self.pair_counts[pair] += 1

        # Minimize lookups by using max function directly
        most_common_pair, freq = max(
            self.pair_counts.items(), key=self.score_item, default=((None, None), 0)
        )
        return most_common_pair, freq

    def score_item(self, item):
        """
        Helper for sorting/max functions based on frequency.

        Args:
            item (tuple): A tuple containing the pair and its frequency.

        Returns:
            int: The frequency of the pair.
        """
        return item[1]

    def _train(self, corpus, max_vocab_size=None, min_freq_for_merge=None):
        """
        Executes the training algorithm.

        It pre-tokenizes the corpus, then iteratively merges the most frequent adjacent pairs
        until the target vocabulary size is reached or no more pairs satisfy the minimum frequency.

        Args:
            corpus (List[str]): List of SMILES strings to train on.
            max_vocab_size (int, optional): The target vocabulary size. Defaults to instance config.
            min_freq_for_merge (int, optional): The minimum frequency required to merge a pair. Defaults to instance config.
        """
        if max_vocab_size is None:
            max_vocab_size = self.max_vocab_size
        if min_freq_for_merge is None:
            min_freq_for_merge = self.min_freq_for_merge

        # Preprocessing: Tokenize and count word frequencies upfront
        LOGGER.info(f"Starting tokenization training on {len(corpus)} sequences.")
        print("Pretokenizing corpus.", end="\r")
        words = [word for sentence in corpus for word in self.pre_tokenize(sentence)]
        vocabulary_frequency = defaultdict(int)
        for word in words:
            vocabulary_frequency[word] += 1
        LOGGER.info(
            f"Pretokenization complete! Found {len(vocabulary_frequency)} initial tokens from {len(words)} total tokens"
        )

        merged_counter = len(vocabulary_frequency) + 1
        iteration = 0

        LOGGER.info(
            f"Starting merge iterations (target vocab size: {max_vocab_size}, min frequency: {min_freq_for_merge})."
        )

        while True:
            iteration += 1

            if len(vocabulary_frequency) > self.max_vocab_size:
                LOGGER.info(
                    f"Max vocabulary size reached: {len(vocabulary_frequency)} tokens"
                )
                break

            most_common_pair, freq = self.get_most_common_pair(words)
            if freq < self.min_freq_for_merge:
                LOGGER.info(
                    f"Stopping: pair frequency ({freq}) below minimum threshold ({self.min_freq_for_merge})"
                )
                break

            merged_word = "".join(most_common_pair)
            if merged_word not in vocabulary_frequency.keys():
                progress_pct = round(merged_counter / max_vocab_size * 100, 2)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                LOGGER.info(
                    f"[{timestamp}] Iteration {iteration}: Merging '{most_common_pair[0]}' + '{most_common_pair[1]}' → '{merged_word}' "
                    f"(freq: {freq}) | Vocab: {merged_counter}/{max_vocab_size} ({progress_pct}%)"
                )
                merged_counter += 1
            merged_word_freq = vocabulary_frequency.get(merged_word, 0)
            vocabulary_frequency[merged_word] = merged_word_freq + freq

            # Minimize dictionary lookups inside the loop
            new_words = []
            skip_next = False
            for i in range(len(words)):
                if skip_next:
                    skip_next = False
                    continue

                # Look ahead to minimize lookups
                if (
                    i < len(words) - 1
                    and words[i] == most_common_pair[0]
                    and words[i + 1] == most_common_pair[1]
                ):
                    new_words.append(merged_word)
                    skip_next = True
                else:
                    new_words.append(words[i])

            words = new_words

            # Clear pair counts for next iteration
            self.pair_counts.clear()

        # Convert vocabulary_frequency to a regular dictionary for final output
        self.vocabulary_frequency = dict(vocabulary_frequency)

        # Build vocab from special tokens + learned vocabulary
        # Start from next available ID after special tokens
        max_special_id = max(self.special_tokens.values())
        self.vocab = {**self.special_tokens}
        for idx, word in enumerate(
            vocabulary_frequency.keys(), start=max_special_id + 1
        ):
            if word not in self.vocab:  # Don't override special tokens
                self.vocab[word] = idx

        # Update vocabulary alias and decoder
        self.vocabulary = self.vocab
        self.decoder = {v: k for k, v in self.vocab.items()}

        LOGGER.info("Training complete.")

        return None

    def create_vocabulary(
        self,
        text,
        append_to_existing_vocabulary=False,
        vocab_path=None,
        save_vocabulary=False,
    ):
        """
        Wrapper method compatible with build_and_save_tokenizer.
        Accepts a list of strings (SMILES) and trains the tokenizer on them.

        Args:
            text: List of strings or single string to train on
            append_to_existing_vocabulary: Not used, kept for compatibility
            vocab_path: Not used, kept for compatibility
            save_vocabulary: Not used, kept for compatibility

        Returns:
            dict: The trained vocabulary mapping.
        """
        # Handle both single string and list of strings
        if isinstance(text, str):
            corpus = [text]
        else:
            corpus = text

        LOGGER.info(f"Training Manual SPE tokenizer on {len(corpus)} sequences.")
        self._train(
            corpus,
            max_vocab_size=self.max_vocab_size,
            min_freq_for_merge=self.min_freq_for_merge,
        )

        return self.vocab

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a token string or a list of token strings into their integer IDs.

        Args:
            tokens (str or List[str]): Input token(s).

        Returns:
            int or List[int]: The corresponding ID(s).
        """
        if isinstance(tokens, str):  # Single token
            return self.vocab.get(tokens, self.vocab.get(self.unk_token, 0))
        else:  # List of tokens
            return [
                self.vocab.get(token, self.vocab.get(self.unk_token, 0))
                for token in tokens
            ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts a token ID or a list of token IDs back into their string tokens.

        Args:
            token_ids (int or List[int]): Input token ID(s).

        Returns:
            str or List[str]: The corresponding token string(s).
        """
        # Map each token ID to its corresponding string token
        return [self.decoder.get(token_id, self.unk_token) for token_id in token_ids]

    def get_vocab(self):
        """
        Converts a list of integer IDs back into their string tokens.
        Required by PreTrainedTokenizerBase

        Args:
            token_ids (List[int]): List of token IDs.

        Returns:
            List[str]: The corresponding list of string tokens.
        """
        return self.vocab.copy()

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Saves the vocabulary to a JSON file.

        Required by PreTrainedTokenizerBase.

        Args:
            save_directory (str): Directory where the file will be saved.
            filename_prefix (str, optional): Prefix for the filename.

        Returns:
            Tuple[str]: Path to the saved vocabulary file.
        """
        if filename_prefix:
            vocab_file = f"{filename_prefix}-vocab.json"
        else:
            vocab_file = "vocab.json"

        path = os.path.join(save_directory, vocab_file)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (path,)

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves the tokenizer vocabulary, configuration, and training state (optional)
        to the specified directory.

        Args:
            save_directory (str): The output directory.
            **kwargs: Additional keyword arguments passed to the parent method.

        Returns:
            Tuple[str, str]: Paths to the vocabulary file and config file.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save vocabulary using standard method
        self.save_vocabulary(str(save_directory))

        # Save tokenizer config
        config_dict = {
            "max_len": getattr(self, "model_max_length", 1024),
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "model_type": "manual_spe",
            "max_vocab_size": self.max_vocab_size,
            "min_freq_for_merge": self.min_freq_for_merge,
        }

        config_file = save_directory / "tokenizer_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        # Optionally save training state for further training
        if self.vocabulary_frequency:
            vocabulary_frequency_serializable = {
                str(k): v for k, v in self.vocabulary_frequency.items()
            }
            pair_counts_serializable = {str(k): v for k, v in self.pair_counts.items()}

            training_state = {
                "vocabulary_frequency": vocabulary_frequency_serializable,
                "pair_counts": pair_counts_serializable,
            }

            training_state_file = save_directory / "training_state.json"
            with open(training_state_file, "w", encoding="utf-8") as f:
                json.dump(training_state, f, ensure_ascii=False, indent=4)

        LOGGER.info(f"Tokenizer saved in {save_directory}")
        return str(save_directory / "vocab.json"), str(config_file)

    @classmethod
    def from_pretrained(cls, pretrained_directory, **kwargs):
        """
        Loads the tokenizer from a directory containing saved files (vocab.json, tokenizer_config.json).

        Args:
            pretrained_directory (str): Directory containing the tokenizer files.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            ManualSPE: The loaded tokenizer instance.

        Raises:
            FileNotFoundError: If `vocab.json` is missing.
        """
        pretrained_directory = Path(pretrained_directory)
        vocab_file = pretrained_directory / "vocab.json"
        config_file = pretrained_directory / "tokenizer_config.json"
        training_state_file = pretrained_directory / "training_state.json"

        # Load vocabulary
        if not vocab_file.is_file():
            raise FileNotFoundError(f"Vocabulary file {vocab_file} not found.")

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocabulary = json.load(f)

        # Load config if exists
        config = {}
        if config_file.is_file():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

        # Initialize the tokenizer with config
        tokenizer = cls(
            unk_token=config.get("unk_token", "[UNK]"),
            pad_token=config.get("pad_token", "[PAD]"),
            bos_token=config.get("bos_token", "[BOS]"),
            eos_token=config.get("eos_token", "[EOS]"),
            max_vocab_size=config.get("max_vocab_size", 5000),
            min_freq_for_merge=config.get("min_freq_for_merge", 2000),
        )

        # Load the vocabulary
        tokenizer.vocab = vocabulary
        tokenizer.vocabulary = vocabulary
        tokenizer.decoder = {v: k for k, v in vocabulary.items()}

        # Load training state if it exists
        if training_state_file.is_file():
            with open(training_state_file, "r", encoding="utf-8") as f:
                training_state = json.load(f)
            tokenizer.vocabulary_frequency = defaultdict(
                int, training_state.get("vocabulary_frequency", {})
            )
            tokenizer.pair_counts = defaultdict(
                int, training_state.get("pair_counts", {})
            )

        return tokenizer
