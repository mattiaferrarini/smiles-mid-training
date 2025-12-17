import os
import json
from pathlib import Path
from utils.logging import get_logger
from transformers import PreTrainedTokenizerFast
from tokenizers import models, pre_tokenizers, trainers, Tokenizer

LOGGER = get_logger(__name__)


class SmilesWordPieceTokenizer(PreTrainedTokenizerFast):
    """
    Modified SMILES Pair Encoding tokenizer using HuggingFace's WordPiece implementation.
    """

    vocab_files_names = {
        "vocab_file": "vocab.txt",
        "tokenizer_file": "tokenizer.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def get_pretokenization_pattern(self):
        """
        This pattern is designed to split SMILES strings into chemically meaningful units
        (atoms, bonds, rings, branches) before the WordPiece algorithm is applied.

        Returns:
            str: The regex pattern string.
        """
        return r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
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
            tokenizer_file (str, optional): Path to a tokenizer.json file.
            unk_token (str): The unknown token. Defaults to "[UNK]".
            pad_token (str): The padding token. Defaults to "[PAD]".
            bos_token (str): The beginning of sequence token. Defaults to "[BOS]".
            eos_token (str): The end of sequence token. Defaults to "[EOS]".
            config (dict, optional): A dictionary containing tokenizer configuration parameters.
            **kwargs: Additional keyword arguments passed to `PreTrainedTokenizerFast`.
        """
        # Get config parameters
        if config and isinstance(config, dict) and "tokenizer" in config:
            self.max_vocab_size = config["tokenizer"]["params"].get(
                "max_vocab_size", 20000
            )
            self.min_freq_for_merge = config["tokenizer"]["params"].get(
                "min_freq_for_merge", 2
            )
        else:
            self.max_vocab_size = kwargs.get("max_vocab_size", 20000)
            self.min_freq_for_merge = kwargs.get("min_freq_for_merge", 2)

        # If loading from files
        if tokenizer_file and os.path.exists(tokenizer_file):
            super().__init__(
                tokenizer_file=tokenizer_file,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs,
            )
        elif vocab_file and os.path.exists(vocab_file):
            super().__init__(
                vocab_file=vocab_file,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs,
            )
        else:
            # Initialize with WordPiece model for training
            tokenizer_object = Tokenizer(models.WordPiece(unk_token=unk_token))

            # Set pre-tokenizer using built-in Split pattern for SMILES
            tokenizer_object.pre_tokenizer = pre_tokenizers.Split(
                pattern=self.get_pretokenization_pattern(),
                behavior="isolated",  # Keep matched tokens
                invert=False,  # Split by matches (not by separators)
            )

            # Define special tokens
            special_tokens = [
                t for t in [unk_token, bos_token, eos_token, pad_token] if t is not None
            ]

            # Add special tokens
            if special_tokens:
                tokenizer_object.add_special_tokens(special_tokens)

            super().__init__(
                tokenizer_object=tokenizer_object,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs,
            )

    def create_vocabulary(
        self,
        text,
        append_to_existing_vocabulary=False,
        vocab_path=None,
        save_vocabulary=False,
    ):
        """
        Train the WordPiece tokenizer on SMILES data using HuggingFace's fast implementation.

        Args:
            text: List of SMILES strings or single SMILES string
            append_to_existing_vocabulary: Not used, kept for compatibility
            vocab_path: Not used, kept for compatibility
            save_vocabulary: Not used, kept for compatibility
        """
        LOGGER.info(
            f"Training SmilesWordPieceTokenizer with vocab_size={self.max_vocab_size}, min_frequency={self.min_freq_for_merge}."
        )

        # Initialize WordPiece Tokenizer with SMILES pre-tokenizer
        unk_token = self.unk_token if self.unk_token else "[UNK]"
        tokenizer = Tokenizer(models.WordPiece(unk_token=unk_token))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=self.get_pretokenization_pattern(),
            behavior="isolated",
            invert=False,
        )

        # Define special tokens
        special_tokens = [
            self.unk_token if self.unk_token else "[UNK]",
            self.bos_token if self.bos_token else "[BOS]",
            self.eos_token if self.eos_token else "[EOS]",
            self.pad_token if self.pad_token else "[PAD]",
        ]
        # Filter duplicates and None
        special_tokens = list(set([t for t in special_tokens if t]))

        # Create WordPiece trainer with parameters
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.max_vocab_size,
            min_frequency=self.min_freq_for_merge,
            special_tokens=special_tokens,
            show_progress=True,
            continuing_subword_prefix="##",  # Standard WordPiece prefix for continuation tokens
        )

        # Handle both single string and list of strings
        if isinstance(text, str):
            iterator = [text]
        else:
            iterator = text

        LOGGER.info(f"Training on {len(iterator)} SMILES sequences.")
        tokenizer.train_from_iterator(iterator, trainer=trainer)

        # Update the underlying tokenizer
        self._tokenizer = tokenizer

        LOGGER.info(f"Training complete! Vocabulary size: {self.vocab_size}")

        return self.get_vocab()

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save vocabulary files

        Args:
            save_directory (str): The directory where the vocabulary will be saved.
            filename_prefix (str, optional): A prefix to add to the file names.

        Returns:
            Tuple[str]: A tuple containing the paths to the saved files.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        # Save using the model's save method
        files = self._tokenizer.model.save(save_directory, prefix=filename_prefix)

        return tuple(files)

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves tokenizer in HuggingFace-compatible format

        Args:
            save_directory (str): The directory where the files will be saved.
            **kwargs: Additional keyword arguments passed to the parent `save_pretrained`.

        Returns:
            Tuple[str, str]: Paths to the `tokenizer.json` and `tokenizer_config.json` files.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save the full tokenizer (includes vocab and config)
        super().save_pretrained(str(save_directory), **kwargs)

        # Also save explicit config for compatibility
        config_dict = {
            "max_len": getattr(self, "model_max_length", 1024),
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "model_type": "smiles_wordpiece_tokenizer",
            "max_vocab_size": self.max_vocab_size,
            "min_freq_for_merge": self.min_freq_for_merge,
        }

        config_file = save_directory / "tokenizer_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        LOGGER.info(f"SmilesWordPieceTokenizer saved in {save_directory}")
        return str(save_directory / "tokenizer.json"), str(config_file)

    @classmethod
    def from_pretrained(cls, pretrained_directory, **kwargs):
        """
        Loads tokenizer from HuggingFace-compatible format

        It attempts to load `tokenizer.json` first. If not found, it attempts to load
        from `vocab.json` (or `vocab.txt`) and `tokenizer_config.json`.

        Args:
            pretrained_directory (str): The directory containing the tokenizer files.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            SmilesWordPieceTokenizer: The loaded tokenizer instance.

        Raises:
            FileNotFoundError: If `tokenizer.json` and `vocab.json` are not found.
        """
        pretrained_directory = Path(pretrained_directory)

        # Load config if exists to ensure we get custom parameters
        config_file = pretrained_directory / "tokenizer_config.json"
        if config_file.is_file():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Add config parameters to kwargs if not already present
            for key, value in config.items():
                if key not in kwargs:
                    kwargs[key] = value

        # Try loading tokenizer.json first (full serialization)
        tokenizer_file = pretrained_directory / "tokenizer.json"
        if tokenizer_file.is_file():
            return super().from_pretrained(str(pretrained_directory), **kwargs)

        # Fallback to vocab.txt
        vocab_file = pretrained_directory / "vocab.txt"

        if not vocab_file.is_file():
            raise FileNotFoundError(
                f"Could not find tokenizer files in {pretrained_directory}. "
                f"Expected either tokenizer.json or vocab.txt"
            )

        return cls(vocab_file=str(vocab_file), **kwargs)

    @property
    def vocab_size(self):
        """
        Returns:
            int: The size of the vocabulary.
        """
        return self._tokenizer.get_vocab_size()

    def reset_vocabulary(self):
        """
        Reset to a fresh BPE model with special tokens
        """
        unk_token = self.unk_token if self.unk_token else "[UNK]"
        tokenizer_object = Tokenizer(models.WordPiece(unk_token=unk_token))
        tokenizer_object.pre_tokenizer = pre_tokenizers.Split(
            pattern=self.get_pretokenization_pattern(),
            behavior="isolated",
            invert=False,
        )

        special_tokens = [
            t
            for t in [self.unk_token, self.bos_token, self.eos_token, self.pad_token]
            if t is not None
        ]
        if special_tokens:
            tokenizer_object.add_special_tokens(special_tokens)

        self._tokenizer = tokenizer_object
