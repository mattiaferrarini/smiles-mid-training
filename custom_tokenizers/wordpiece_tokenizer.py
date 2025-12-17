import os
from utils.logging import get_logger
from transformers import PreTrainedTokenizerFast
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

LOGGER = get_logger(__name__)


class WordPieceTokenizer(PreTrainedTokenizerFast):
    """
    WordPiece tokenizer trained specifically on SMILES strings.
    """

    vocab_files_names = {
        "vocab_file": "vocab.txt",
        "tokenizer_file": "tokenizer.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

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
            vocab_file (str, optional): Path to a vocabulary file.
            tokenizer_file (str, optional): Path to a tokenizer.json file.
            unk_token (str): The unknown token. Defaults to "[UNK]".
            pad_token (str): The padding token. Defaults to "[PAD]".
            bos_token (str): The beginning of sequence token. Defaults to "[BOS]".
            eos_token (str): The end of sequence token. Defaults to "[EOS]".
            config (dict, optional): A dictionary containing tokenizer configuration parameters.
            **kwargs: Additional keyword arguments passed to `PreTrainedTokenizerFast`.
        """
        # Store config parameters
        self.config = config or {}
        self.max_vocab_size = (
            self.config["tokenizer"]["params"].get("max_vocab_size", 20000)
            if config
            else 20000
        )
        self.min_freq_for_merge = (
            self.config["tokenizer"]["params"].get("min_freq_for_merge", 2)
            if config
            else 2
        )

        print(
            f"Tokenizer config - max_vocab_size: {self.max_vocab_size}, min_freq_for_merge: {self.min_freq_for_merge}"
        )

        # If loading from files
        if tokenizer_file:
            super().__init__(
                tokenizer_file=tokenizer_file,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs,
            )

        elif vocab_file:
            super().__init__(
                vocab_file=vocab_file,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs,
            )
        else:
            # Initialize with a dummy WordPiece if no file provided (for training phase)
            tokenizer_object = Tokenizer(models.WordPiece(unk_token=unk_token))

            # Define special tokens
            special_tokens = [
                t
                for t in [
                    unk_token,
                    bos_token,
                    eos_token,
                    pad_token,
                    "[START_SMILES]",
                    "[END_SMILES]",
                ]
                if t is not None
            ]

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
        self, text, save_vocabulary=False, vocab_size=None, min_frequency=None
    ):
        """
        Trains the WordPiece tokenizer on the provided text.

        Args:
            text (iterator): An iterator over the training data.
            save_vocabulary (bool): Whether to save the vocabulary. Defaults to False.
            vocab_size (int, optional): The desired vocabulary size. Uses config value if not provided.
            min_frequency (int, optional): The minimum frequency for a token to be included. Uses config value if not provided.
        """
        LOGGER.info("Training WordPiece tokenizer.")

        vocab_size = vocab_size or self.max_vocab_size
        min_frequency = min_frequency or self.min_freq_for_merge

        tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece()

        special_tokens = [
            self.unk_token if self.unk_token else "[UNK]",
            self.bos_token if self.bos_token else "[BOS]",
            self.eos_token if self.eos_token else "[EOS]",
            self.pad_token if self.pad_token else "[PAD]",
            "[START_SMILES]",
            "[END_SMILES]",
        ]
        special_tokens = list(set([t for t in special_tokens if t]))

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

        if isinstance(text, str):
            iterator = [text]
        else:
            iterator = text

        tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._tokenizer = tokenizer

        return self.get_vocab()

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Saves the vocabulary to a file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        files = self._tokenizer.model.save(save_directory, prefix=filename_prefix)

        return tuple(files)

    def reset_vocabulary(self):
        """
        Resets the vocabulary to an empty state.
        """
        tokenizer_object = Tokenizer(models.WordPiece(unk_token=self.unk_token))
        special_tokens = [
            t
            for t in [
                self.unk_token,
                self.bos_token,
                self.eos_token,
                self.pad_token,
                "[START_SMILES]",
                "[END_SMILES]",
            ]
            if t is not None
        ]
        tokenizer_object.add_special_tokens(special_tokens)
        self._tokenizer = tokenizer_object

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()