import os
from utils.logging import get_logger
from transformers import PreTrainedTokenizerFast
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

LOGGER = get_logger(__name__)


class SmilesBpeTokenizer(PreTrainedTokenizerFast):
    """
    BPE tokenizer trained specifically on SMILES strings.
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "tokenizer_file": "tokenizer.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        config=None,
        **kwargs
    ):
        """
        Initializes the SmilesBpeTokenizer.

        Args:
            vocab_file (str, optional): Path to a JSON file containing the vocabulary mapping.
            merges_file (str, optional): Path to a merges.txt file.
            tokenizer_file (str, optional): Path to a tokenizer.json file.
            unk_token (str): The unknown token. Defaults to "[UNK]".
            pad_token (str): The padding token. Defaults to "[PAD]".
            bos_token (str): The beginning of sequence token. Defaults to "[BOS]".
            eos_token (str): The end of sequence token. Defaults to "[EOS]".
            config (dict, optional): A dictionary containing tokenizer configuration parameters.
            **kwargs: Additional keyword arguments passed to `PreTrainedTokenizerFast`.
        """
        # If loading from files
        if tokenizer_file:
            super().__init__(
                tokenizer_file=tokenizer_file,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs
            )
        elif vocab_file and merges_file:
            tokenizer_object = Tokenizer(models.BPE.from_file(vocab_file, merges_file))
            super().__init__(
                tokenizer_object=tokenizer_object,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs
            )
        else:
            # Initialize with a dummy BPE if no file provided (for training phase)
            # This allows instantiation before training
            tokenizer_object = Tokenizer(models.BPE())

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

            # Add special tokens to the dummy tokenizer so it doesn't complain
            tokenizer_object.add_special_tokens(special_tokens)

            super().__init__(
                tokenizer_object=tokenizer_object,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs
            )

    def create_vocabulary(
        self, text, save_vocabulary=False, vocab_size=2000, min_frequency=2
    ):
        """
        Trains the BPE tokenizer on the provided text.

        Args:
            text (iterator): An iterator over the training data.
            save_vocabulary (bool): Whether to save the vocabulary. Defaults to False.
            vocab_size (int): The desired vocabulary size. Defaults to 2000.
            min_frequency (int): The minimum frequency for a token to be included. Defaults to 2.
        """
        LOGGER.info("Training BPE tokenizer...")

        # Initialize BPE Tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.ByteLevel()

        # Define special tokens
        special_tokens = [
            self.unk_token if self.unk_token else "[UNK]",
            self.bos_token if self.bos_token else "[BOS]",
            self.eos_token if self.eos_token else "[EOS]",
            self.pad_token if self.pad_token else "[PAD]",
            "[START_SMILES]",
            "[END_SMILES]",
        ]
        # Filter duplicates and None
        special_tokens = list(set([t for t in special_tokens if t]))

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

        # Train on the provided text
        # text can be a single string or a list of strings
        if isinstance(text, str):
            iterator = [text]
        else:
            iterator = text

        tokenizer.train_from_iterator(iterator, trainer=trainer)

        # Update the underlying tokenizer of this instance
        self._tokenizer = tokenizer

        return self.get_vocab()

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Saves the vocabulary to a file.

        Args:
            save_directory (str): The directory to save the vocabulary.
            filename_prefix (str, optional): Prefix for the vocabulary file name.

        Returns:
            tuple: Path to the saved vocabulary file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        # Save vocab.json and merges.txt
        # The model.save method of tokenizers returns the paths
        files = self._tokenizer.model.save(save_directory, prefix=filename_prefix)

        return tuple(files)

    def reset_vocabulary(self):
        """
        Resets the vocabulary to an empty state.
        """
        # Reset to a fresh BPE model with special tokens
        tokenizer_object = Tokenizer(models.BPE())
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
        """
        Returns:
            int: The size of the vocabulary.
        """
        return self._tokenizer.get_vocab_size()
