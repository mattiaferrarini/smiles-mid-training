from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import json
import os
from typing import Optional, Tuple, List

class SmilesBpeTokenizer(PreTrainedTokenizerFast):
    
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
             super().__init__(
                vocab_file=vocab_file,
                merges_file=merges_file,
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
                t for t in [unk_token, bos_token, eos_token, pad_token, "[START_SMILES]", "[END_SMILES]"] 
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

    def create_vocabulary(self, text, save_vocabulary=False, vocab_size=2000, min_frequency=2):
        """
        Trains the BPE tokenizer on the provided text.
        """
        print("Training BPE tokenizer...")
        
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
            "[END_SMILES]"
        ]
        # Filter duplicates and None
        special_tokens = list(set([t for t in special_tokens if t]))

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
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

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
            
        # Save vocab.json and merges.txt
        # The model.save method of tokenizers returns the paths
        files = self._tokenizer.model.save(save_directory, prefix=filename_prefix)
        
        return tuple(files)
    
    def reset_vocabulary(self):
        # Reset to a fresh BPE model with special tokens
        tokenizer_object = Tokenizer(models.BPE())
        special_tokens = [
            t for t in [self.unk_token, self.bos_token, self.eos_token, self.pad_token, "[START_SMILES]", "[END_SMILES]"] 
            if t is not None
        ]
        tokenizer_object.add_special_tokens(special_tokens)
        self._tokenizer = tokenizer_object

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()