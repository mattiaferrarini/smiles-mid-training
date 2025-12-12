'''
Optimized APE (Atom Pair Encoding) tokenizer using HuggingFace's fast BPE implementation.
This replaces the slow custom BPE training with tokenizers library for significant speedup.
'''

import os
import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from tokenizers import models, pre_tokenizers, trainers, Tokenizer

class APEHFTokenizer(PreTrainedTokenizerFast):
    """
    Fast APE tokenizer using HuggingFace's optimized BPE implementation.
    Compatible with build_tokenizer.py and assemble_tokenizer.py workflows.
    """
    
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "tokenizer_file": "tokenizer.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def get_pretokenization_pattern(self):
        """
        Returns the regex pattern for pre-tokenization.
        Override this method to customize pre-tokenization behavior.
        """
        return r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

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
        # Get config parameters
        self.max_vocab_size = config["tokenizer"]["params"].get("max_vocab_size", 20000) if config else 20000
        self.min_freq_for_merge = config["tokenizer"]["params"].get("min_freq_for_merge", 2) if config else 2
        
        # If loading from files
        if tokenizer_file and os.path.exists(tokenizer_file):
            super().__init__(
                tokenizer_file=tokenizer_file,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                **kwargs
            )
        elif vocab_file and merges_file and os.path.exists(vocab_file) and os.path.exists(merges_file):
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
            # Initialize with BPE model for training
            tokenizer_object = Tokenizer(models.BPE())
            
            # Set pre-tokenizer using built-in Split pattern for SMILES
            tokenizer_object.pre_tokenizer = pre_tokenizers.Split(
                pattern=self.get_pretokenization_pattern(),
                behavior="isolated",  # Keep matched tokens
                invert=False  # Split by matches (not by separators)
            )
            
            # Define special tokens
            special_tokens = [
                t for t in [unk_token, bos_token, eos_token, pad_token] 
                if t is not None
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
                **kwargs
            )

    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path=None, save_vocabulary=False):
        """
        Train the BPE tokenizer on SMILES data using HuggingFace's fast implementation.
        
        Args:
            text: List of SMILES strings or single SMILES string
            append_to_existing_vocabulary: Not used, kept for compatibility
            vocab_path: Not used, kept for compatibility  
            save_vocabulary: Not used, kept for compatibility
        """
        print(f"Training APE-HF tokenizer with vocab_size={self.max_vocab_size}, min_frequency={self.min_freq_for_merge}...")
        
        # Initialize BPE Tokenizer with SMILES pre-tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=self.get_pretokenization_pattern(),
            behavior="isolated",
            invert=False
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

        # Create BPE trainer with APE parameters
        trainer = trainers.BpeTrainer(
            vocab_size=self.max_vocab_size,
            min_frequency=self.min_freq_for_merge,
            special_tokens=special_tokens,
            show_progress=True,
            initial_alphabet=[]  # Let BPE discover the alphabet from pre-tokenized units
        )
        
        # Handle both single string and list of strings
        if isinstance(text, str):
            iterator = [text]
        else:
            iterator = text
        
        print(f"Training on {len(iterator)} SMILES sequences...")
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        
        # Update the underlying tokenizer
        self._tokenizer = tokenizer
        
        print(f"Training complete! Vocabulary size: {self.vocab_size}")
        
        return self.get_vocab()

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save vocabulary files (vocab.json and merges.txt)"""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
            
        # Save using the model's save method
        files = self._tokenizer.model.save(save_directory, prefix=filename_prefix)
        
        return tuple(files)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save tokenizer in HuggingFace-compatible format"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save the full tokenizer (includes vocab, merges, and config)
        super().save_pretrained(str(save_directory), **kwargs)
        
        # Also save explicit config for compatibility
        config_dict = {
            "max_len": getattr(self, "model_max_length", 1024),
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "model_type": "ape_hf_tokenizer",
            "max_vocab_size": self.max_vocab_size,
            "min_freq_for_merge": self.min_freq_for_merge,
        }
        
        config_file = save_directory / "tokenizer_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"APE-HF tokenizer saved in {save_directory}")
        return str(save_directory / "tokenizer.json"), str(config_file)

    @classmethod
    def from_pretrained(cls, pretrained_directory, **kwargs):
        """Load tokenizer from HuggingFace-compatible format"""
        pretrained_directory = Path(pretrained_directory)
        
        # Try loading tokenizer.json first (full serialization)
        tokenizer_file = pretrained_directory / "tokenizer.json"
        if tokenizer_file.is_file():
            return super().from_pretrained(str(pretrained_directory), **kwargs)
        
        # Fallback to vocab + merges
        vocab_file = pretrained_directory / "vocab.json"
        merges_file = pretrained_directory / "merges.txt"
        
        if not vocab_file.is_file() or not merges_file.is_file():
            raise FileNotFoundError(
                f"Could not find tokenizer files in {pretrained_directory}. "
                f"Expected either tokenizer.json or both vocab.json and merges.txt"
            )
        
        # Load config if exists
        config_file = pretrained_directory / "tokenizer_config.json"
        config = {}
        if config_file.is_file():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        return cls(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file),
            unk_token=config.get("unk_token", "[UNK]"),
            pad_token=config.get("pad_token", "[PAD]"),
            bos_token=config.get("bos_token", "[BOS]"),
            eos_token=config.get("eos_token", "[EOS]"),
            **kwargs
        )

    @property
    def vocab_size(self):
        """Return vocabulary size"""
        return self._tokenizer.get_vocab_size()

    def reset_vocabulary(self):
        """Reset to a fresh BPE model with special tokens"""
        tokenizer_object = Tokenizer(models.BPE())
        tokenizer_object.pre_tokenizer = pre_tokenizers.Split(
            pattern=self.get_pretokenization_pattern(),
            behavior="isolated",
            invert=False
        )
        
        special_tokens = [
            t for t in [self.unk_token, self.bos_token, self.eos_token, self.pad_token] 
            if t is not None
        ]
        if special_tokens:
            tokenizer_object.add_special_tokens(special_tokens)
        
        self._tokenizer = tokenizer_object


