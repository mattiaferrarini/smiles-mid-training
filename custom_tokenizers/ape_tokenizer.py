'''
Adapted and simplified from https://github.com/mikemayuare/apetokenizer.
'''

from collections import defaultdict
import re
import json
import os
from pathlib import Path
from transformers import PreTrainedTokenizerBase
from typing import Optional, List, Dict, Any
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.helpers as helpers

class APETokenizer(PreTrainedTokenizerBase):
    def __init__(self, 
        vocab_file=None,
        unk_token="[UNK]",
        pad_token="[PAD]", 
        bos_token="[BOS]", 
        eos_token="[EOS]",
        config=None,
        **kwargs
    ):
        self.max_vocab_size = config["tokenizer"]["params"].get("max_vocab_size", 20000) if config else 20000
        self.min_freq_for_merge = config["tokenizer"]["params"].get("min_freq_for_merge", 2) if config else 2
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
            **kwargs
        )
        
        # Store special tokens for compatibility
        self.special_tokens = {
            unk_token: self.vocab[unk_token],
            pad_token: self.vocab[pad_token],
            bos_token: self.vocab[bos_token],
            eos_token: self.vocab[eos_token],
        }
        
        # Alias vocabulary to vocab for compatibility
        # Alias vocabulary to vocab for compatibility
        self.vocabulary = self.vocab

    @property
    def vocab_size(self) -> int:
        """Required by PreTrainedTokenizerBase"""
        return len(self.vocab)

    @property
    def unk_token_id(self):
        return self.vocab.get(self.unk_token, 0)

    def _encode_plus(
        self, 
        text, 
        text_pair=None, 
        add_special_tokens=True,
        padding_strategy="do_not_pad", 
        truncation_strategy="do_not_truncate", 
        max_length=None, 
        is_split_into_words=False, 
        **kwargs
    ) -> dict:
        """Required by PreTrainedTokenizerBase"""
        input_ids = self.encode(text)
        
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids)
        }

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        # Use _encode_plus for compatibility
        return self._encode_plus(text, **kwargs)

    def __len__(self):
        return len(self.vocab)

    def _tokenize(self, text):
        """Tokenize method for compatibility - just returns pre-tokenized result"""
        return self.pre_tokenize(text)

    def _convert_token_to_id(self, token):
        """Required by PreTrainedTokenizerBase"""
        return self.vocab.get(token, self.vocab.get(self.unk_token, 0))
    
    def _convert_id_to_token(self, index):
        """Required by PreTrainedTokenizerBase"""
        return self.decoder.get(index, self.unk_token)

    def pre_tokenize(self, molecule):
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        words = re.findall(pattern, molecule)
        return words

    def score_item(self, item):
        """Score a pair item for ranking. Can be overridden by subclasses."""
        return item[1]  # Default: just return frequency
    
    def get_most_common_pair(self):
        """Get the most frequent pair from pre-computed pair counts"""
        if not self.pair_counts:
            return (None, None), 0
        
        most_common_pair, freq = max(
            self.pair_counts.items(), key=self.score_item, default=((None, None), 0)
        )
        return most_common_pair, freq
    
    def compute_pair_counts(self, words):
        """Initial computation of all pair frequencies"""
        pair_counts = defaultdict(int)
        for i in range(len(words) - 1):
            pair = (words[i], words[i + 1])
            pair_counts[pair] += 1
        return pair_counts
    
    def update_pair_counts(self, words, pair_to_merge, merged_word):
        """Incrementally update pair counts after a merge"""
        # Remove old pairs and add new pairs around merge positions
        new_pairs = defaultdict(int)
        old_pairs = defaultdict(int)
        
        i = 0
        while i < len(words) - 1:
            if words[i] == pair_to_merge[0] and words[i + 1] == pair_to_merge[1]:
                # This pair will be merged
                # Remove the pair being merged
                old_pairs[pair_to_merge] += 1
                
                # Check for new pairs formed
                if i > 0:
                    # Left context changes
                    old_pairs[(words[i - 1], pair_to_merge[0])] += 1
                    new_pairs[(words[i - 1], merged_word)] += 1
                
                if i + 2 < len(words):
                    # Right context changes
                    old_pairs[(pair_to_merge[1], words[i + 2])] += 1
                    new_pairs[(merged_word, words[i + 2])] += 1
                
                i += 2  # Skip the merged pair
            else:
                i += 1
        
        # Apply updates to pair_counts
        for pair, count in old_pairs.items():
            self.pair_counts[pair] -= count
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]
        
        for pair, count in new_pairs.items():
            self.pair_counts[pair] += count

    def _train(self, corpus, max_vocab_size: int = None, min_freq_for_merge: int = None):
        """Internal training method - optimized version"""
        if max_vocab_size is None:
            max_vocab_size = self.max_vocab_size
        if min_freq_for_merge is None:
            min_freq_for_merge = self.min_freq_for_merge
            
        # Preprocessing: Tokenize and count word frequencies upfront
        print(f"Starting tokenization training on {len(corpus)} sequences...")
        print("Pretokenizing corpus...", end="\r")
        words = [word for sentence in corpus for word in self.pre_tokenize(sentence)]
        vocabulary_frequency = defaultdict(int)
        for word in words:
            vocabulary_frequency[word] += 1
        print(
            f"Pretokenization complete! Found {len(vocabulary_frequency)} initial tokens from {len(words)} total tokens"
        )

        # Initial pair count computation (done once)
        print("Computing initial pair frequencies...", end="\r")
        self.pair_counts = self.compute_pair_counts(words)
        print(f"Initial pair count complete! Found {len(self.pair_counts)} unique pairs")

        merged_counter = len(vocabulary_frequency) + 1
        iteration = 0

        print(f"\nStarting merge iterations (target vocab size: {max_vocab_size}, min frequency: {min_freq_for_merge})...")
        
        while True:
            iteration += 1
            
            if len(vocabulary_frequency) > self.max_vocab_size:
                print(f"\n✓ Max vocabulary size reached: {len(vocabulary_frequency)} tokens")
                break

            most_common_pair, freq = self.get_most_common_pair()
            if freq < self.min_freq_for_merge:
                print(f"\n✓ Stopping: pair frequency ({freq}) below minimum threshold ({self.min_freq_for_merge})")
                break

            merged_word = "".join(most_common_pair)
            if merged_word not in vocabulary_frequency.keys():
                progress_pct = round(merged_counter / max_vocab_size * 100, 2)
                print(
                    f"Iteration {iteration}: Merging '{most_common_pair[0]}' + '{most_common_pair[1]}' → '{merged_word}' "
                    f"(freq: {freq}) | Vocab: {merged_counter}/{max_vocab_size} ({progress_pct}%)"
                )
                merged_counter += 1
            merged_word_freq = vocabulary_frequency.get(merged_word, 0)
            vocabulary_frequency[merged_word] = merged_word_freq + freq

            # Update pair counts incrementally before merging words
            self.update_pair_counts(words, most_common_pair, merged_word)

            # Apply merge to words list
            new_words = []
            i = 0
            while i < len(words):
                # Look ahead to find pairs to merge
                if (
                    i < len(words) - 1
                    and words[i] == most_common_pair[0]
                    and words[i + 1] == most_common_pair[1]
                ):
                    new_words.append(merged_word)
                    i += 2  # Skip both elements of the pair
                else:
                    new_words.append(words[i])
                    i += 1

            words = new_words

        # Convert vocabulary_frequency to a regular dictionary for final output
        self.vocabulary_frequency = dict(vocabulary_frequency)
        
        # Build vocab from special tokens + learned vocabulary
        # Start from next available ID after special tokens
        max_special_id = max(self.special_tokens.values())
        self.vocab = {**self.special_tokens}
        for idx, word in enumerate(vocabulary_frequency.keys(), start=max_special_id + 1):
            if word not in self.vocab:  # Don't override special tokens
                self.vocab[word] = idx
        
        # Update vocabulary alias and decoder
        self.vocabulary = self.vocab
        self.decoder = {v: k for k, v in self.vocab.items()}
        
        print("\nTraining complete.")

        return None

    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path=None, save_vocabulary=False):
        """
        Wrapper method compatible with build_and_save_tokenizer.
        Accepts a list of strings (SMILES) and trains the APE tokenizer on them.
        
        Args:
            text: List of strings or single string to train on
            append_to_existing_vocabulary: Not used, kept for compatibility
            vocab_path: Not used, kept for compatibility  
            save_vocabulary: Not used, kept for compatibility
        """
        # Handle both single string and list of strings
        if isinstance(text, str):
            corpus = [text]
        else:
            corpus = text
            
        print(f"Training APE tokenizer on {len(corpus)} sequences...")
        self._train(corpus, max_vocab_size=self.max_vocab_size, min_freq_for_merge=self.min_freq_for_merge)
        
        return self.vocab

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):  # Single token
            return self.vocab.get(tokens, self.vocab.get(self.unk_token, 0))
        else:  # List of tokens
            return [
                self.vocab.get(token, self.vocab.get(self.unk_token, 0))
                for token in tokens
            ]


    def convert_ids_to_tokens(self, token_ids):
        # Map each token ID to its corresponding string token
        return [self.decoder.get(token_id, self.unk_token) for token_id in token_ids]

    def get_vocab(self):
        """Required by PreTrainedTokenizerBase"""
        return self.vocab.copy()

    def decode(self, token_ids):
        tokens = self.convert_ids_to_tokens(token_ids)
        return ''.join(tokens)

    def encode(self, text):
        # Initialize the list of encoded tokens
        encoded_tokens = []

        # Scan and tokenize the text based on the vocabulary
        i = 0
        while i < len(text):
            match = None
            # Check for the longest sequence in the vocabulary that matches the text
            for j in range(len(text), i, -1):
                possible_match = text[i:j]
                if possible_match in self.vocab:
                    match = possible_match
                    break
            if match:
                # Add the token's index to the encoded tokens
                encoded_tokens.append(self.vocab[match])
                i += len(match)  # Move past the matched text
            else:
                # If no match is found, use the unknown token and move one character forward
                encoded_tokens.append(self.vocab.get(self.unk_token, 0))
                i += 1

        return encoded_tokens

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """Required by PreTrainedTokenizerBase - save just the vocab file"""
        if filename_prefix:
            vocab_file = f"{filename_prefix}-vocab.json"
        else:
            vocab_file = "vocab.json"
            
        path = os.path.join(save_directory, vocab_file)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        return (path,)

    def save_pretrained(self, save_directory, **kwargs):
        """Save tokenizer in Huggingface-compatible format"""
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
            "model_type": "ape_tokenizer",
            "max_vocab_size": self.max_vocab_size,
            "min_freq_for_merge": self.min_freq_for_merge,
        }
        
        config_file = save_directory / "tokenizer_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        # Optionally save training state for further training
        if self.vocabulary_frequency:
            vocabulary_frequency_serializable = {str(k): v for k, v in self.vocabulary_frequency.items()}
            pair_counts_serializable = {str(k): v for k, v in self.pair_counts.items()}
            
            training_state = {
                "vocabulary_frequency": vocabulary_frequency_serializable,
                "pair_counts": pair_counts_serializable,
            }

            training_state_file = save_directory / "training_state.json"
            with open(training_state_file, "w", encoding="utf-8") as f:
                json.dump(training_state, f, ensure_ascii=False, indent=4)
        
        print(f"Tokenizer saved in {save_directory}")
        return str(save_directory / "vocab.json"), str(config_file)

    @classmethod
    def from_pretrained(cls, pretrained_directory, **kwargs):
        """Load tokenizer from Huggingface-compatible format"""
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
