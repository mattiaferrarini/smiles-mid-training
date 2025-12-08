'''
Adapted and simplified from https://github.com/mikemayuare/apetokenizer.
Parallelized for high-core-count clusters with deadlock prevention.
'''

from collections import defaultdict
import re
import json
import os
from pathlib import Path
from transformers import PreTrainedTokenizerBase
from typing import Optional, List, Dict, Tuple
import multiprocessing
from functools import partial
import sys
from datetime import datetime

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils.helpers as helpers

# -----------------------------------------------------------------------------
# GLOBAL WORKER FUNCTIONS
# -----------------------------------------------------------------------------

def _worker_pre_tokenize(chunk: List[str], tokenizer) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Worker: 
    1. Tokenizes a chunk of text using the tokenizer instance.
    2. Counts the initial vocabulary frequency locally.
    """
    tokenized_chunk = []
    local_freq = defaultdict(int)
    
    for text in chunk:
        tokens = tokenizer.pre_tokenize(text)
        tokenized_chunk.append(tokens)
        for token in tokens:
            local_freq[token] += 1
            
    return tokenized_chunk, dict(local_freq)

def _worker_count_pairs(chunk: List[List[str]]) -> Dict[Tuple[str, str], int]:
    """
    Worker: Counts adjacent pairs in a chunk.
    """
    local_counts = defaultdict(int)
    for sequence in chunk:
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            local_counts[pair] += 1
    return dict(local_counts)

def _worker_merge_pair(chunk: List[List[str]], target_pair: Tuple[str, str], merged_token: str) -> List[List[str]]:
    """
    Worker: Replaces all instances of `target_pair` with `merged_token`.
    """
    new_chunk = []
    p0, p1 = target_pair
    
    for sequence in chunk:
        new_sequence = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == p0 and sequence[i+1] == p1:
                new_sequence.append(merged_token)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        new_chunk.append(new_sequence)
    return new_chunk

# -----------------------------------------------------------------------------
# MAIN CLASS
# -----------------------------------------------------------------------------

class ParallelAPETokenizer(PreTrainedTokenizerBase):
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
        self.min_freq_for_merge = config["tokenizer"]["params"].get("min_freq_for_merge", 0) if config else 0
        self.vocabulary_frequency = defaultdict(int)
        self.pair_counts = defaultdict(int)
        
        # Default regex pattern
        self.regex_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        
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
        
        self.special_tokens = {
            unk_token: self.vocab[unk_token],
            pad_token: self.vocab[pad_token],
            bos_token: self.vocab[bos_token],
            eos_token: self.vocab[eos_token],
        }
        
        self.vocabulary = self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def unk_token_id(self):
        return self.vocab.get(self.unk_token, 0)

    def _encode_plus(self, text, text_pair=None, **kwargs) -> dict:
        input_ids = self.encode(text)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def __call__(self, text, **kwargs):
        return self._encode_plus(text, **kwargs)

    def __len__(self):
        return len(self.vocab)

    def _tokenize(self, text):
        return self.pre_tokenize(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token, 0))
    
    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    def pre_tokenize(self, molecule: str) -> List[str]:
        """
        Splits a molecule string into initial tokens.
        Override this to change initial splitting logic.
        """
        words = re.findall(self.regex_pattern, molecule)
        return words

    def _aggregate_pair_counts(self, tokenized_chunks, pool=None):
        """
        Internal method: Aggregates pair counts from workers.
        """
        self.pair_counts.clear()

        if pool:
            # Parallel Counting
            # We use map here because we need all results before proceeding
            results = pool.map(_worker_count_pairs, tokenized_chunks)
            for res in results:
                for pair, count in res.items():
                    self.pair_counts[pair] += count
        else:
            # Sequential Fallback
            for chunk in tokenized_chunks:
                counts = _worker_count_pairs(chunk)
                for pair, count in counts.items():
                    self.pair_counts[pair] += count

    def get_most_common_pair(self):
        """
        Selects the best pair from self.pair_counts.
        Override this to change the selection strategy.
        """
        most_common_pair, freq = max(
            self.pair_counts.items(), key=lambda x: x[1], default=((None, None), 0)
        )
        return most_common_pair, freq

    def _train(self, corpus, max_vocab_size: int = None, min_freq_for_merge: int = None):
        start_time = datetime.now()
        print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if max_vocab_size is None:
            max_vocab_size = self.max_vocab_size
        if min_freq_for_merge is None:
            min_freq_for_merge = self.min_freq_for_merge
            
        # Safer multiprocessing context for clusters
        try:
            ctx = multiprocessing.get_context("forkserver")
        except ValueError:
            # Fallback for systems that don't support forkserver (e.g. Windows)
            ctx = multiprocessing.get_context("spawn")

        num_workers = min(multiprocessing.cpu_count() - 1, 128)
        num_workers = max(1, num_workers)
        
        # 1. Chunk the corpus
        print(f"Dividing corpus of {len(corpus)} sequences for {num_workers} workers...")
        chunk_size = max(1, len(corpus) // num_workers)
        text_chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)]
        total_chunks = len(text_chunks)
        
        print(f"Created {total_chunks} chunks. Initializing worker pool...")
        print(f"0/{total_chunks} chunks pre-tokenized...", end="\r", flush=True)

        tokenized_chunks = []
        vocabulary_frequency = defaultdict(int)
        
        # 2. Parallel Pre-tokenization
        with ctx.Pool(num_workers) as pool:
            pre_tok_func = partial(_worker_pre_tokenize, tokenizer=self)
            
            # chunksize=1 is important here because our 'items' are already large chunks
            for i, (chunk_result, chunk_vocab) in enumerate(pool.imap(pre_tok_func, text_chunks, chunksize=1)):
                tokenized_chunks.append(chunk_result)
                
                for token, count in chunk_vocab.items():
                    vocabulary_frequency[token] += count
                    
                if (i + 1) % max(1, total_chunks // 20) == 0 or (i + 1) == total_chunks:
                    print(f"{i + 1}/{total_chunks} chunks pre-tokenized...", end="\r", flush=True)

            print(f"\nPre-tokenization complete! Found {len(vocabulary_frequency)} initial tokens.")

            merged_counter = len(vocabulary_frequency) + 1
            iteration = 0

            print(f"Starting merge iterations...")
            
            while True:
                iteration += 1
                
                if len(vocabulary_frequency) > max_vocab_size:
                    print(f"\n✓ Max vocabulary size reached.")
                    break

                # 3. Aggregation
                self._aggregate_pair_counts(tokenized_chunks, pool=pool)

                # 4. Selection
                most_common_pair, freq = self.get_most_common_pair()
                
                if freq < min_freq_for_merge:
                    print(f"\n✓ Stopping: pair frequency ({freq}) below threshold.")
                    break

                merged_word = "".join(most_common_pair)
                
                if merged_word not in vocabulary_frequency:
                    progress_pct = round(merged_counter / max_vocab_size * 100, 2)
                    print(
                        f"Iter {iteration}: Merging {most_common_pair} -> {merged_word} "
                        f"(freq: {freq}) | Vocab: {merged_counter}/{max_vocab_size} ({progress_pct}%)"
                    )
                    merged_counter += 1
                
                vocabulary_frequency[merged_word] = vocabulary_frequency.get(merged_word, 0) + freq

                # 5. Parallel Merge
                merge_func = partial(_worker_merge_pair, target_pair=most_common_pair, merged_token=merged_word)
                tokenized_chunks = pool.map(merge_func, tokenized_chunks)

        self.vocabulary_frequency = dict(vocabulary_frequency)
        
        # Build final vocab
        max_special_id = max(self.special_tokens.values())
        self.vocab = {**self.special_tokens}
        for idx, word in enumerate(vocabulary_frequency.keys(), start=max_special_id + 1):
            if word not in self.vocab:
                self.vocab[word] = idx
        
        self.vocabulary = self.vocab
        self.decoder = {v: k for k, v in self.vocab.items()}
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nTraining complete at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        return None

    def create_vocabulary(self, text, append_to_existing_vocabulary=False, vocab_path=None, save_vocabulary=False):
        if isinstance(text, str):
            corpus = [text]
        else:
            corpus = text
        self._train(corpus, max_vocab_size=self.max_vocab_size, min_freq_for_merge=self.min_freq_for_merge)
        return self.vocab

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.vocab.get(self.unk_token, 0))
        else:
            return [self.vocab.get(token, self.vocab.get(self.unk_token, 0)) for token in tokens]

    def convert_ids_to_tokens(self, token_ids):
        return [self.decoder.get(token_id, self.unk_token) for token_id in token_ids]

    def get_vocab(self):
        return self.vocab.copy()

    def decode(self, token_ids):
        tokens = self.convert_ids_to_tokens(token_ids)
        return ''.join(tokens)

    def encode(self, text):
        encoded_tokens = []
        i = 0
        while i < len(text):
            match = None
            for j in range(len(text), i, -1):
                possible_match = text[i:j]
                if possible_match in self.vocab:
                    match = possible_match
                    break
            if match:
                encoded_tokens.append(self.vocab[match])
                i += len(match)
            else:
                encoded_tokens.append(self.vocab.get(self.unk_token, 0))
                i += 1
        return encoded_tokens

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        if filename_prefix:
            vocab_file = f"{filename_prefix}-vocab.json"
        else:
            vocab_file = "vocab.json"
        path = os.path.join(save_directory, vocab_file)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)

    def save_pretrained(self, save_directory, **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.save_vocabulary(str(save_directory))
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
        with open(save_directory / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        
        if self.vocabulary_frequency:
            training_state = {
                "vocabulary_frequency": {str(k): v for k, v in self.vocabulary_frequency.items()},
            }
            with open(save_directory / "training_state.json", "w", encoding="utf-8") as f:
                json.dump(training_state, f, ensure_ascii=False, indent=4)
        
        print(f"Tokenizer saved in {save_directory}")
        return str(save_directory / "vocab.json"), str(save_directory / "tokenizer_config.json")

    @classmethod
    def from_pretrained(cls, pretrained_directory, **kwargs):
        pretrained_directory = Path(pretrained_directory)
        vocab_file = pretrained_directory / "vocab.json"
        config_file = pretrained_directory / "tokenizer_config.json"
        training_state_file = pretrained_directory / "training_state.json"

        if not vocab_file.is_file():
            raise FileNotFoundError(f"Vocabulary file {vocab_file} not found.")
            
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocabulary = json.load(f)

        config = {}
        if config_file.is_file():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

        tokenizer = cls(
            unk_token=config.get("unk_token", "[UNK]"),
            pad_token=config.get("pad_token", "[PAD]"),
            bos_token=config.get("bos_token", "[BOS]"),
            eos_token=config.get("eos_token", "[EOS]"),
            max_vocab_size=config.get("max_vocab_size", 5000),
            min_freq_for_merge=config.get("min_freq_for_merge", 2000),
        )
        
        tokenizer.vocab = vocabulary
        tokenizer.vocabulary = vocabulary
        tokenizer.decoder = {v: k for k, v in vocabulary.items()}

        if training_state_file.is_file():
            with open(training_state_file, "r", encoding="utf-8") as f:
                training_state = json.load(f)
            tokenizer.vocabulary_frequency = defaultdict(
                int, training_state.get("vocabulary_frequency", {})
            )

        return tokenizer
