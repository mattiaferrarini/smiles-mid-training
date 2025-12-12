import os
import json
import collections
import utils.helpers as helpers

from tqdm import tqdm
from typing import Optional
from transformers import PreTrainedTokenizer
from SmilesPE.pretokenizer import kmer_tokenizer

class KmerTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self, 
        vocab_file=None, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        bos_token="[BOS]", 
        eos_token="[EOS]", 
        ngram=4, 
        stride=1,
        max_vocab_size=None,
        **kwargs
    ):
        self.vocab = {}
        self.decoder = {}
        self.ngram = ngram
        self.stride = stride
        self.max_vocab_size = max_vocab_size
        
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
            ngram=ngram,
            stride=stride,
            max_vocab_size=max_vocab_size,
            **kwargs
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        return kmer_tokenizer(text, ngram=self.ngram, stride=self.stride)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if filename_prefix:
            vocab_file = f"{filename_prefix}-vocab.json"
        else:
            vocab_file = "vocab.json"
        path = os.path.join(save_directory, vocab_file)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)

    def get_vocab(self):
        return self.vocab
        
    def create_vocabulary(self, text_iterator, append_to_existing_vocabulary=False, vocab_path="../json/vocab_symbol_to_number.json", save_vocabulary=False):
        print(f"Counting k-mers (ngram={self.ngram}, limit={self.max_vocab_size})...")
        counter = collections.Counter()
        
        for i, item in enumerate(tqdm(text_iterator)):
            text = item["text"] if isinstance(item, dict) else item
            tokens = self._tokenize(text)
            counter.update(tokens)
            
        print(f"Total unique k-mers found: {len(counter)}")

        new_vocab = {
            self.unk_token: 0, 
            self.pad_token: 1, 
            self.bos_token: 2, 
            self.eos_token: 3
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
        
        print(f"Final vocab size: {len(self.vocab)}")
        
        if save_vocabulary and vocab_path:
             with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return self.vocab
        
    def _load_vocab_from_json(self, path, append_to_existing_vocabulary=False):
        return helpers._load_vocab_from_json(path, append_to_existing_vocabulary, self.vocab)
