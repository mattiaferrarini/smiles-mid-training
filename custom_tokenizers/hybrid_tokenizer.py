import os
import re
import json

from dotenv import load_dotenv
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers.tokenization_utils import AddedToken
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

class HybridTokenizer(PreTrainedTokenizerBase):
    vocab_files_names = {
        "chem_vocab_file": "chem_vocab.json"
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, base_tokenizer, chem_tokenizer, chem_start, chem_end, **kwargs):
        super().__init__(**kwargs)
        
        self.is_fast = False

        self.base_tokenizer = base_tokenizer
        self.chem_tokenizer = chem_tokenizer
        self.chem_start = chem_start
        self.chem_end = chem_end

        if base_tokenizer.pad_token:
            self.pad_token = base_tokenizer.pad_token
        if base_tokenizer.eos_token:
            self.eos_token = base_tokenizer.eos_token
        if base_tokenizer.bos_token:
            self.bos_token = base_tokenizer.bos_token
        if base_tokenizer.unk_token:
            self.unk_token = base_tokenizer.unk_token
        
        if hasattr(base_tokenizer, 'chat_template') and base_tokenizer.chat_template:
            self.chat_template = base_tokenizer.chat_template

        print(f"Base tokenizer type: {type(base_tokenizer)}")
        print("Base tokenizer vocab size:", len(self.base_tokenizer))
        
        self.base_tokenizer.add_special_tokens({
            'additional_special_tokens': [self.chem_start, self.chem_end]
        })
        print("Base tokenizer vocab size after special tokens:", len(self.base_tokenizer))

        self.chem_start_id = self.base_tokenizer.convert_tokens_to_ids(self.chem_start)
        self.chem_end_id = self.base_tokenizer.convert_tokens_to_ids(self.chem_end)

        print(f"Chem Start ID: {self.chem_start_id}, Chem End ID: {self.chem_end_id}")

        self.chem_vocab, self.chem_ids_map = self.create_chem_vocab()
        self.id_to_chem_token = {v: k for k, v in self.chem_vocab.items()}

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        return self.base_tokenizer._add_tokens(new_tokens, special_tokens=special_tokens)

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        return self.base_tokenizer.added_tokens_encoder

    @property
    def added_tokens_decoder(self) -> Dict[int, Any]:
        return self.base_tokenizer.added_tokens_decoder

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        base_files = self.base_tokenizer.save_vocabulary(save_directory, filename_prefix)
        
        if filename_prefix:
            chem_file = os.path.join(save_directory, f"{filename_prefix}-chem_vocab.json")
        else:
            chem_file = os.path.join(save_directory, "chem_vocab.json")
            
        with open(chem_file, 'w', encoding='utf-8') as f:
            json.dump(self.chem_vocab, f, ensure_ascii=False, indent=2)
            
        return base_files + (chem_file,)

    def create_chem_vocab(self):
        base_vocab = self.base_tokenizer.get_vocab().copy()
        chem_vocab = self.chem_tokenizer.get_vocab().copy()
        chem_ids_map = {}
        
        next_chem_id = max(base_vocab.values()) + 1
        print("First available ID for chemical tokens:", next_chem_id)
        
        for token, idx in chem_vocab.items():
            chem_vocab[token] = next_chem_id
            chem_ids_map[idx] = next_chem_id
            next_chem_id += 1

        print("Chemical vocabulary size:", len(chem_vocab))
        return chem_vocab, chem_ids_map

    def get_chem_vocab(self):
        return self.chem_vocab

    def get_chem_ids_map(self):
        return self.chem_ids_map

    @property
    def vocab_size(self):
        return len(self.base_tokenizer) + len(self.chem_ids_map)
    
    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        vocab = self.base_tokenizer.get_vocab().copy()
        vocab.update(self.chem_vocab)
        return vocab

    def _convert_token_to_id(self, token: str) -> int:
        base_id = self.base_tokenizer.convert_tokens_to_ids(token)
        
        if base_id != self.base_tokenizer.unk_token_id:
            return base_id
            
        if token in self.chem_vocab:
            return self.chem_vocab[token]
            
        return self.base_tokenizer.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        if index < len(self.base_tokenizer):
            return self.base_tokenizer.convert_ids_to_tokens(index)
        
        if index in self.id_to_chem_token:
            return self.id_to_chem_token[index]
            
        return self.base_tokenizer.unk_token

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens=False) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(i) for i in ids]

    def _tokenize_single_text(self, text):
        segments = re.split(f"({re.escape(self.chem_start)}.*?{re.escape(self.chem_end)})", text)
        input_ids = []

        for i, segment in enumerate(segments):
            if not segment:
                continue
            
            if segment.startswith(self.chem_start) and segment.endswith(self.chem_end):
                chem_content = segment[len(self.chem_start):-len(self.chem_end)]
                
                chem_token_results = self.chem_tokenizer(chem_content)
                chem_ids = chem_token_results["input_ids"]
                
                input_ids.append(self.chem_start_id)
                for id in chem_ids:
                    if id in self.chem_ids_map:
                         input_ids.append(self.chem_ids_map[id])
                    else:
                        input_ids.append(self.base_tokenizer.unk_token_id)
                input_ids.append(self.chem_end_id)
                
            else:
                base_ids = self.base_tokenizer(segment)["input_ids"]
                if i > 0 and len(base_ids) > 0 and base_ids[0] == self.base_tokenizer.bos_token_id:
                    base_ids = base_ids[1:]
                
                input_ids.extend(base_ids)
        
        return input_ids

    def __call__(self, text, **kwargs):
        if text is None:
            return None
        
        if isinstance(text, str):
            text_list = [text]
        elif isinstance(text, list):
            text_list = text
        else:
            raise ValueError(f"Expected string or list of strings, got {type(text)}")

        padding = kwargs.get("padding", False)
        return_tensors = kwargs.get("return_tensors", None)

        max_length = kwargs.get("max_length", None)
        stride = kwargs.get("stride", 0)
        return_overflowing_tokens = kwargs.get("return_overflowing_tokens", False)
        truncation = kwargs.get("truncation", False)
        
        batch_input_ids = []
        overflow_mapping = []

        for sample_idx, t in enumerate(text_list):
            full_ids = self._tokenize_single_text(t)
            
            if max_length and (truncation or return_overflowing_tokens):
                step = max_length - stride if stride > 0 else max_length
                
                if return_overflowing_tokens:
                    current_chunks = 0
                    for start_idx in range(0, len(full_ids), step):
                        end_idx = min(start_idx + max_length, len(full_ids))
                        chunk = full_ids[start_idx:end_idx]
                        
                        if chunk or (start_idx == 0 and len(full_ids) == 0):
                            batch_input_ids.append(chunk)
                            overflow_mapping.append(sample_idx)
                            current_chunks += 1
                        
                        if end_idx == len(full_ids):
                            break
                    
                    if current_chunks == 0:
                        batch_input_ids.append(full_ids)
                        overflow_mapping.append(sample_idx)

                elif truncation:
                    batch_input_ids.append(full_ids[:max_length])
            else:
                batch_input_ids.append(full_ids)

        if padding and batch_input_ids:
            max_len_in_batch = max(len(ids) for ids in batch_input_ids)
            pad_id = self.pad_token_id if self.pad_token_id is not None else self.eos_token_id

            for i in range(len(batch_input_ids)):
                diff = max_len_in_batch - len(batch_input_ids[i])
                if diff > 0:
                    batch_input_ids[i] = batch_input_ids[i] + [pad_id] * diff

        pad_id = self.pad_token_id if self.pad_token_id is not None else self.eos_token_id
        attention_mask = []
        for ids in batch_input_ids:
            mask = [1 if token != pad_id else 0 for token in ids]
            attention_mask.append(mask)
        # attention_mask = [[1] * len(ids) for ids in batch_input_ids]

        output = {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask
        }
        
        if return_overflowing_tokens:
            output["overflow_to_sample_mapping"] = overflow_mapping

        return BatchEncoding(output, tensor_type=return_tensors)

    def decode(self, token_ids, **kwargs):
        tokens = []
        reverse_chem_map = {v: k for k, v in self.chem_ids_map.items()}
        
        if isinstance(token_ids, int):
            token_ids = [token_ids]
            
        for token_id in token_ids:
            if token_id == self.chem_start_id:
                tokens.append(self.chem_start)
            elif token_id == self.chem_end_id:
                tokens.append(self.chem_end)
            elif token_id in reverse_chem_map:
                original_chem_id = reverse_chem_map[token_id]
                chem_token = self.chem_tokenizer.decode([original_chem_id])
                tokens.append(chem_token)
            else:
                base_token = self.base_tokenizer.decode([token_id])
                tokens.append(base_token)
        
        return ''.join(tokens)
