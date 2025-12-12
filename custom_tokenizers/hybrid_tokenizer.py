import os
import re
import json

from typing import List, Union, Dict, Optional, Tuple, Any
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

class HybridTokenizer(PreTrainedTokenizerBase):
    def __init__(self, base_tokenizer, chem_tokenizer, chem_start, chem_end, **kwargs):
        def __init__(self, base_tokenizer, chem_tokenizer, chem_start, chem_end, **kwargs):
        # 1. Sync special tokens from base_tokenizer if not provided in kwargs
        # This prevents eos_token or pad_token from being None
        for token_attr in ["pad_token", "eos_token", "bos_token", "unk_token"]:
            if token_attr not in kwargs and hasattr(base_tokenizer, token_attr):
                token_val = getattr(base_tokenizer, token_attr)
                if token_val is not None:
                    kwargs[token_attr] = token_val

        # Initialize base class (handles special tokens logic)
        super().__init__(**kwargs)
        
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
        
        # Ensure base tokenizer has special tokens
        if chem_start not in self.base_tokenizer.get_vocab():
            self.base_tokenizer.add_special_tokens({
                'additional_special_tokens': [self.chem_start, self.chem_end]
            })
        print("Base tokenizer vocab size after special tokens:", len(self.base_tokenizer))

        # Ids for special tokens delimiting chemical segments
        self.chem_start_id = self.base_tokenizer.convert_tokens_to_ids(self.chem_start)
        self.chem_end_id = self.base_tokenizer.convert_tokens_to_ids(self.chem_end)
        print(f"Chem Start ID: {self.chem_start_id}, Chem End ID: {self.chem_end_id}")

        # Create chemical vocabulary and ID mapping
        self.chem_vocab, self.chem_ids_map = self.create_chem_vocab()
        
        # Create reverse map for faster decoding/conversion
        self.id_to_chem_map = {v: k for k, v in self.chem_ids_map.items()}

    def save_pretrained(self, save_directory, **kwargs):
        """
        Delegates saving to the base tokenizer. 
        """
        return self.base_tokenizer.save_pretrained(save_directory, **kwargs)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        base_files = self.base_tokenizer.save_vocabulary(save_directory, filename_prefix)
        
        if filename_prefix:
            chem_file = os.path.join(save_directory, f"{filename_prefix}-chem_vocab.json")
        else:
            chem_file = os.path.join(save_directory, "chem_vocab.json")
            
        with open(chem_file, 'w', encoding='utf-8') as f:
            json.dump(self.chem_vocab, f, ensure_ascii=False, indent=2)
            
        return base_files + (chem_file,)

    @property
    def added_tokens_decoder(self) -> Dict[int, Any]:
        """
        Required for handling special tokens during saving.
        """
        return self.base_tokenizer.added_tokens_decoder
    
    def create_chem_vocab(self):
        base_vocab = self.base_tokenizer.get_vocab().copy()
        chem_vocab = self.chem_tokenizer.get_vocab().copy()
        chem_ids_map = {}
        
        # Find the first available ID after the base vocabulary
        next_chem_id = max(base_vocab.values()) + 1
        print("First available ID for chemical tokens:", next_chem_id)
        
        # Map chemical tokenizer IDs to new unique IDs
        for token, idx in chem_vocab.items():
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
        """Returns the total size of the vocabulary"""
        return len(self.base_tokenizer) + len(self.chem_ids_map)

    def __len__(self):
        """Required for len(tokenizer) calls"""
        return self.vocab_size
    
    def get_vocab(self):
        """Required by PreTrainedTokenizerBase"""
        vocab = self.base_tokenizer.get_vocab().copy()
        # Add chem tokens with their new IDs
        reverse_chem_vocab = {v: k for k, v in self.chem_tokenizer.get_vocab().items()}
        for chem_id, new_id in self.chem_ids_map.items():
            token = reverse_chem_vocab.get(chem_id, f"[CHEM_{chem_id}]")
            # Avoid overwriting base tokens if they have the same string but different ID
            if token not in vocab:
                vocab[token] = new_id
        return vocab
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens=False):
        """Converts a single index or a list of indices to token(s)."""
        if isinstance(ids, int):
            return self._convert_id_to_token_single(ids, skip_special_tokens)
        return [self._convert_id_to_token_single(i, skip_special_tokens) for i in ids]

    def _convert_id_to_token_single(self, index: int, skip_special_tokens: bool) -> str:
        # 1. Check special hybrid tokens
        if index == self.chem_start_id:
            return "" if skip_special_tokens else self.chem_start
        if index == self.chem_end_id:
            return "" if skip_special_tokens else self.chem_end
            
        # 2. Check if it's a mapped chemical ID
        if index in self.id_to_chem_map:
            original_chem_id = self.id_to_chem_map[index]
            return self.chem_tokenizer.convert_ids_to_tokens(original_chem_id)
            
        # 3. Fallback to base tokenizer
        return self.base_tokenizer.convert_ids_to_tokens(index)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str], None]):
        """Converts a token string (or list of strings) to a single integer ID (or list of IDs)."""
        # FIX: Handle None input (which happens if pad_token is missing)
        if tokens is None:
            return None
        
        if isinstance(tokens, str):
            return self._convert_token_to_id_single(tokens)
        
        return [self._convert_token_to_id_single(t) for t in tokens]
    
    def _convert_token_to_id_single(self, token: str) -> int:
        # 1. Check special tokens
        if token == self.chem_start:
            return self.chem_start_id
        if token == self.chem_end:
            return self.chem_end_id
        
        # 2. Try base tokenizer
        base_id = self.base_tokenizer.convert_tokens_to_ids(token)
        if base_id != self.base_tokenizer.unk_token_id:
            return base_id
            
        # 3. If not in base, try chemical tokenizer
        chem_id = self.chem_tokenizer.convert_tokens_to_ids(token)
        if chem_id in self.chem_ids_map:
            return self.chem_ids_map[chem_id]
            
        # 4. Return UNK
        return self.base_tokenizer.unk_token_id
    
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
    
    def _tokenize_single_text(self, text):
        """Helper to tokenize a single string into a list of IDs"""
        segments = re.split(f"({re.escape(self.chem_start)}.*?{re.escape(self.chem_end)})", text)
        input_ids = []

        for i, segment in enumerate(segments):
            if not segment:  # Skip empty segments
                continue
            if segment.startswith(self.chem_start) and segment.endswith(self.chem_end):
                # Extract content between tags
                content = segment[len(self.chem_start):-len(self.chem_end)]
                
                # Tokenize the chemical segment
                chem_token_results = self.chem_tokenizer(content, add_special_tokens=False)
                
                # Handle different return types
                if isinstance(chem_token_results, (dict, BatchEncoding)):
                    chem_ids = chem_token_results["input_ids"]
                else:
                    chem_ids = chem_token_results
                
                input_ids.append(self.chem_start_id)
                for id in chem_ids:
                    if id in self.chem_ids_map:
                        input_ids.append(self.chem_ids_map[id])
                    else:
                        input_ids.append(self.base_tokenizer.unk_token_id)
                input_ids.append(self.chem_end_id)
            else:
                # Tokenize the non-chemical segment
                base_ids = self.base_tokenizer(segment, add_special_tokens=False)["input_ids"]
                if i > 0 and len(base_ids) > 0 and base_ids[0] == self.base_tokenizer.bos_token_id:
                    base_ids = base_ids[1:]
                input_ids.extend(base_ids)
        
        return input_ids

    ####
    def __call__(self, text: Union[str, List[str]], text_pair=None, **kwargs):

        if text is None:
            return None
        
        # 1. Normalize input to list
        if isinstance(text, str):
            text_list = [text]
        elif isinstance(text, (list, tuple)):
            text_list = list(text)
        else:
            raise ValueError(f"Expected string or list of strings, got {type(text)}")
        
        padding = kwargs.get("padding", False)
        return_tensors = kwargs.get("return_tensors", None)

        max_length = kwargs.get("max_length", None)
        stride = kwargs.get("stride", 0)
        return_overflowing_tokens = kwargs.get("return_overflowing_tokens", False)
        truncation = kwargs.get("truncation", False)

        # 2. Tokenize all texts fully
        batch_full_ids = [self._tokenize_single_text(t) for t in text_list]

        # 3. Apply truncation and windowing (stride)
        final_input_ids = []
        overflow_to_sample_mapping = []

        # Determine effective max length
        if max_length is None:
            max_length = self.model_max_length
            if max_length > 1_000_000: 
                max_length = 2048 

        for sample_idx, ids in enumerate(batch_full_ids):
            total_len = len(ids)
            
            if not truncation or total_len <= max_length:
                final_input_ids.append(ids)
                if return_overflowing_tokens:
                    overflow_to_sample_mapping.append(sample_idx)
                continue
            
            if not return_overflowing_tokens:
                final_input_ids.append(ids[:max_length])
            else:
                # Sliding window (stride)
                step = max_length - stride
                if step <= 0:
                    raise ValueError(f"Stride ({stride}) must be strictly less than max_length ({max_length})")
                
                for i in range(0, total_len, step):
                    window = ids[i : i + max_length]
                    final_input_ids.append(window)
                    overflow_to_sample_mapping.append(sample_idx)
                    if i + max_length >= total_len:
                        break
        
        # 4. Handle Padding
        if padding and final_input_ids:
            max_len_in_batch = max(len(x) for x in final_input_ids)
            pad_id = self.pad_token_id if self.pad_token_id is not None else self.eos_token_id
            
            for i in range(len(final_input_ids)):
                diff = max_len_in_batch - len(final_input_ids[i])
                if diff > 0:
                    final_input_ids[i] = final_input_ids[i] + [pad_id] * diff

        # 5. Create Attention Masks
        attention_mask = []
        pad_id = self.pad_token_id if self.pad_token_id is not None else self.eos_token_id

        for ids in final_input_ids:
            mask = [1 if token != pad_id else 0 for token in ids]
            attention_mask.append(mask)

        # 6. Construct Output
        data = {
            "input_ids": final_input_ids,
            "attention_mask": attention_mask
        }
        
        if return_overflowing_tokens:
            data["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        return BatchEncoding(data, tensor_type=return_tensors)
    