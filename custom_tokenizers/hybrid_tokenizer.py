from transformers import PreTrainedTokenizer
import re
import os
import json
from typing import List, Optional, Tuple

class HybridTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, base_tokenizer, chem_tokenizer, chem_start="[START_SMILES]", chem_end="[END_SMILES]", **kwargs):
        self.base_tokenizer = base_tokenizer
        self.chem_tokenizer = chem_tokenizer
        self.chem_start = chem_start
        self.chem_end = chem_end
        
        self.chem_ids_map = {}
        self.id_to_chem_token = {}

        self._build_chem_vocab()

        if chem_start not in base_tokenizer.get_vocab():
            base_tokenizer.add_special_tokens({'additional_special_tokens': [chem_start, chem_end]})

        self.chem_start_id = base_tokenizer.convert_tokens_to_ids(chem_start)
        self.chem_end_id = base_tokenizer.convert_tokens_to_ids(chem_end)

        super().__init__(
            bos_token=base_tokenizer.bos_token,
            eos_token=base_tokenizer.eos_token,
            unk_token=base_tokenizer.unk_token,
            sep_token=base_tokenizer.sep_token,
            pad_token=base_tokenizer.pad_token,
            cls_token=base_tokenizer.cls_token,
            mask_token=base_tokenizer.mask_token,
            additional_special_tokens=[chem_start, chem_end],
            model_max_length=base_tokenizer.model_max_length,
            **kwargs
        )

    def _build_chem_vocab(self):
        base_vocab = self.base_tokenizer.get_vocab()
        chem_vocab = self.chem_tokenizer.get_vocab()
        
        # Iniziamo dopo l'ID più alto del base, NON dopo la lunghezza
        next_chem_id = max(base_vocab.values()) + 1
        
        for token, _ in chem_vocab.items():
            self.chem_ids_map[token] = next_chem_id
            self.id_to_chem_token[next_chem_id] = token
            next_chem_id += 1
        self.chem_vocab = chem_vocab

    def get_chem_vocab(self): return self.chem_vocab
    def get_chem_ids_map(self): return self.chem_ids_map

    # --- FIX CUDA INDEX ASSERTION ERROR ---
    @property
    def vocab_size(self):
        # La dimensione DEVE essere basata sull'indice massimo, non sul conteggio.
        # Se ci sono buchi negli ID, len() < max_id, e questo causa il crash.
        max_base = max(self.base_tokenizer.get_vocab().values())
        max_chem = max(self.chem_ids_map.values()) if self.chem_ids_map else 0
        return max(max_base, max_chem) + 1

    def __len__(self):
        return self.vocab_size
    # --------------------------------------

    def get_vocab(self):
        v = self.base_tokenizer.get_vocab().copy()
        v.update(self.chem_ids_map)
        return v

    def _tokenize(self, text): return self.base_tokenizer.tokenize(text)
    
    def _convert_token_to_id(self, token):
        if token in self.chem_ids_map: return self.chem_ids_map[token]
        return self.base_tokenizer.convert_tokens_to_ids(token)
    
    def _convert_id_to_token(self, index):
        if index in self.id_to_chem_token: return self.id_to_chem_token[index]
        return self.base_tokenizer.convert_ids_to_tokens(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.exists(save_directory): os.makedirs(save_directory)
        vocab_filename = "vocab.json"
        if filename_prefix: vocab_filename = f"{filename_prefix}-{vocab_filename}"
        path = os.path.join(save_directory, vocab_filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False, indent=2)
        return (path,)

    def _tokenize_single_string(self, text):
        segments = re.split(f"({re.escape(self.chem_start)}.*?{re.escape(self.chem_end)})", text)
        input_ids = []
        for segment in segments:
            if not segment: continue
            if segment.startswith(self.chem_start) and segment.endswith(self.chem_end):
                content = segment[len(self.chem_start):-len(self.chem_end)]
                
                chem_encoded = self.chem_tokenizer(content, add_special_tokens=False)
                
                chem_ids = None
                if hasattr(chem_encoded, "input_ids"): chem_ids = chem_encoded.input_ids
                elif isinstance(chem_encoded, dict) and "input_ids" in chem_encoded: chem_ids = chem_encoded["input_ids"]
                else: chem_ids = chem_encoded 

                chem_tokens = self.chem_tokenizer.convert_ids_to_tokens(chem_ids)
                input_ids.append(self.chem_start_id)
                for t in chem_tokens:
                    input_ids.append(self.chem_ids_map.get(t, self.base_tokenizer.unk_token_id))
                input_ids.append(self.chem_end_id)
            else:
                input_ids.extend(self.base_tokenizer(segment, add_special_tokens=False)["input_ids"])
        return input_ids
    
    def __call__(self, text, **kwargs):
        is_batched = isinstance(text, (list, tuple))
        texts = text if is_batched else [text]
        
        batch_ids = [self._tokenize_single_string(t) for t in texts]

        if kwargs.get("truncation", False) and kwargs.get("max_length"):
            max_len = kwargs["max_length"]
            batch_ids = [ids[:max_len] for ids in batch_ids]

        pad_kwargs = {
            "padding": kwargs.get("padding", False),
            "max_length": kwargs.get("max_length") if kwargs.get("padding") else None,
            "return_tensors": kwargs.get("return_tensors", None)
        }

        return self.pad(
            {"input_ids": batch_ids, "attention_mask": [[1]*len(i) for i in batch_ids]},
            **pad_kwargs
        )