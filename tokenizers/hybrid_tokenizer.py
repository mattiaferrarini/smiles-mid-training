from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv
import re
from element_tokenizer import ElementTokenizer

class HybridTokenizer(PreTrainedTokenizerBase):
    def __init__(self, base_tokenizer, chem_tokenizer, chem_start, chem_end):
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.chem_tokenizer = chem_tokenizer
        self.chem_start = chem_start
        self.chem_end = chem_end

        print("Base tokenizer vocab size:", len(self.base_tokenizer))
        # Add special tokens for chemical segments
        self.base_tokenizer.add_special_tokens({
            'additional_special_tokens': [self.chem_start, self.chem_end]
        })
        print("Base tokenizer vocab size after special tokens:", len(self.base_tokenizer))

        # Ids for special tokens delimiting chemical segments
        self.chem_start_id = self.base_tokenizer.convert_tokens_to_ids(self.chem_start)
        self.chem_end_id = self.base_tokenizer.convert_tokens_to_ids(self.chem_end)

        print(self.chem_start_id, self.chem_end_id)

        # Create chemical vocabulary and ID mapping
        self.chem_vocab, self.chem_ids_map = self.create_chem_vocab()

    def create_chem_vocab(self):
        base_vocab = self.base_tokenizer.get_vocab().copy()
        chem_vocab = self.chem_tokenizer.get_vocab().copy()
        chem_ids_map = {}
        next_chem_id = max(base_vocab.values()) + 1
        print("First available ID for chemical tokens:", next_chem_id)
        
        # Map chemical tokenizer IDs to new unique IDs
        for token, idx in chem_vocab.items():
            chem_vocab[token] = next_chem_id
            chem_ids_map[idx] = next_chem_id
            next_chem_id += 1

        print("Chemical vocabulary size:", len(chem_vocab))
        print("Chemical IDs map:", chem_ids_map)

        return chem_vocab, chem_ids_map

    def get_chem_vocab(self):
        return self.chem_vocab

    def get_chem_ids_map(self):
        return self.chem_ids_map

    def __len__(self):
        return len(self.base_tokenizer) + len(self.chem_ids_map)
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        tokens = []
        # Create reverse mapping from new Ids to original chem Ids
        reverse_chem_map = {v: k for k, v in self.chem_ids_map.items()}
        
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
    
    def __call__(self, text, **kwargs):
        segments = re.split(f"({re.escape(self.chem_start)}.*?{re.escape(self.chem_end)})", text)
        input_ids = []

        print("Segments after splitting:", segments)

        for i, segment in enumerate(segments):
            if not segment:  # Skip empty segments
                continue
            if segment.startswith(self.chem_start) and segment.endswith(self.chem_end):
                # Tokenize the chemical segment using the chemical tokenizer
                chem_token_results = self.chem_tokenizer(segment[len(self.chem_start):-len(self.chem_end)])
                print("Chemical tokenization results:", chem_token_results)
                chem_ids = chem_token_results["input_ids"]
                input_ids.append(self.chem_start_id)
                for id in chem_ids:
                    if id not in self.chem_ids_map:
                        raise ValueError(f"Token {id} not found in chemical vocabulary.")
                    input_ids.append(self.chem_ids_map[id])
                input_ids.append(self.chem_end_id)
            else:
                # Tokenize the non-chemical segment using the base tokenizer
                base_ids = self.base_tokenizer(segment)["input_ids"]
                # Remove <bos> token from all segments except the first one
                if i > 0 and len(base_ids) > 0 and base_ids[0] == self.base_tokenizer.bos_token_id:
                    base_ids = base_ids[1:]
                input_ids.extend(base_ids)

        return {"input_ids": input_ids}

if __name__ == "__main__":
    load_dotenv()
    
    base_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    print(base_tokenizer("Hello, world!"))
    print(type(base_tokenizer))

    test_string = "This is a test [CHEM]CH[/CHEM] with chemical formula."

    res = base_tokenizer(test_string)
    print("Base tokenizer result:", res)
    print("Base tokenizer decoded:", base_tokenizer.decode(res["input_ids"]))

    tokenizer = HybridTokenizer(base_tokenizer, ElementTokenizer(), "[CHEM]", "[/CHEM]")
    result = tokenizer(test_string)
    print("Tokenized IDs:", result)
    print("Decoded text:", tokenizer.decode(result["input_ids"]))