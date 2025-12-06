from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv
import re
# from .element_tokenizer import ElementTokenizer # Removed dependency on ElementTokenizer

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
            # Avoid overwriting if token exists in base (optional, but safer to keep separate)
            # For BPE, tokens might overlap (e.g. "C"). We map EVERYTHING from chem tokenizer to new IDs
            # to ensure we use the chem-specific embedding/tokenization.
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
                # Use chem_tokenizer to decode this specific ID
                chem_token = self.chem_tokenizer.decode([original_chem_id])
                tokens.append(chem_token)
            else:
                base_token = self.base_tokenizer.decode([token_id])
                tokens.append(base_token)
        
        return ''.join(tokens)
    
    def __call__(self, text, **kwargs):
        segments = re.split(f"({re.escape(self.chem_start)}.*?{re.escape(self.chem_end)})", text)
        input_ids = []

        # print("Segments after splitting:", segments)

        for i, segment in enumerate(segments):
            if not segment:  # Skip empty segments
                continue
            if segment.startswith(self.chem_start) and segment.endswith(self.chem_end):
                # Extract content between tags
                content = segment[len(self.chem_start):-len(self.chem_end)]
                
                # Tokenize the chemical segment using the chemical tokenizer (BPE or Element)
                # Note: PreTrainedTokenizerFast returns a BatchEncoding, we need input_ids
                chem_token_results = self.chem_tokenizer(content, add_special_tokens=False)
                chem_ids = chem_token_results["input_ids"]
                
                input_ids.append(self.chem_start_id)
                for id in chem_ids:
                    if id not in self.chem_ids_map:
                        # This should theoretically not happen if vocab is mapped correctly
                        # But if BPE produces a new token not in initial vocab (unlikely for static BPE), raise error
                        raise ValueError(f"Token {id} not found in chemical vocabulary map.")
                    input_ids.append(self.chem_ids_map[id])
                input_ids.append(self.chem_end_id)
            else:
                # Tokenize the non-chemical segment using the base tokenizer
                base_ids = self.base_tokenizer(segment, add_special_tokens=False)["input_ids"]
                input_ids.extend(base_ids)
        
        # Add BOS/EOS if needed (usually handled by base tokenizer, but we split it manually)
        
        return {"input_ids": input_ids}

if __name__ == "__main__":
    load_dotenv()
    
    base_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    
    # Example: Load your custom BPE tokenizer here
    # chem_tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_tokenizers/smiles_bpe/tokenizer.json")
    # For now, using base as placeholder or ElementTokenizer if available
    from .element_tokenizer import ElementTokenizer
    chem_tokenizer = ElementTokenizer()

    test_string = "This is a test [START_SMILES]CH[END_SMILES] with chemical formula."

    tokenizer = HybridTokenizer(base_tokenizer, chem_tokenizer, "[START_SMILES]", "[END_SMILES]")
    result = tokenizer(test_string)
    print("Tokenized IDs:", result)
    print("Decoded text:", tokenizer.decode(result["input_ids"]))