# Custom Tokenizers

This folder contains the implementations of various tokenizers studied for processing strings containing SMILES formulas, as well as scripts for building and assembling them.

## Tokenizer Implementations

The tokenizers are designed to handle chemical syntax and structure at different levels of granularity. They are structured to be compatible with Hugging Face's Base Tokenizer, in order to make easier handling both in the same sample. This operation is handled by the `HybridTokenizer`

### Rule-Based & Simple Tokenizers
- `ElementTokenizer`: Tokenizes SMILES strings into chemical elements and symbols.
    - Variants: `ElementAllParenthesisTokenizer`, `ElementAromaticsTokenizer`, `ElementNoParenthesisTokenizer`, `ElementRingsTokenizer` (handle specific SMILES syntax rules differently, following chemical rules and patterns).
- `KmerTokenizer`: Splits text into overlapping k-mers (n-grams).
- `CharacterTokenizer`: Tokenizes text into individual characters.

### Learned Tokenizers
These tokenizers learn a vocabulary from data using algorithms like BPE, WordPiece, or Atom Pair Encoding.
- `SmilesBpeTokenizer`: Byte-Pair Encoding trained specifically on SMILES.
- `SmilesWPTokenizer`: WordPiece tokenizer for SMILES.
- `APETokenizer`: Atom Pair Encoding tokenizer.
- `APEHFTokenizer`: Optimized APE tokenizer using Hugging Face's fast BPE implementation.
- `APEWPHFTokenizer`: Optimized APE tokenizer using Hugging Face's fast WordPiece implementation.
- `ChemAPETokenizer`: A variant of APE that incorporates chemical validity scores into the merging process.

### Hybrid Tokenizer
- `HybridTokenizer`: Combines a pre-trained base tokenizer (we use the pretrained Gemma 3 Tokenizer) with one of ours specialized chemical tokenizer. It switches between them based on special control tags (`[START_SMILES]` and `[END_SMILES]`).

## Core Scripts

- `build_tokenizer.py`: Script to train and build a tokenizer from a dataset based on a configuration file.
- `assemble_tokenizer.py`: Factory script to instantiate and assemble tokenizers that are then used in training or evaluation.
- `registry.py`: Central registry mapping string identifiers to tokenizer classes.

## Subdirectories

- `scorers/`: Contains scoring modules (`ChemScorer`) used by `ChemAPETokenizer` to evaluate the chemical validity of token merges.