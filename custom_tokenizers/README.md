# Custom Tokenizers

This folder contains the implementations of various tokenizers studied for processing strings containing SMILES formulas, as well as scripts for building and assembling them.
The code exposes a small registry used by the project to refer to tokenizer types from config files.

## Tokenizer Implementations

The tokenizers are designed to handle chemical syntax and structure at different levels of granularity. They are structured to be compatible with Hugging Face's Base Tokenizer, in order to make easier handling both in the same sample. This operation is handled by the `HybridTokenizer`

### Rule-Based & Simple Tokenizers
- `ElementTokenizer`: Tokenizes SMILES strings into chemical elements and symbols.
    - Variants: `ElementAllParenthesisTokenizer`, `ElementAromaticsTokenizer`, `ElementNoParenthesisTokenizer`, `ElementRingsTokenizer` (handle specific SMILES syntax rules differently, following chemical rules and patterns).
- `KmerTokenizer`: Splits text into overlapping k-mers (n-grams).
- `CharacterTokenizer`: Tokenizes text into individual characters.

### Learned Tokenizers
These tokenizers learn a vocabulary from data using algorithms like BPE, WordPiece, or Atom Pair Encoding.
- `BpeTokenizer`: Byte-Pair Encoding trained specifically on SMILES.
- `WordPieceTokenizer`: general WordPiece training that uses whitespace pre-tokenization.
- `SmilesWordPieceTokenizer`: WordPiece variants adapted to SMILES pre-tokenization.
- `SPETokenizer`: SentencePiece-based tokenizer
- `ScoredSPETokenizer`: A variant of SPE that incorporates chemical validity scores into the merging process.

## Subdirectories

- `scorers/`: Contains scoring modules (`ChemScorer`) used by `ChemAPETokenizer` to evaluate the chemical validity of token merges.

### Hybrid Tokenizer
- `HybridTokenizer`: Combines a pre-trained base tokenizer (we use the pretrained Gemma 3 Tokenizer) with one of ours specialized chemical tokenizer. It switches between them based on special control tags (`[START_SMILES]` and `[END_SMILES]`).

## Core Scripts

- `build_tokenizer.py`: Script to train and build a tokenizer from a dataset based on a configuration file.
- `assemble_tokenizer.py`: Factory script to instantiate and assemble tokenizers that are then used in training or evaluation.
- `registry.py`: Central registry mapping string identifiers to tokenizer classes.
 

## Canonical tokenizer keys (registry)
The canonical keys accepted by `build_tokenizer.py` and `assemble_tokenizer.py` are defined in `registry.py`. Use these keys in your config under `tokenizer.type` or `tokenizer.chem_type`:

- `bpe` → class `BPETokenizer` (file: `bpe_tokenizer.py`)
- `wordpiece` → class `WordPieceTokenizer` (file: `wordpiece_tokenizer.py`)
- `swp` → class `SmilesWordPieceTokenizer` (file: `smiles_wordpiece_tokenizer.py`)
- `spe` → class `SPETokenizer` (file: `spe_tokenizer.py`)
- `scored_spe` → class `ScoredSPETokenizer` (file: `scored_spe_tokenizer.py`)
- `character` → class `CharacterTokenizer` (file: `character_tokenizer.py`)
- `element` → class `ElementTokenizer` (file: `element_tokenizer.py`)
- `kmer` → class `KmerTokenizer` (file: `kmer_tokenizer.py`)

Notes:
- The `element` tokenizer has several variant implementations in this folder that apply slightly different tokenization rules (files: `elementallparenthesis_tokenizer.py`, `elementaromatics_tokenizer.py`, `elementnoparenthesis_tokenizer.py`, `elementrings_tokenizer.py`). The registry exposes the generic key `element` (class `ElementTokenizer`). You can instantiate other variants directly if needed.
