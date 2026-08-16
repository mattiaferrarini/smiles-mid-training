from .element_tokenizer import ElementTokenizer
from .bpe_tokenizer import BPETokenizer
from .wordpiece_tokenizer import WordPieceTokenizer
from .spe_tokenizer import SPETokenizer
from .smiles_wordpiece_tokenizer import SmilesWordPieceTokenizer
from .scored_spe_tokenizer import ScoredSPETokenizer
from .kmer_tokenizer import KmerTokenizer
from .character_tokenizer import CharacterTokenizer

TOKENIZER_CLASSES = {
    "bpe": BPETokenizer,
    "character": CharacterTokenizer,
    "element": ElementTokenizer,
    "kmer": KmerTokenizer,
    "scored_spe": ScoredSPETokenizer,
    "spe": SPETokenizer,
    "swp": SmilesWordPieceTokenizer,
    "wordpiece": WordPieceTokenizer,
}
