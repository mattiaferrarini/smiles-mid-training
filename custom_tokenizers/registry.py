from .element_tokenizer import ElementTokenizer
from .smiles_bpe_tokenizer import SmilesBpeTokenizer
from .smiles_wp_tokenizer import SmilesWPTokenizer
from .ape_hf_tokenizer import APEHFTokenizer
from .ape_wp_hf_tokenizer import APEWPHFTokenizer
from .chem_ape import ChemAPETokenizer
from .kmer_tokenizer import KmerTokenizer
from .character_tokenizer import CharacterTokenizer

TOKENIZER_CLASSES = {
    "bpe": SmilesBpeTokenizer,
    "character": CharacterTokenizer,
    "element": ElementTokenizer,
    "kmer": KmerTokenizer,
    "scored_spe": ChemAPETokenizer,
    "spe": APEHFTokenizer,
    "swp": APEWPHFTokenizer,
    "wordpiece": SmilesWPTokenizer,
}
