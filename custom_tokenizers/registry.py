from .element_tokenizer import ElementTokenizer
from .smiles_bpe_tokenizer import SmilesBpeTokenizer
from .smiles_wp_tokenizer import SmilesWPTokenizer
from .ape_hf_tokenizer import APEHFTokenizer
from .ape_wp_hf_tokenizer import APEWPHFTokenizer
from .chem_ape import ChemAPETokenizer
from .kmer_tokenizer import KmerTokenizer
from .character_tokenizer import CharacterTokenizer

TOKENIZER_CLASSES = {
    "character": CharacterTokenizer,
    "element": ElementTokenizer,
    "kmer": KmerTokenizer,
    "smiles_bpe": SmilesBpeTokenizer,
    "smiles_wp": SmilesWPTokenizer,
    "ape_hf": APEHFTokenizer,
    "ape_wp_hf": APEWPHFTokenizer,
    "chem_ape": ChemAPETokenizer,
}