try:
    from .ape_tokenizer import APETokenizer
except ImportError:
    from ape_tokenizer import APETokenizer

try:
    from .scorers.chem_scorer import ChemScorer
except ImportError:
    from scorers.chem_scorer import ChemScorer

ELEMENTS = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
            "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
            "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
            "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
            "Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
            "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
            "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]
    
ELEMENTS = sorted(ELEMENTS, key=lambda x: -len(x))  # longest first

ELEMENT_PATTERN = "|".join(ELEMENTS)  # NO parentheses
ATOM_LEVEL_PATTERN = r"\[|\]|" + ELEMENT_PATTERN + r"|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9]"


class ChemAPETokenizer(APETokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs.get("config")
        self.scorer = ChemScorer(
            base_score=config["tokenizer"]["params"].get("base_score", 1.0) if config else 1.0,
            bonus_pattern=config["tokenizer"]["params"].get("bonus_pattern", 1.1) if config else 1.1,
            bonus_valid=config["tokenizer"]["params"].get("bonus_valid", 1.2) if config else 1.2
        )
    ''' 
    def pre_tokenize(self, molecule):
        pattern = re.compile(ATOM_LEVEL_PATTERN)
        tokens = pattern.findall(molecule)
        return tokens
    '''
    def score_item(self, item):
        merged_word = "".join(item[0])
        multiplier = self.scorer.get_merge_multiplier(merged_word)
        freq = item[1]
        score = freq * multiplier
        return score

    
