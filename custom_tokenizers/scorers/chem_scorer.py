from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

class ChemScorer:
    def __init__(self, base_score=1.0, bonus_pattern=1.1, bonus_valid=1.2):
        self.BASE_SCORE = base_score
        self.BONUS_PATTERN = bonus_pattern
        self.BONUS_VALID = bonus_valid
        print(f"ChemScorer initialized with BASE_SCORE={self.BASE_SCORE}, BONUS_VALID={self.BONUS_VALID}, BONUS_PATTERN={self.BONUS_PATTERN}")

    def get_merge_multiplier(self, text):
        # Perfect molecule
        if Chem.MolFromSmiles(text, sanitize=True):
            return self.BONUS_VALID

        # Valid substructure or pattern
        if Chem.MolFromSmarts(text):
            return self.BONUS_PATTERN

        # Base case: return base score
        return self.BASE_SCORE

if __name__ == "__main__":
    scorer = ChemScorer()
    
    candidates = [
        ("C(=O)O", "Valid Acid (Functional Group)"),
        ("c1ccccc1", "Valid Benzene (Ring)"),
        ("OH", "Valid Alcohol"),
        ("cc", "Aromatic Pair (Valid Pattern)"),
        (")O", "Glue (Invalid but necessary)"),
        ("c1ccccc", "Open Ring (Glue)"),
        ("C((", "Open Branch (Glue)")
    ]

    print(f"{'Token':<10} | {'Mult':<5} | {'Note'}")
    print("-" * 40)
    for tok, note in candidates:
        print(f"{tok:<10} | {scorer.get_merge_multiplier(tok)}   | {note}")