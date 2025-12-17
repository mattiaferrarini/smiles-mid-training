try:
    from .manual_spe_tokenizer import ManualSPETokenizer
except ImportError:
    from custom_tokenizers.manual_spe_tokenizer import ManualSPETokenizer

try:
    from .scorers.chem_scorer import ChemScorer
except ImportError:
    from scorers.chem_scorer import ChemScorer


class ScoredSPETokenizer(ManualSPETokenizer):
    """
    Modified SPE tokenizer to score merges based on chemical validity and patterns.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the ScoredSPE tokenizer.

        Args:
            *args: Variable length argument list passed to ManualSPE.
            **kwargs: Arbitrary keyword arguments passed to ManualSPE.
        """
        super().__init__(*args, **kwargs)
        config = kwargs.get("config")
        self.scorer = ChemScorer(
            base_score=(
                config["tokenizer"]["params"].get("base_score", 1.0) if config else 1.0
            ),
            bonus_pattern=(
                config["tokenizer"]["params"].get("bonus_pattern", 1.1)
                if config
                else 1.1
            ),
            bonus_valid=(
                config["tokenizer"]["params"].get("bonus_valid", 1.2) if config else 1.2
            ),
        )

    def score_item(self, item):
        """
        Scores a candidate merge item based on frequency and chemical validity.

        Args:
            item (tuple): A tuple containing the merge candidate and its frequency.

        Returns:
            float: The calculated score.
        """
        merged_word = "".join(item[0])
        multiplier = self.scorer.get_merge_multiplier(merged_word)
        freq = item[1]
        score = freq * multiplier
        return score
