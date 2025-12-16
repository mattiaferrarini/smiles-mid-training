try:
    from .ape_tokenizer import APETokenizer
except ImportError:
    from ape_tokenizer import APETokenizer

try:
    from .scorers.chem_scorer import ChemScorer
except ImportError:
    from scorers.chem_scorer import ChemScorer


class ChemAPETokenizer(APETokenizer):
    """
    Chemical APE tokenizer that uses chemical validity and patterns to score merges.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the ChemAPETokenizer.

        Args:
            *args: Variable length argument list passed to APETokenizer.
            **kwargs: Arbitrary keyword arguments passed to APETokenizer.
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
