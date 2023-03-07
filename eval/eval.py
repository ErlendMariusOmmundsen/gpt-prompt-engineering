from rouge_score import rouge_scorer
from pyrouge import Rouge155


class eval:
    scorer = ""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rogue2", "rougeL"], use_stemmer=True
        )

    def rogue(candidate, reference):
        """
        Args:
          rouge_types: A list of rouge types to calculate.
          use_stemmer: Bool indicating whether Porter stemmer should be used to
            strip word suffixes to improve matching. This arg is used in the
            DefaultTokenizer, but other tokenizers might or might not choose to
            use this.
          split_summaries: whether to add newlines between sentences for rougeLsum
          tokenizer: Tokenizer object which has a tokenize() method.
        Returns:
          A dict mapping rouge types to Score tuples.
        """
        scores = self.scorer.score(candidate, reference)
        return scores
