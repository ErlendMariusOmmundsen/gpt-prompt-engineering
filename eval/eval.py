from rouge_score import rouge_scorer
from pyrouge import Rouge155
import language_tool_python
from nltk.tokenize import word_tokenize, sent_tokenize
from bert_score import score
import matplotlib.pyplot as plt


class eval:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rogue2", "rougeL"], use_stemmer=True)
    langtool = language_tool_python.LanguageToolPublicAPI(
        "en-US"
    )  # use a remote server (automatically set up), language English

    def __init__(self):
        pass

    def rogue(self, candidate, reference):
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

    def error_count_score(self, sent):
        check = self.langtool.check(sent)
        numTokens = len(word_tokenize(sent))
        numErrors = len(check)
        return 1 - float(numErrors) / float(numTokens)

    def avg_error_count_score(self, text):
        sentences = sent_tokenize(text)
        total = 0
        for sent in sentences:
            total += self.error_count_score(sent)
        return total / len(sentences)

    def bert_score(self, candidates, references):
        # score inputs: list of candidate sentences, list of reference sentences
        # score outputs: precision, recall, f1 tensors. Same number of elements as input

        p, r, f1 = score(candidates, references, lang="en", verbose=True)

        # get mean:
        mean = f1.mean()

        return p, r, f1, mean

    def textual_entailment(self):
        pass
