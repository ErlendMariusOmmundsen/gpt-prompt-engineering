import language_tool_python
from nltk import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer, scoring

# import rouge_scorer
import bert_score

from dataclss import DfDict


class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True, split_summaries=True
        )
        self.langtool = language_tool_python.LanguageToolPublicAPI(
            "en-US"
        )  # use a remote server (automatically set up), language English
        self.b_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)

    def rogue(self, reference, candidate):
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
        scores = self.scorer.score(reference, candidate)
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

    def bert_score(self, reference, candidate):
        # score inputs: list of candidate sentences, list of reference sentences
        # score outputs: precision, recall, f1 tensors. Same number of elements as input

        ref_sents = sent_tokenize(reference)
        cand_sents = sent_tokenize(candidate)

        p, r, f1 = self.b_scorer.score(cand_sents, ref_sents, verbose=True)

        # get mean:
        mean = f1.mean()

        return p, r, f1, mean

    def textual_entailment(self):
        pass


    def evaluate_dict(self, info_dict: DfDict, reference: str = ""):
        if reference != "":
            rogue_scores = self.rogue(reference, info_dict.prediction)
            info_dict.rogue_1 = rogue_scores["rouge1"].fmeasure
            info_dict.rogue_2 = rogue_scores["rouge2"].fmeasure
            info_dict.rogue_L = rogue_scores["rougeLsum"].fmeasure
            p, r, f1, mean = self.bert_score(reference, info_dict.prediction)
            info_dict.bert_score = float(mean)
        
        info_dict.avg_error_count_score = self.avg_error_count_score(info_dict.prediction)

        return info_dict