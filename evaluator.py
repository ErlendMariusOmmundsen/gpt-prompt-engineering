import re

from constants import (
    BULLET_MAX_LENGTH,
    GEVAL_COHERENCE,
    GEVAL_CONSISTENCY,
    GEVAL_FLUENCY,
    GEVAL_RELEVANCE,
    SUBHEADING_MAX_LENGTH,
)
import language_tool_python
from nltk import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer, scoring
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
import torch
import bert_score
from dataclss import DfDict
from gpt import Gpt

import inflect
from word2number import w2n


class Entailor:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    def logits_to_probs(self, logits):
        probs = logits.softmax(dim=1)
        return {
            "contradiction": probs[0][0].item(),
            "neutral": probs[0][1].item(),
            "entailment": probs[0][2].item(),
        }

    def classify_bullet(self, text, bullet):
        tokens = self.tokenizer.encode(
            text, bullet, return_tensors="pt", truncation=True
        )
        outputs = self.model(tokens)
        logits = outputs.logits
        return self.logits_to_probs(logits)

    def get_contradiction_ratio(self, classifications):
        contradictions = 0
        for classification in classifications:
            if max(classification, key=classification.get) == "contradiction":
                contradictions += 1
        return contradictions / len(classifications)

    def get_neutral_contradiction_ratio(self, classifications):
        neutral_contradications = 0
        for classification in classifications:
            if (
                max(classification, key=classification.get) == "neutral"
                and classification["contradiction"] > classification["entailment"]
            ):
                neutral_contradications += 1
        return neutral_contradications / len(classifications)

    def classify_text(self, text, prediction):
        split_predictions = prediction.split("\n")
        classifications = []
        for line in split_predictions:
            classifications.append(self.classify_bullet(text, line))

        return self.get_contradiction_ratio(
            classifications
        ), self.get_neutral_contradiction_ratio(classifications)


class SLORer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        self.model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-560m",
            device_map="auto",
            torch_dtype="auto",
            output_hidden_states=True,
        ).to(self.device)

    # Changed from perplexity calculation: https://huggingface.co/docs/transformers/perplexity

    def sentProb(self, sent):
        encodings = self.tokenizer(sent, return_tensors="pt")
        max_length = 2048
        stride = 512

        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        sentProb = 0
        for begin_loc in tqdm(range(0, seq_len, stride), leave=False):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids)
                # https://stackoverflow.com/a/64796860/15557377
                action_logits = outputs.logits
                probs = torch.softmax(outputs.logits, dim=-1)
                for tensor in probs[0]:
                    prob = torch.max(tensor)
                    sentProb += prob

                # Sum of all actions will equal the length of input as all distributions for each word prediction sum to 1
                # == print(len(torch.softmax(outputs.logits, dim = -1)[0])))
                # print(torch.softmax(outputs.logits, dim = -1))
        sentLen = len(sent)
        sentProb = sentProb / sentLen
        return sentProb

    def sent_slor(self, sent):
        sentP = self.sentProb(sent)
        sumWordProb = 1
        ids = self.tokenizer.encode(sent.lower())
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        for token in tokens:
            sumWordProb = sumWordProb * self.sentProb(token)
        return (1 / len(tokens) * (torch.log(sentP) - torch.log(sumWordProb))).item()

    def slor(self, text):
        sent_tokens = text.split("\n")
        slors = []
        for sent in sent_tokens:
            processed_sent = sent.replace("\n", "")
            # processed_sent = processed_sent.replace("- ", "")
            slors.append(self.sent_slor(sent))
        return sum(slors) / len(slors)


class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True, split_summaries=True
        )
        self.langtool = language_tool_python.LanguageToolPublicAPI(
            "en-US"
        )  # use a remote server (automatically set up), language English
        self.b_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
        self.slorer = SLORer()
        self.entailor = Entailor()
        self.inflect = inflect.engine()

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
        if len(check) > 0:
            print(check)
        numTokens = len(word_tokenize(sent))
        numErrors = len(check)
        return 1 - float(numErrors) / float(numTokens), numErrors

    def avg_error_count_score(self, text):
        sentences = text.split("\n")
        sum = 0
        total_errors = 0
        for sent in sentences:
            score, num_errors = self.error_count_score(sent)
            sum += score
            total_errors += num_errors
        return sum / len(sentences), total_errors

    def number_hallucinations(self, text, candidate_summary):
        number_hallucinations = 0
        regex = r"\d+([,.](\d+))?"
        text_numbers_iter = re.finditer(regex, text)
        cand_numbers_iter = re.finditer(regex, candidate_summary)
        text_numbers = [match.group(0) for match in text_numbers_iter]
        cand_numbers = [match.group(0) for match in cand_numbers_iter]

        cleaned_text = re.sub(r"[^\w\s\-,\.]", "", text.lower()).split()
        cleaned_cand = re.sub(r"[^\w\s\-,\.]", "", candidate_summary.lower()).split()

        # Checks if ints and floats from candidate summary are in text
        for num in cand_numbers:
            try:
                ordinal = self.inflect.number_to_words(self.inflect.ordinal(num))
                if ordinal in cleaned_text:
                    continue
            except:
                pass
            if (
                num not in text_numbers
                and num.replace(",", ".") not in text_numbers
                and num.replace(".", ",") not in text_numbers
                and self.inflect.number_to_words(num) not in cleaned_text
            ):
                number_hallucinations += 1

        # Checks if numbers written as words from candidate summary are in text
        for word in cleaned_cand:
            try:
                if word.replace(",", "").replace(".", "").isdigit():
                    continue
                number = w2n.word_to_num(word)
                if (
                    str(number) not in text_numbers
                    and word not in cleaned_text
                    and number not in cand_numbers
                ):
                    number_hallucinations += 1
            except:
                pass

        return number_hallucinations

    def check_format(self, text):
        period_split = text.split(".")
        newline_split = text.split("\n")
        split = period_split if len(period_split) == 12 else newline_split
        three_by_three = 0
        truncated_prediction = ""

        if len(split) == 12:
            three_by_three = 1
        else:
            try:
                split = split[:12]
                truncated_prediction = "\n".join(split)
            except:
                pass

        long_subheadings = 0
        subheadings = split[::4]
        for subheading in subheadings:
            if len(subheading) > SUBHEADING_MAX_LENGTH:
                long_subheadings += 1

        long_bullets = 0
        bullets = [bullet for bullet in split if bullet not in subheadings]
        for bullet in bullets:
            if len(bullet) > BULLET_MAX_LENGTH:
                long_bullets += 1

        return truncated_prediction, three_by_three, long_subheadings, long_bullets

    def bert_score(self, reference, candidate):
        # score inputs: list of candidate sentences, list of reference sentences
        # score outputs: precision, recall, f1 tensors. Same number of elements as input

        ref_sents = reference.split("\n")
        cand_sents = candidate.split("\n")

        p, r, f1, mean = 0.0, 0.0, 0.0, 0.0

        if len(ref_sents) != len(cand_sents):
            try:
                p, r, f1 = self.b_scorer.score(candidate, reference, verbose=True)
                mean = f1.mean()
            except:
                print("BERT score failed")
        else:
            p, r, f1 = self.b_scorer.score(cand_sents, ref_sents, verbose=True)
            mean = f1.mean()

        # get mean:

        return p, r, f1, mean

    def evaluate_dict(self, gpt: Gpt, info_dict: DfDict, reference: str = ""):
        (
            truncated_prediction,
            info_dict.three_by_three,
            info_dict.long_subheadings,
            info_dict.long_bullets,
        ) = self.check_format(info_dict.prediction)
        if info_dict.three_by_three == 0:
            info_dict.prediction = truncated_prediction

        if reference != "":
            rogue_scores = self.rogue(reference, info_dict.prediction)
            info_dict.rogue_1 = rogue_scores["rouge1"].fmeasure
            info_dict.rogue_2 = rogue_scores["rouge2"].fmeasure
            info_dict.rogue_L = rogue_scores["rougeLsum"].fmeasure
            print(info_dict.prediction)
            p, r, f1, mean = self.bert_score(reference, info_dict.prediction)
            info_dict.bert_score = float(mean)
            info_dict.number_hallucinations = self.number_hallucinations(
                info_dict.text, info_dict.prediction
            )

        (
            info_dict.contradiction_ratio,
            info_dict.neutral_contradiction_ratio,
        ) = self.entailor.classify_text(info_dict.text, info_dict.prediction)
        info_dict.slor = self.slorer.slor(info_dict.prediction)

        info_dict.avg_error_count_score, info_dict.errors = self.avg_error_count_score(
            info_dict.prediction
        )

        info_dict.geval_fluency = gpt.geval(
            info_dict.text, info_dict.prediction, "fluency"
        )
        info_dict.geval_fluency = gpt.geval(
            info_dict.text, info_dict.prediction, "coherence"
        )
        info_dict.geval_fluency = gpt.geval(
            info_dict.text, info_dict.prediction, "consistency"
        )
        info_dict.geval_fluency = gpt.geval(
            info_dict.text, info_dict.prediction, "relevancy"
        )

        return info_dict
