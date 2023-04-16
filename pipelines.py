from typing import List
from constants import TOPIC_TEMPLATE
from gpt import Gpt
from evaluator import Evaluator
from bert_score import BERTScorer


summarizer = Gpt()
evaluator = Evaluator()

bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
print(bert_scorer.score("As soon as you can.", "As soon as you can."))


def follow_up_pipe(gpt: Gpt, evaluator: Evaluator, text: str, reference: str = ""):
    info_dict = gpt.follow_up_summarization(text)
    info_dict = evaluator.evaluate_dict(info_dict, reference)
    gpt.save_df(info_dict, "follow-up.csv")


def topic_pipe(
    gpt: Gpt, evaluator: Evaluator, text: str, topic: str, reference: str = ""
):
    info_dict = gpt.to_df_dict(
        TOPIC_TEMPLATE, gpt.topic_completion(text, topic), text=text
    )
    info_dict = evaluator.evaluate_dict(info_dict, reference)

    gpt.save_df(info_dict, "in-context.csv")


def in_context_pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    examples: List[List[str]],
    text: str,
    num_examples: int,
    useChat=False,
    reference: str = "",
):
    info_dict = gpt.in_context_prediction(examples, text, num_examples, useChat)
    info_dict = evaluator.evaluate_dict(info_dict, reference)
    gpt.save_df(info_dict, "in-context.csv")


def induce_pipe(
    gpt: Gpt, evaluator: Evaluator, examples: List[List[str]], num_examples: int
):
    info_dict = gpt.induce_instruction(examples, num_examples)
    gpt.save_df(info_dict, "instruction-induction.csv")


def persona_pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    text: str,
    topics: List[str],
    useChat: bool = True,
    reference: str = "",
):
    persona_context = gpt.generate_persona_context(topics)
    info_dict = gpt.persona_summarization(text, persona_context, useChat)
    info_dict = evaluator.evaluate_dict(info_dict, reference)
    gpt.save_df(info_dict, "persona.csv")


def improve_pipe(
    gpt: Gpt, evaluator: Evaluator, text: str, useChat: bool = True, reference: str = ""
):
    info_dict = gpt.improve_summarization(text, useChat)
    info_dict = evaluator.evaluate_dict(info_dict, reference)
    gpt.save_df(info_dict, "improve.csv")


def briefness_pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    text: str,
    reference: str = "",
):
    bullet_max_length = 100  # TODO: change this
    subheading_max_length = 100  # TODO: change this
    # Without words that are potentially rude
    length_modifiers = [
        "brief",
        "short",
        "shortened",
        "abbreviated",
        "abridged",
        "curtailed",
        "less than " + str(bullet_max_length) + " characters long",
    ]
    for modifier in length_modifiers:
        for _ in range(10):
            info_dict = gpt.briefness_summarize(text, modifier)
            info_dict = evaluator.evaluate_dict(info_dict, reference)
            gpt.save_df(info_dict, "briefness.csv")
