from typing import List
from constants import TOPIC_TEMPLATE
from gpt import Gpt
from evaluator import Evaluator
from bert_score import BERTScorer
from dataclss import Message
from string import Template


summarizer = Gpt()
evaluator = Evaluator()

num_examples = 10
bullet_max_length = 45
subheading_max_length = 40

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


def repeat_pipe(
    gpt: Gpt, evaluator: Evaluator, text: str, useChat: bool = True, reference: str = ""
):
    info_dict = gpt.repeat_summarization(text, useChat)
    info_dict = evaluator.evaluate_dict(info_dict, reference)
    gpt.save_df(info_dict, "repeat.csv")


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


def briefness_pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    text: str,
    reference: str = "",
):
    for modifier in length_modifiers:
        for _ in range(num_examples):
            info_dict = gpt.briefness_summarize(text, modifier)
            info_dict = evaluator.evaluate_dict(info_dict, reference)
            gpt.save_df(info_dict, "briefness.csv")

    messages = gpt.create_chat_messages("", text, "shorten_as_possible")

    # Shorten as possible
    for _ in range(num_examples):
        result = gpt.chat_completion(messages).choices[0].message
        info_dict = gpt.to_df_dict(
            Template("shorten_as_possible"),
            result,
            text=text,
        )
        info_dict = evaluator.evaluate_dict(info_dict, reference)
        gpt.save_df(info_dict, "briefness.csv")


quality_modifiers = [
    "articulate",
    "eloquent",
    "well-written",
    "well-put",
    "well-expressed",
    "well-stated",
]

flow_modifier = []

structure_modifiers = [
    "well-constructed",
    "well-organized",
    "well-arranged",
    "well-ordered",
    "well-structured",
    "well-phrased",
    "well-worded",
    "well-composed",
]

# Words contained: marked by the use of few words to convey much information or meaning
dense_modifiers = [
    "compendious",
    "concise",
    "succinct",
    "pithy",
    "terse",
    "epigrammatic",
    "telegraphic",
    "condensed",
    "crisp",
    "aphoristic",
    "compact",
    "monosyllabic",
    "laconic",
    "sententious",
    "elliptical",
    "elliptic",
    "apothegmatic",
    "significant",
    "well-turned",
]
