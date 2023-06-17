from typing import List
from constants import TOPIC_TEMPLATE
from gpt import Gpt
from evaluator import Evaluator
from bert_score import BERTScorer
from dataclss import Message
from string import Template
from constants import BULLET_MAX_LENGTH


num_examples = 10


def pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    text: str,
    title: str,
    reference: str = "",
    topic="",
    examples: List[List[str]] = [[]],
    num_examples: int = 0,
    name="",
    use_chat=True,
    only_outputs=False,
):
    path = "results/"
    path += "gpt4/" if use_chat else "gpt3/"

    name_to_function = {
        "baseline": gpt.baseline_summarization,
        "follow_up": gpt.follow_up_summarization,
        "topic": gpt.topic_summarization,
        "persona": gpt.persona_summarization,
        "improve": gpt.improve_summarization,
        "repeat": gpt.repeat_summarization,
        "important_parts": gpt.important_parts_summarization,
        "in-context": gpt.in_context_summarization,
        "induce": gpt.induce_instruction,
    }

    info_dict = {}
    if name == "topic" or name == "persona":
        info_dict = name_to_function[name](text=text, topic=topic, use_chat=use_chat)
    elif name == "in-context":
        info_dict = name_to_function[name](
            text=text,
            examples=examples,
            num_examples=num_examples,
            use_chat=use_chat,
            only_outputs=only_outputs,
        )
    elif name == "induce":
        info_dict = name_to_function[name](
            examples=examples, num_examples=num_examples, use_chat=use_chat
        )
    else:
        info_dict = name_to_function[name](text=text, use_chat=use_chat)

    info_dict.update({"title": title})

    info_dict = evaluator.evaluate_dict(info_dict, reference)
    gpt.save_df(info_dict, path + name + ".csv", use_chat)


def in_context_pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    text: str,
    reference: str = "",
    examples: List[List[str]] = [[]],
    num_examples: int = 0,
    use_chat=False,
):
    info_dict = gpt.in_context_summarization(examples, text, num_examples, use_chat)
    info_dict = evaluator.evaluate_dict(info_dict, reference)
    folder = "results/gpt-4/" if use_chat else "results/gpt-3/"
    gpt.save_df(info_dict, folder + "in-context.csv")


def induce_pipe(gpt: Gpt, examples: List[List[str]], num_examples: int, use_chat=False):
    info_dict = gpt.induce_instruction(examples, num_examples, use_chat)
    if not use_chat:
        gpt.save_df(info_dict, "results/gpt3/instruction-induction.csv", use_chat)
    else:
        print("Saving to results/gpt4/instruction-induction.csv")
        gpt.save_df(info_dict, "results/gpt4/instruction-induction.csv", use_chat)


# Without words that are potentially rude
length_modifiers = [
    "brief",
    "short",
    "shortened",
    "abbreviated",
    "abridged",
    "curtailed",
    "less than " + str(BULLET_MAX_LENGTH) + " characters long",
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
            gpt.save_df(info_dict, "results/briefness.csv")

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
        gpt.save_df(info_dict, "results/briefness.csv")

    for modifier in dense_modifiers:
        for _ in range(num_examples):
            info_dict = gpt.briefness_summarize(text, modifier)
            info_dict = evaluator.evaluate_dict(info_dict, reference)
            gpt.save_df(info_dict, "results/dense.csv")


quality_modifiers = [
    "articulate",
    "eloquent",
    "well-written",
    "well-put",
    "well-expressed",
    "well-stated",
]

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


def quality_pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    text: str,
    reference: str = "",
):
    for modifier in quality_modifiers:
        for _ in range(num_examples):
            info_dict = gpt.quality_summarize(text, modifier)
            info_dict = evaluator.evaluate_dict(info_dict, reference)
            gpt.save_df(info_dict, "results/quality.csv")

    for modifier in structure_modifiers:
        for _ in range(num_examples):
            info_dict = gpt.quality_summarize(text, modifier)
            info_dict = evaluator.evaluate_dict(info_dict, reference)
            gpt.save_df(info_dict, "results/structure.csv")
