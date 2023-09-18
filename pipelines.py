from typing import List
from constants import (
    DENSENESS_MODIFIERS,
    FORMAT_MODIFIERS,
    LENGTH_MODIFIERS,
    QUALITY_MODIFIERS,
    STRUCTURE_MODIFIERS,
)
from gpt import Gpt
from evaluator import Evaluator
from string import Template
from utils import messages_to_string


num_examples = 10


def pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    text: str,
    title: str,
    references: List[str] = [],
    topic: str = "",
    description: str = "",
    examples: List[List[str]] = [[]],
    num_examples: int = 0,
    name="",
    evaluate: List[str] = [
        "format",
        "rouge",
        "bert_score",
        "number_hallucination",
        "entailment",
        "slor",
        "errors",
        "geval_fluency",
        "geval_coherence",
        "geval_consistency",
        "geval_relevance",
    ],
    use_chat=True,
    only_outputs=False,
):
    path = "results/"
    path += gpt.chat_model + "/with modifiers/" if use_chat else gpt.model + "/"

    name_to_function = {
        "base_prompt": gpt.base_prompt_summarization,
        "baseline": gpt.baseline_summarization,
        "follow_up": gpt.follow_up_summarization,
        "topic": gpt.topic_summarization,
        "persona": gpt.persona_summarization,
        "improve": gpt.improve_summarization,
        "repeat": gpt.repeat_summarization,
        "important_parts": gpt.important_parts_summarization,
        "in_context": gpt.in_context_summarization,
        "induce": gpt.induce_instruction,
        "headings_first": gpt.headings_first_summarization,
        "template": gpt.template_summarization,
        "description": gpt.description_summarization,
        "step_by_step": gpt.step_by_step_summarization,
        "analyze_main_themes": gpt.zero_shot_summarization,
        "main_points": gpt.zero_shot_summarization,
    }

    info_dict = {}
    if name == "topic" or name == "persona":
        info_dict = name_to_function[name](text=text, topic=topic, use_chat=use_chat)
    elif name == "description":
        info_dict = name_to_function[name](
            text=text, description=description, use_chat=use_chat
        )
    elif name == "in_context":
        info_dict = name_to_function[name](
            text=text,
            examples=examples,
            num_examples=num_examples,
            use_chat=use_chat,
            only_outputs=only_outputs,
        )
    elif name == "induce":
        info_dict = name_to_function[name](
            examples=examples,
            num_examples=num_examples,
            use_chat=use_chat,
            only_outputs=only_outputs,
        )
        info_dict.title = title
        info_dict = evaluator.evaluate_dict(gpt, info_dict, references, evaluate)
        gpt.save_df(info_dict, path + name + ".csv", use_chat)
        return
    elif name == "template":
        info_dict = name_to_function[name](
            text=text,
            reference_text=examples[0][0],
            reference_summary=examples[1][0],
            use_chat=use_chat,
        )
    elif name_to_function[name] == gpt.zero_shot_summarization:
        info_dict = name_to_function[name](
            text=text,
            approach=name,
        )
    else:
        info_dict = name_to_function[name](text=text)

    info_dict.title = title

    info_dict = evaluator.evaluate_dict(gpt, info_dict, references, evaluate)
    gpt.save_df(info_dict, path + name + ".csv", use_chat)


def induce_pipe(gpt: Gpt, examples: List[List[str]], num_examples: int, use_chat=False):
    info_dict = gpt.induce_instruction(examples, num_examples, use_chat)
    if not use_chat:
        gpt.save_df(info_dict, "results/gpt3/instruction-induction.csv", use_chat)
    else:
        print("Saving to results/gpt4/instruction-induction.csv")
        gpt.save_df(info_dict, "results/gpt4/instruction-induction.csv", use_chat)


def modifier_pipe(
    gpt: Gpt,
    evaluator: Evaluator,
    text: str,
    modifiers: List[str],
    modifies: str,
    runs_per_modifier: int = 1,
    references: List[str] = [],
    evaluate: List[str] = ["format"],
):
    for modifier in modifiers:
        print("\tOn modifier: ", modifier)
        for _ in range(runs_per_modifier):
            info_dict = gpt.modifier_summarize(text, modifier, modifies)
            info_dict = evaluator.evaluate_dict(gpt, info_dict, references, evaluate)
            gpt.save_df(
                info_dict,
                "results/modifiers/" + gpt.chat_model + "/" + modifies + ".csv",
            )
