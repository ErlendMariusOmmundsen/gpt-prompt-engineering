import evaluator as e
import gpt as g
import pandas as pd

e = e.Evaluator()
g = g.Gpt()
gold_df = pd.read_csv("data/manual_summaries2.csv", sep=";")
full_eval = [
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
]
eval_dict = {
    "format": [],
    "denseness": [
        "rouge",
        "bert_score",
        "errors",
        "entailment",
        "geval_fluency",
        "geval_relevance",
        "geval_coherence",
        "geval_consistency",
    ],
    "quality": [
        # "entailment",
        # "errors",
        "geval_fluency",
        # "geval_coherence"
    ],
    "structure": [
        # "entailment",
        # "errors",
        # "geval_fluency",
        "geval_coherence"
    ],
    "length": [
        # "entailment",
        "errors",
        # "geval_fluency",
        # "geval_relevance",
        # "geval_coherence",
        # "geval_consistency",
    ],
    "relevance": [
        # "rouge",
        # "bert_score",
        # "entailment",
        "geval_relevance",
        # "geval_consistency",
    ],
}

# base_path = "results/" + g.chat_model + "/initial/"
base_path = "results/modifiers/" + g.chat_model + "/"

# csv_names = [
# "base_prompt",
# "baseline",
# "follow_up",
# "topic",
# "persona",
# "improve",
# "repeat",
# "important_parts",
# "headings_first",
# "template",
# # "description",
# "step_by_step",
# "analyze_main_themes",
# "main_points",
# "in_context",
# ]

csv_names = [
    # "relevance",
    # "structure",
    "quality",
]

for csv_name in csv_names:
    full_path = base_path + csv_name + ".csv"
    df = pd.read_csv(full_path)

    print(csv_name)
    new_full_path = base_path + "new" + csv_name + ".csv"
    e.evaluate_df(
        df,
        g,
        gold_df,
        new_full_path,
        eval_dict[csv_name],
    )
