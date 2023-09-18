from constants import (
    DENSENESS_MODIFIERS,
    FORMAT_MODIFIERS,
    LENGTH_MODIFIERS,
    QUALITY_MODIFIERS,
    RELEVANCE_MODIFIERS,
    STRUCTURE_MODIFIERS,
)
import pandas as pd
from gpt import Gpt
from evaluator import Evaluator
import pipelines as pipes
from utils import get_examples

ex = get_examples()
transcripts, summ1, summ2, summ3, summ4 = ex[0], ex[1], ex[2], ex[3], ex[4]
gold_df = pd.read_csv("data/manual_summaries2.csv", sep=";")


g = Gpt()
e = Evaluator()


all_modifiers = [
    # FORMAT_MODIFIERS,
    # DENSENESS_MODIFIERS,
    # LENGTH_MODIFIERS,
    # QUALITY_MODIFIERS,
    STRUCTURE_MODIFIERS,
    RELEVANCE_MODIFIERS,
]

modifies = [
    # "format",
    # "denseness",
    # "length",
    # "quality",
    "structure",
    "relevance",
]

eval_dict = {
    "format": [
        "format",
    ],
    "denseness": [
        "format",
        "rouge",
        "bert_score",
        "errors",
        "entailment",
        "geval_fluency",
        "geval_relevance",
        "geval_coherence",
        "geval_consistency",
    ],
    "quality": ["entailment", "errors", "geval_fluency", "geval_coherence"],
    "structure": ["entailment", "errors", "geval_fluency", "geval_coherence"],
    "length": [
        "format",
        # "entailment",
        "errors",
        # "geval_fluency",
        # "geval_relevance",
        # "geval_coherence",
        # "geval_consistency",
    ],
    "relevance": [
        "rouge",
        "bert_score",
        "entailment",
        "geval_relevance",
        "geval_consistency",
    ],
}
# print(len(transcripts))
# print(len(all_modifiers))

for j in range(8, len(transcripts)):
    print("\tOn transcript #", str(j))
    pipes.modifier_pipe(
        gpt=g,
        evaluator=e,
        text=transcripts[j],
        modifiers=QUALITY_MODIFIERS,
        modifies="quality",
        runs_per_modifier=1,
        references=[summ1[j], summ2[j], summ3[j], summ4[j]],
        evaluate=eval_dict["quality"],
    )
print("\tDone with Quality Modifiers")

for i in range(len(all_modifiers)):
    print("On modifiers: ", str(all_modifiers[i]))
    for j in range(len(transcripts)):
        print("\tOn transcript #", str(j))
        pipes.modifier_pipe(
            gpt=g,
            evaluator=e,
            text=transcripts[j],
            modifiers=all_modifiers[i],
            modifies=modifies[i],
            runs_per_modifier=1,
            references=[summ1[j], summ2[j], summ3[j], summ4[j]],
            evaluate=eval_dict[modifies[i]],
        )
    print("\tDone with Modifiers", str(all_modifiers[i]))
