import time
from unicodedata import normalize
from constants import (
    FORMAT_MODIFIERS,
    DENSENESS_MODIFIERS,
    LENGTH_MODIFIERS,
    QUALITY_MODIFIERS,
    STRUCTURE_MODIFIERS,
)
import pandas as pd
from gpt import Gpt

from evaluator import Evaluator
import pipelines as pipes
from utils import clean_summary, clean_transcript, get_examples, get_uncleaned_examples

ex = get_examples()
transcripts, summ1, summ2, summ3, summ4 = ex[0], ex[1], ex[2], ex[3], ex[4]
gold_df = pd.read_csv("data/manual_summaries2.csv", sep=";")

unex = get_uncleaned_examples()
ts, unsumm1, unsumm2, unsumm3, unsumm4 = unex[0], unex[1], unex[2], unex[3], unex[4]


def run_pipeline(
    gpt,
    evaluator,
    pipe_name,
    gold_df,
    runs_per_example=1,
    num_examples=0,
    only_outputs=False,
    use_chat=True,
):
    for i, row in gold_df.iterrows():
        # if i == 1:
        #     break
        transcript = transcripts[i]
        topics = row["topic"].replace(",", ", ")
        description = row["description"]
        references = [
            summ1[i],
            summ2[i],
            summ3[i],
            summ4[i],
        ]

        ex_transcripts, ex_summ4 = [], []
        if pipe_name in ["in_context", "induce"]:
            ex_transcripts = transcripts.copy()
            ex_transcripts.remove(transcript)
            ex_summ4 = unsumm4.copy()
            ex_summ4.remove(unsumm4[i])

        # examples[0].remove(row["transcript"])
        # examples[1].remove(row["summary"])

        for _ in range(runs_per_example):
            print("Running", pipe_name, "on example ", str(i + 1) + "...")
            pipes.pipe(
                gpt=gpt,
                evaluator=evaluator,
                text=transcript,
                title=row["title"],
                references=references,
                topic=topics,
                description=description,
                examples=[ex_transcripts, ex_summ4],
                num_examples=num_examples,
                name=pipe_name,
                only_outputs=only_outputs,
                use_chat=use_chat,
            )
            print("Done")


g = Gpt()
e = Evaluator()

# for i in range(5):
#     run_pipeline(g, e, "repeat", example_index, True, True)

# run_pipeline(g, e, "baseline", gold_df, runs_per_example=2)

base_path = "results/" + g.chat_model + "/new_modifiers/"

all_pipes = [
    # "follow_up",
    # "topic",
    # "persona",
    # "improve",
    # "repeat",
    # "important_parts",
    # "headings_first",
    # "template",
    # "description",
    # "step_by_step",
    # "base_prompt",
    # "in_context",
    "induce",
]

# print(predictions[9])
# print(e.slorer.slor(predictions[9]))


run_pipeline(
    gpt=g,
    evaluator=e,
    pipe_name="induce",
    gold_df=gold_df,
    runs_per_example=1,
    num_examples=1,
    only_outputs=False,
    use_chat=True,
)

run_pipeline(
    gpt=g,
    evaluator=e,
    pipe_name="induce",
    gold_df=gold_df,
    runs_per_example=1,
    num_examples=4,
    only_outputs=False,
    use_chat=True,
)

run_pipeline(
    gpt=g,
    evaluator=e,
    pipe_name="induce",
    gold_df=gold_df,
    runs_per_example=1,
    num_examples=8,
    only_outputs=True,
    use_chat=True,
)

run_pipeline(
    gpt=g,
    evaluator=e,
    pipe_name="induce",
    gold_df=gold_df,
    runs_per_example=1,
    num_examples=14,
    only_outputs=True,
    use_chat=True,
)

run_pipeline(
    gpt=g,
    evaluator=e,
    pipe_name="in_context",
    gold_df=gold_df,
    runs_per_example=1,
    num_examples=14,
    only_outputs=True,
    use_chat=True,
)
