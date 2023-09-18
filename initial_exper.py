import pandas as pd
from gpt import Gpt

from evaluator import Evaluator
import pipelines as pipes
from utils import clean_summary, clean_transcript, get_examples

ex = get_examples()
transcripts, summ1, summ2, summ3, summ4 = ex[0], ex[1], ex[2], ex[3], ex[4]
gold_df = pd.read_csv("data/manual_summaries2.csv", sep=";")


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
        transcript = row["transcript"]
        transcript = clean_transcript(transcript)

        references = [
            clean_summary(row["summary"]),
            clean_summary(row["summary2"]),
            clean_summary(row["summary3"]),
            clean_summary(row["summary4"]),
        ]

        topics = row["topic"].replace(",", ", ")
        description = row["description"]

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
                examples=[transcripts, summ4],
                num_examples=num_examples,
                name=pipe_name,
                only_outputs=only_outputs,
                use_chat=use_chat,
            )
            print("Done")


g = Gpt()
e = Evaluator()

base_path = "results/" + g.chat_model + "/new_modifiers/"

all_pipes = [
    "follow_up",
    "topic",
    "persona",
    "improve",
    "repeat",
    "important_parts",
    "headings_first",
    "template",
    "description",
    "step_by_step",
]

for pipe_name in all_pipes:
    run_pipeline(
        gpt=g,
        evaluator=e,
        pipe_name=pipe_name,
        gold_df=gold_df,
        runs_per_example=1,
        only_outputs=True,
        use_chat=True,
    )
