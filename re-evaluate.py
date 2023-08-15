import evaluator as e
import gpt as g
import pandas as pd

e = e.Evaluator()
g = g.Gpt()
gold_df = pd.read_csv("data/manual_summaries2.csv", sep=";")


base_path = "results/" + g.chat_model + "/"

csv_names = [
    "baseline",
    # "headings_first",
    # "in-context",
    # "repeat",
    # "template",
    # "improve",
    # "persona",
    # "topic",
    # "important_parts",
]

for csv_name in csv_names:
    full_path = base_path + csv_name + ".csv"
    df = pd.read_csv(full_path)

    new_full_path = base_path + "new" + csv_name + ".csv"
    e.evaluate_df(df, g, gold_df, new_full_path, ["rouge"])
