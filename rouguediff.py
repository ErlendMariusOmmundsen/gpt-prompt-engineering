import evaluator
import gpt
from utils import clean_summary, get_examples
import pandas as pd
from dataclss import DfDict
import numpy as np
import json

e = evaluator.Evaluator()
g = gpt.Gpt()
ex = get_examples()
transcripts, summ1, summ2, summ3, summ4 = ex[0], ex[1], ex[2], ex[3], ex[4]

base_path = "results/" + g.chat_model + "/"
baseline_df = pd.read_csv(base_path + "baseline.csv")

# for i in range(len(transcripts)):
#     # for i in range(1):
#     for j in range(10):
#         three_by_three = 0
#         baseline_prediction = ""

#         while three_by_three == 0:
#             baseline_prediction = g.baseline_summarization(transcripts[i])
#             info_dict = e.rouge_eval(
#                 baseline_prediction, [summ1[i], summ2[i], summ3[i], summ4[i]]
#             )
#             three_by_three = int(baseline_prediction.three_by_three)
#             print(baseline_prediction.three_by_three)

#         path = base_path + "transcript" + str(i + 1) + ".csv"

#         g.save_df(info_dict, path, True)

#     print("Done with transcript " + str(i))

titles = baseline_df["title"].unique()

dataframes = []

for title in titles:
    title_df = baseline_df[baseline_df["title"] == title]
    dataframes.append(title_df)

r1 = []
r2 = []
rL = []

for df in dataframes:
    pre_r1 = []
    pre_r2 = []
    pre_rL = []
    for i, row in df.iterrows():
        pre_r1.append(np.mean(json.loads(row["rouge_1"])))
        pre_r2.append(np.mean(json.loads(row["rouge_2"])))
        pre_rL.append(np.mean(json.loads(row["rouge_L"])))
    r1.append(np.mean(pre_r1))
    r2.append(np.mean(pre_r2))
    rL.append(np.mean(pre_rL))

dic = {}

for i in range(len(titles)):
    dic.update({str(i) + ": " + titles[i]: r1[i]})

    # print("Transcript " + str(i + 1))
    # print("rouge_1: " + str(r1[i]))
    # print("rouge_2: " + str(r2[i]))
    # print("rouge_L: " + str(rL[i]))
    # print("\n")

sorted_dic = dict(sorted(dic.items(), key=lambda item: item[1]))

# print(sorted_dic)


# print(summ1[0])
# print()
# print(clean_summary(summ1[0]))
print()
print(e.rouge_preprocess(clean_summary(summ1[0])))
