import time
from constants import (
    FORMAT_MODIFIERS,
    DENSENESS_MODIFIERS,
    LANGUAGETOOL_CATEGORIES,
    LENGTH_MODIFIERS,
    QUALITY_MODIFIERS,
    STRUCTURE_MODIFIERS,
    RELEVANCE_MODIFIERS,
)
from gpt import Gpt

from evaluator import Evaluator
from utils import get_examples

ex = get_examples()
transcripts, summ1, summ2, summ3, summ4 = ex[0], ex[1], ex[2], ex[3], ex[4]
g = Gpt()
e = Evaluator()


base_path = "results/modifiers/" + g.chat_model + "/paraphrasing/"
all_modifiers = [
    # FORMAT_MODIFIERS,
    # DENSENESS_MODIFIERS,
    # LENGTH_MODIFIERS,
    # QUALITY_MODIFIERS,
    # STRUCTURE_MODIFIERS,
    RELEVANCE_MODIFIERS
]

modifies = [
    # "format",
    # "denseness",
    # "length",
    # "quality",
    # "structure",
    "relevance"
]

g.temperature = 1.0

# # PARAPHRASING ALL MODIFIERS
# for i in range(len(all_modifiers)):
#     for modifier in all_modifiers[i]:
#         for j in range(5):
#             path = base_path + modifies[i] + "_translate" + ".csv"
#             info_dict = g.translation_paraphrase(modifier, "German")
#             g.save_df(info_dict, path, True)

#             path = base_path + modifies[i] + "_paraphrase" + ".csv"
#             info_dict = g.paraphrase(modifier)
#             g.save_df(info_dict, path, True)


# e.langtool.check(
#     input_text="This is out side, please come in.",
#     text="This is out side, please come in.",
#     api_url="https://api.languagetool.org/",
#     lang="en-US",
#     enabled_categories=["TYPOS"],
#     enabled_only=True,
# )
res = e.langtool.check(
    input_text="I have you",
    api_url="https://languagetool.org/api/v2/",
    lang="en-US",
)

for i in res["matches"]:
    if i["rule"]["category"]["id"] not in LANGUAGETOOL_CATEGORIES:
        print(i)
