# Demo from https://github.com/Tiiiger/bert_score
from bert_score import score
import matplotlib.pyplot as plt
from bert_score import plot_example


cands, refs = [], []

# score inputs: list of candidate sentences, list of reference sentences
# score outputs: precision, recall, f1 tensors. Same number of elements as input 

P, R, F1 = score(cands, refs, lang="en", verbose=True)


# get mean:
print(f"System level F1 score: {F1.mean():.3f}")

# F1 distribution
plt.hist(F1, bins=20)
plt.show()

# Sentence-level visualization: pairwise cosine similarity
cand = cands[0]
ref = refs[0]
plot_example(cand, ref, lang="en")
