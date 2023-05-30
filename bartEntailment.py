# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline, BertTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from datasets import load_dataset
import configparser
import torch

config = configparser.ConfigParser()
config.read(".env")
huggingface_token = config["keys"]["HUGGINGFACE_KEY"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rte = load_dataset("EndMO/rte")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

model = AutoModelForSequenceClassification.from_pretrained("EndMO/text-entailment-bert")


classifier = pipeline(
    model="EndMO/text-entailment-bert",
    tokenizer=tokenizer,
    device=device,
    task="zero-shot-classification",
)


ex_index = 21

ex = (
    rte["validation"][ex_index]["premise"]
    + tokenizer.sep_token
    + rte["validation"][ex_index]["hypothesis"]
    + tokenizer.sep_token
)

print(rte["validation"][ex_index])

model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
label = "entailment"

premise = rte["validation"][ex_index]["premise"]
hypothesis = rte["validation"][ex_index]["hypothesis"]
# hypothesis = f"This example is {label}."

# run through model pre-trained on MNLI
tokens = tokenizer.encode(premise, hypothesis, return_tensors="pt", truncation=True)

outputs = model(tokens)
# print(model.id2label)
print(outputs.keys())


def logits_to_probs(logits):
    probs = logits.softmax(dim=1)
    return {
        "contradiction": probs[0][0].item(),
        "neutral": probs[0][1].item(),
        "entailment": probs[0][2].item(),
    }


logits = outputs.logits

print(model.config.id2label)
print(logits_to_probs(logits))

# print(prob_label_is_true)
