from huggingface_hub import login
from datasets import load_dataset, ClassLabel
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import AutoTokenizer
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorWithPadding,
)

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# login(token="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")
rte = imdb = load_dataset("EndMO/rte")

print(rte["train"][0])

dicts = [{key: value[i] for key, value in rte.items()} for i in range(len(rte))]

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
id2label = {0: "not_entailment", 1: "entailment"}
label2id = {"not_entailment": 0, "entailment": 1}

rte.cast_column(
    "label", ClassLabel(num_classes=2, names=["not_entailment", "entailment"])
)

print(rte["train"][0])


def preprocess(split):
    MAX_LEN = 4096
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []
    premise_list = [
        value.get("premise") for value in rte[split].select_columns("premise")
    ]
    hypothesis_list = [
        value.get("hypothesis") for value in rte[split].select_columns("hypothesis")
    ]
    label_list = [value.get("label") for value in rte[split].select_columns("label")]

    for premise, hypothesis, label in zip(premise_list, hypothesis_list, label_list):
        premise_id = tokenizer.encode(premise, add_special_tokens=False)
        hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens=False)
        pair_token_ids = (
            [tokenizer.cls_token_id]
            + premise_id
            + [tokenizer.sep_token_id]
            + hypothesis_id
            + [tokenizer.sep_token_id]
        )
        premise_len = len(premise_id)
        hypothesis_len = len(hypothesis_id)

        segment_ids = torch.tensor(
            [0] * (premise_len + 2) + [1] * (hypothesis_len + 1)
        )  # sentence 0 and sentence 1
        attention_mask_ids = torch.tensor(
            [1] * (premise_len + hypothesis_len + 3)
        )  # mask padded values

        token_ids.append(torch.tensor(pair_token_ids))
        seg_ids.append(segment_ids)
        mask_ids.append(attention_mask_ids)
        y.append(label2id[label])

    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    print(len(dataset))
    return dataset


# tokenized_rte_train = preprocess("train")
# tokenized_rte_validate = preprocess("validation")


def tokenize_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=4096,
    )


tokenized_dataset = rte.map(tokenize_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

# print(tokenized_dataset.__getitem__(1))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()


# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."


# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
# classifier(text)

# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
# classifier(text)
