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
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorWithPadding,
    AdamW,
)

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


class TextEntailmentDataset(Dataset):
    def __init__(self, premises, hypotheses, labels):
        self.label_dict = {"not_entailment": 0, "entailment": 1}
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        premise = self.premises[index]
        hypothesis = self.hypotheses[index]
        label = self.labels[index]

        return {"premise": premise, "hypothesis": hypothesis, "label": label}


# login(token="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")
rte = load_dataset("EndMO/rte")

print(rte["train"][0])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
id2label = {0: "not_entailment", 1: "entailment"}
label2id = {"not_entailment": 0, "entailment": 1}

# rte.cast_column(
#     "label", ClassLabel(num_classes=2, names=["not_entailment", "entailment"])
# )

# print(rte["train"][0])


def preprocess(split):
    MAX_LEN = 4096
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []
    premises = [value.get("premise") for value in rte[split].select_columns("premise")]
    hypotheses = [
        value.get("hypothesis") for value in rte[split].select_columns("hypothesis")
    ]
    labels = [value.get("label") for value in rte[split].select_columns("label")]

    tokenized_inputs = tokenizer(
        premises, hypotheses, truncation=True, padding=True, return_tensors="pt"
    )

    # print(premises[-1])
    # print(hypotheses[-1])
    # print(labels[-1])

    input_ids = tokenized_inputs["input_ids"]
    print(tokenized_inputs)
    attention_mask = tokenized_inputs["attention_mask"]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)
    labels = torch.tensor(encoded_labels)

    dataset = TextEntailmentDataset(input_ids, attention_mask, labels)

    print(len(dataset))
    return dataset

    for premise, hypothesis, label in zip(premises, hypotheses, labels):
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
    entailment_set = TextEntailmentDataset(premises, hypotheses, labels)
    print(len(dataset))
    return dataset


tokenized_rte_train = preprocess("train")
tokenized_rte_validate = preprocess("validation")

print(tokenized_rte_train[0])
# print(tokenized_rte_validate[0])

batch_size = 32
train_dataloader = DataLoader(tokenized_rte_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(tokenized_rte_validate, batch_size=batch_size)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_training_steps)

# Start the training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids
        attention_mask = attention_mask
        labels = labels

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss}")

# Save the fine-tuned model
output_dir = "./output_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_rte_train,
#     eval_dataset=tokenized_rte_validate,
#     compute_metrics=lambda pred: {
#         "accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1))
#     },
# )

# trainer.train()


# def tokenize_function(examples):
#     return tokenizer(
#         examples["premise"],
#         examples["hypothesis"],
#         padding="max_length",
#         truncation=True,
#         max_length=4096,
#     )


# tokenized_dataset = rte.map(tokenize_function, batched=True)


# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# accuracy = evaluate.load("accuracy")

# # print(tokenized_dataset.__getitem__(1))


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)


# model = AutoModelForSequenceClassification.from_pretrained(
#     "allenai/longformer-base-4096", num_labels=2, id2label=id2label, label2id=label2id
# )

# training_args = TrainingArguments(
#     output_dir="longformer_entailment",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     push_to_hub=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()
# trainer.push_to_hub()


# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."


# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
# classifier(text)

# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
# classifier(text)
