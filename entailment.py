# https://github.com/dh1105/Sentence-Entailment/blob/main/Sentence_Entailment_BERT.ipynb


import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, Trainer
from sklearn.metrics import classification_report
from datasets import load_dataset, ClassLabel

import pandas as pd
import re
import torch

# import torch_xla
# import torch_xla.core.xla_model as xm
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    SequentialSampler,
    RandomSampler,
)
from torch.nn.utils.rnn import pad_sequence

# from keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np
import configparser
from huggingface_hub import login


config = configparser.ConfigParser()
config.read(".env")
huggingface_token = config["keys"]["HUGGINGFACE_KEY"]
login(token=huggingface_token)
rte = load_dataset("EndMO/rte")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


id2label = {0: "not_entailment", 1: "entailment"}
label2id = {"not_entailment": 0, "entailment": 1}


def get_splits(split):
    premises = [value.get("premise") for value in rte[split].select_columns("premise")]
    hypotheses = [
        value.get("hypothesis") for value in rte[split].select_columns("hypothesis")
    ]
    labels = [value.get("label") for value in rte[split].select_columns("label")]
    return premises, hypotheses, labels


class EntailmentData(Dataset):
    def __init__(self):
        self.label_dict = {"not_entailment": 0, "entailment": 1}

        self.base_path = "/content/"
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.train_data = None
        self.val_data = None
        self.init_data()

    def init_data(self):
        self.train_data = self.load_data("train")
        self.val_data = self.load_data("validation")

    def load_data(self, split):
        MAX_LEN = 512
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        premises, hypotheses, labels = get_splits(split)

        for premise, hypothesis, label in zip(premises, hypotheses, labels):
            premise_id = self.tokenizer.encode(premise, add_special_tokens=False)
            hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens=False)
            pair_token_ids = (
                [self.tokenizer.cls_token_id]
                + premise_id
                + [self.tokenizer.sep_token_id]
                + hypothesis_id
                + [self.tokenizer.sep_token_id]
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
            y.append(self.label_dict[label])

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        print(len(dataset))
        return dataset

    def get_data_loaders(self, batch_size=32, shuffle=True):
        train_loader = DataLoader(
            self.train_data, shuffle=shuffle, batch_size=batch_size
        )

        val_loader = DataLoader(self.val_data, shuffle=shuffle, batch_size=batch_size)

        return train_loader, val_loader


entailmentdata = EntailmentData()

train_loader, val_loader = entailmentdata.get_data_loaders(batch_size=16)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
model.to(device)


param_optimizer = list(model.named_parameters())
no_decay = ["bias", "gamma", "beta"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay_rate": 0.01,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay_rate": 0.0,
    },
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


def multi_acc(y_pred, y_test):
    acc = (
        torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test
    ).sum().float() / float(y_test.size(0))
    return acc


import time

EPOCHS = 5


def train(model, train_loader, val_loader, optimizer):
    total_step = len(train_loader)

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(
            train_loader
        ):
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)
            # prediction = model(pair_token_ids, mask_ids, seg_ids)
            loss, prediction = model(
                pair_token_ids,
                token_type_ids=seg_ids,
                attention_mask=mask_ids,
                labels=labels,
            ).values()

            # loss = criterion(prediction, labels)
            acc = multi_acc(prediction, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc.item()

        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_acc = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(
                val_loader
            ):
                optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                labels = y.to(device)

                # prediction = model(pair_token_ids, mask_ids, seg_ids)
                loss, prediction = model(
                    pair_token_ids,
                    token_type_ids=seg_ids,
                    attention_mask=mask_ids,
                    labels=labels,
                ).values()

                # loss = criterion(prediction, labels)
                acc = multi_acc(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(
            f"Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    return model


model = train(model, train_loader, val_loader, optimizer)

# Save the model
model.save_pretrained("model")

trainer = Trainer(model=model, tokenizer=entailmentdata.tokenizer)
trainer.push_to_hub("text-entailment-bert")
