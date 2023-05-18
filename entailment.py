from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)


# login(token="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")
rte = imdb = load_dataset("EndMO/rte")

print(rte["train"][0])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples[""], truncation=True)


tokenized_rte = rte.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "non_entailment", 1: "entailment"}
label2id = {"non_entailment": 0, "entailment": 1}


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
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
    train_dataset=tokenized_rte["train"],
    eval_dataset=tokenized_rte["test"],
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
