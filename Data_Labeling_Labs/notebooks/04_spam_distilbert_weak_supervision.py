# 04_spam_distilbert_weak_supervision.py
# Train DistilBERT on weak labels produced by Snorkel LFs / LabelModel.
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.spam_data_utils import load_spam_dataset

# -----------------------------
# 1. LOAD DATA + WEAK LABELS
# -----------------------------
# We assume you already ran 01_spam_labeling.ipynb and saved:
#   models/weak_labels.npy
weak_labels_path = "models/weak_labels.npy"
if not os.path.exists(weak_labels_path):
    raise FileNotFoundError(
        "models/weak_labels.npy not found. "
        "Run 01_spam_labeling.ipynb first to generate weak labels."
    )

weak_labels = np.load(weak_labels_path)

# Load same train/test split; here we do not use train true labels,
# only the *weak* labels.
df_train_ws, df_test_true = load_spam_dataset(load_train_labels=True)
texts_train = df_train_ws.text.tolist()
texts_test = df_test_true.text.tolist()
y_test_true = df_test_true.label.values

assert len(texts_train) == len(weak_labels), "Weak label length mismatch."

# Create a small validation split from weakly-labeled train
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts_train,
    weak_labels,
    test_size=0.1,
    random_state=123,
    stratify=weak_labels,
)

# -----------------------------
# 2. TOKENIZER + DATASET CLASS
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
val_dataset = SpamDataset(val_texts, val_labels, tokenizer)
test_dataset = SpamDataset(texts_test, y_test_true, tokenizer)

# -----------------------------
# 3. MODEL + TRAINING ARGS
# -----------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)

training_args = TrainingArguments(
    output_dir="models/distilbert_weak_spam",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
)

# -----------------------------
# 4. METRICS
# -----------------------------
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# -----------------------------
# 5. TRAIN WITH WEAK LABELS
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------
# 6. EVALUATE ON TRUE-LABELED TEST SET
# -----------------------------
test_metrics = trainer.evaluate(test_dataset)
print("\n=== DistilBERT (weak supervision) — Test Metrics ===")
for k, v in test_metrics.items():
    if k.startswith("eval_"):
        print(f"{k}: {v:.4f}")
