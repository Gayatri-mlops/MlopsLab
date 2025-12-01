# %% [markdown]
# # WEAK SUPERVISION — SPAM LABELING (FULL PIPELINE)

# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.spam_data_utils import load_spam_dataset


from snorkel.labeling import labeling_function, LFAnalysis, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from src.spam_data_utils import load_spam_dataset

# Constants
ABSTAIN = -1
SPAM = 1
HAM = 0

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df_train, df_test = load_spam_dataset(load_train_labels=False)
Y_test = df_test.label.values

# -----------------------------
# 2. LABELING FUNCTIONS
# -----------------------------
@labeling_function()
def lf_contains_link(x):
    return SPAM if "http" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_subscribe(x):
    return SPAM if "subscribe" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_check_out(x):
    return SPAM if "check out" in x.text.lower() else ABSTAIN

lfs = [lf_contains_link, lf_subscribe, lf_check_out]

# -----------------------------
# 3. APPLY LFs → LABEL MATRIX
# -----------------------------
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_test = applier.apply(df_test)

# Summaries
print("\nLF SUMMARY:")
print(LFAnalysis(L_train, lfs).lf_summary())


# -----------------------------
# 4. TRAIN LABEL MODEL (DENIOSING)
# -----------------------------
label_model = LabelModel(cardinality=2, verbose=False)
label_model.fit(L_train, n_epochs=300, log_freq=50)

probs_train = label_model.predict_proba(L_train)
preds_train = probs_to_preds(probs_train)

# -----------------------------
# 5. TRAIN CLASSIFIER ON WEAK LABELS
# -----------------------------
vectorizer = CountVectorizer(ngram_range=(1, 3))
X_train = vectorizer.fit_transform(df_train.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())

clf = LogisticRegression(max_iter=300)
clf.fit(X_train, preds_train)

# -----------------------------
# 6. EVALUATE CLASSIFIER
# -----------------------------
y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(Y_test, y_pred))

cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["HAM", "SPAM"])
disp.plot()

# Save weak labels for later (optional DistilBERT pipeline)
os.makedirs("models", exist_ok=True)
np.save("models/weak_labels.npy", preds_train)
