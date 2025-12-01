# %% [markdown]
# # 03 – Spam Data Slicing with Snorkel
# Monitor performance on important slices of the data.

# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from textblob import TextBlob

from snorkel.slicing import (
    slicing_function,
    SlicingFunction,
    PandasSFApplier,
    slice_dataframe,
)
from snorkel.analysis import Scorer

from src.spam_data_utils import load_spam_dataset

# -----------------------------
# 1. LOAD DATA
# -----------------------------
# Here we load *true* labels so we can measure slice F1.
df_train, df_test = load_spam_dataset(load_train_labels=True)
Y_test = df_test.label.values

print("Train size:", len(df_train))
print("Test size:", len(df_test))

# -----------------------------
# 2. BASELINE MODEL (NO SLICES)
# -----------------------------
vectorizer = CountVectorizer(ngram_range=(1, 1))
X_train = vectorizer.fit_transform(df_train.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())

clf = LogisticRegression(C=0.001, solver="liblinear", max_iter=300)
clf.fit(X_train, df_train.label.values)

preds_test = clf.predict(X_test)
print(f"Overall test F1 (no slicing): {f1_score(Y_test, preds_test):.3f}")

# -----------------------------
# 3. DEFINE SLICING FUNCTIONS
# -----------------------------

@slicing_function()
def short_comment(x):
    """Short comments (often ham like 'cool video')."""
    return len(x.text.split()) < 5


def keyword_lookup(x, keywords):
    return any(word in x.text.lower() for word in keywords)


def make_keyword_sf(keywords):
    return SlicingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )


keyword_please = make_keyword_sf(["please", "plz"])


@slicing_function()
def regex_check_out(x):
    """Comments that say 'check ... out'."""
    return bool(re.search(r"check.*out", x.text, flags=re.I))


@slicing_function()
def short_link(x):
    """Shortened .ly links."""
    return bool(re.search(r"\w+\.ly", x.text))


# TextBlob-based sentiment slice
from snorkel.preprocess import preprocessor


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    return x


@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    """Very positive comments (often ham we care about)."""
    return x.polarity > 0.9


sfs = [short_comment, keyword_please, regex_check_out]
applier = PandasSFApplier(sfs)
slice_matrix = applier.apply(df_test)


os.makedirs("models", exist_ok=True)

# Save per-slice metrics
slice_results = []

for sf in sfs:
    slice_name = sf.name
    mask = slice_matrix[slice_name].astype(bool)

    y_true_slice = Y_test[mask]
    y_pred_slice = preds_test[mask]

    if len(y_true_slice) > 0:
        f1_slice = f1_score(y_true_slice, y_pred_slice)
    else:
        f1_slice = None

    slice_results.append({
        "slice_name": slice_name,
        "num_samples": len(y_true_slice),
        "f1_score": f1_slice
    })

slice_df = pd.DataFrame(slice_results)
slice_path = "models/slice_metrics.csv"
slice_df.to_csv(slice_path, index=False)
print(f"[Saved] Slice metrics → {slice_path}")

# Save overall metrics
overall_path = "models/slicing_overall_metrics.txt"
with open(overall_path, "w") as f:
    f.write("=== Data Slicing Results ===\n")
    f.write(f"Train Size: {len(df_train)}\n")
    f.write(f"Test Size: {len(df_test)}\n")
    f.write(f"Overall F1 (no slicing): {f1_score(Y_test, preds_test):.3f}\n")

print(f"[Saved] Overall slicing metrics → {overall_path}")