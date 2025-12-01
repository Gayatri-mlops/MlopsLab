# %% [markdown]
# # DATA AUGMENTATION — SNORKEL TRANSFORMATION FUNCTIONS (FULL PIPELINE)

# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from snorkel.augmentation import transformation_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.spam_data_utils import load_spam_dataset

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df_train, df_test = load_spam_dataset(load_train_labels=True)
Y_test = df_test.label.values

# -----------------------------
# 2. DEFINE TRANSFORMATION FUNCTIONS (TFS)
# -----------------------------
@transformation_function()
def tf_add_exclaim(x):
    x.text = x.text + "!!!"
    return x

@transformation_function()
def tf_lowercase(x):
    x.text = x.text.lower()
    return x

@transformation_function()
def tf_repeat_text(x):
    x.text = x.text + " " + x.text
    return x

tfs = [tf_add_exclaim, tf_lowercase, tf_repeat_text]

# -----------------------------
# 3. APPLY TRANSFORMATION FUNCTIONS (CUSTOM IMPLEMENTATION)
# -----------------------------
augmented_rows = []

for _, row in df_train.iterrows():
    for tf in tfs:
        x = row.copy()
        x = tf(x)
        augmented_rows.append(x)

df_aug = pd.DataFrame(augmented_rows)

print("Original size:", len(df_train))
print("Augmented size:", len(df_aug))

# Combine original + augmented
df_combined = pd.concat([df_train, df_aug]).reset_index(drop=True)

# -----------------------------
# 4. TRAIN CLASSIFIER ON AUGMENTED DATA
# -----------------------------
vectorizer = CountVectorizer(ngram_range=(1, 3))
X_train = vectorizer.fit_transform(df_combined.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())

clf = LogisticRegression(max_iter=300)
clf.fit(X_train, df_combined.label.values)

# -----------------------------
# 5. EVALUATION
# -----------------------------
y_pred = clf.predict(X_test)

print("Accuracy with Augmentation:", accuracy_score(Y_test, y_pred))

# 6. SAVE OUTPUTS (NEW)
os.makedirs("data", exist_ok=True)
augmented_path = "data/augmented_train.csv"
df_combined.to_csv(augmented_path, index=False)
print(f"[Saved] Augmented dataset → {augmented_path}")

os.makedirs("models", exist_ok=True)
metrics_path = "models/data_augmentation_metrics.txt"

with open(metrics_path, "w") as f:
    f.write("=== Data Augmentation Results ===\n")
    f.write(f"Original Train Size: {len(df_train)}\n")
    f.write(f"Augmented Size: {len(df_aug)}\n")
    f.write(f"Combined Size: {len(df_combined)}\n")
    f.write(f"Accuracy with Augmentation: {accuracy_score(Y_test, y_pred)}\n")

print(f"[Saved] Metrics → {metrics_path}")
