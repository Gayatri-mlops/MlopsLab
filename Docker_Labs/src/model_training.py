import json
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Loading Breast Cancer (Wisconsin) dataset...")
data = load_breast_cancer()
X = data.data.astype(np.float32)              # shape (n_samples, 30)
y = data.target.astype(np.float32)            # 0 = malignant, 1 = benign

feature_names = list(data.feature_names)      # ordered feature names
class_names = ["malignant", "benign"]         # aligns with y: 0, 1

# Train/val/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Building model...")
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1, activation='sigmoid')    # binary output
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[keras.metrics.AUC(name="auc"), "accuracy"]
)

print("Training model...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=60,
    batch_size=32,
    verbose=1
)

print("\nEvaluating on held-out test set...")
test_loss, test_auc, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test AUC: {test_auc:.4f} | Test Accuracy: {test_acc*100:.2f}%")

print("Saving model and preprocessing artifacts...")
model.save("cancer_model.keras")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save metadata to preserve feature ordering and classes
with open("metadata.json", "w") as f:
    json.dump({
        "feature_names": feature_names,
        "class_names": class_names,
        "target_definition": "0=malignant, 1=benign"
    }, f, indent=2)

print("Saved: cancer_model.keras, scaler.pkl, metadata.json")
