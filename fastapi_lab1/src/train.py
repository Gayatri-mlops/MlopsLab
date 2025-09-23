# src/train.py
from pathlib import Path
import os
from typing import Dict, Any

from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Handle imports whether run as "python -m src.train" or "python src/train.py"
try:
    from .data import DataProcessor
except ImportError:
    from data import DataProcessor


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "iris_model.pkl"


def build_pipeline() -> Pipeline:
    """Standard scaler + Logistic Regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    # Use DataProcessor to clean/prepare dataset (without scaling, pipeline handles that)
    processor = DataProcessor()  # optionally pass a CSV path if you downloaded iris.csv
    X_train, X_test, y_train, y_test = processor.process_full_pipeline(
        scale_features_flag=False
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Save model + label names
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    artifact: Dict[str, Any] = {
        "pipeline": pipe,
        "label_classes": list(processor.label_encoder.classes_),  # species names
    }
    dump(artifact, MODEL_PATH)
    print(f"\n✅ Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
