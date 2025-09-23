# src/main.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load

# Works whether you start with: `uvicorn src.main:app` or run files directly
try:
    from .predict import IrisIn, PredictionOut, FEATURE_COLUMNS
except ImportError:
    from predict import IrisIn, PredictionOut, FEATURE_COLUMNS

# ---- FastAPI app (this must be named `app`) ----
app = FastAPI(title="Iris Classifier API", version="1.0.0")

# Model location (trained by src/train.py)
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "iris_model.pkl"

# Loaded on startup
_pipeline = None                 # sklearn Pipeline (StandardScaler + LogisticRegression)
_label_names: List[str] = []     # ["setosa","versicolor","virginica"]
_id2name: Dict[int, str] = {}    # 0->"setosa", etc.


@app.on_event("startup")
def _load_model() -> None:
    """Load the trained artifact once when the server starts."""
    global _pipeline, _label_names, _id2name

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Train first with:  python -m src.train"
        )

    artifact = load(MODEL_PATH)           # {"pipeline": ..., "label_classes": [...]}
    _pipeline = artifact["pipeline"]
    _label_names = list(artifact["label_classes"])
    _id2name = {i: name for i, name in enumerate(_label_names)}


@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/info", tags=["health"])
def info():
    return {
        "classes": _label_names,
        "features": FEATURE_COLUMNS,
        "pipeline_type": type(_pipeline).__name__ if _pipeline is not None else None,
        "supports_proba": hasattr(_pipeline, "predict_proba") if _pipeline is not None else False,
    }


@app.post("/predict", response_model=PredictionOut, tags=["inference"])
def predict_one(x: IrisIn):
    """Predict a single iris sample."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([[getattr(x, c) for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    try:
        pred_int = int(_pipeline.predict(df)[0])
        proba = _pipeline.predict_proba(df)[0]
        classes_int = list(_pipeline.classes_)  # e.g. [0,1,2]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    species = _id2name.get(pred_int, str(pred_int))
    probs = {_id2name.get(int(ci), str(ci)): float(p) for ci, p in zip(classes_int, proba)}
    return {"species": species, "probabilities": probs}


@app.post("/predict_batch", response_model=List[PredictionOut], tags=["inference"])
def predict_batch(items: List[IrisIn]):
    """Predict a list of iris samples."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([{c: getattr(item, c) for c in FEATURE_COLUMNS} for item in items])

    try:
        preds_int = _pipeline.predict(df)
        probas = _pipeline.predict_proba(df)
        classes_int = list(_pipeline.classes_)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    out: List[PredictionOut] = []
    for pred_i, proba in zip(preds_int, probas):
        species = _id2name.get(int(pred_i), str(int(pred_i)))
        probs = {_id2name.get(int(ci), str(ci)): float(p) for ci, p in zip(classes_int, proba)}
        out.append(PredictionOut(species=species, probabilities=probs))
    return out
