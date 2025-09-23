# src/predict.py
from __future__ import annotations

from typing import Dict, List, Iterable, Union, Optional
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from joblib import load
from pydantic import BaseModel, Field


# -------------------------
# Schemas for FastAPI usage
# -------------------------

class IrisIn(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictionOut(BaseModel):
    species: str
    probabilities: Dict[str, float]


# Keep the feature order consistent with training
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


# -------------------------
# CLI-friendly Predictor
# -------------------------

class IrisPredictor:
    """
    Loads the trained artifact produced by src/train.py and performs predictions.
    The artifact is a dict with keys:
      - "pipeline": sklearn Pipeline (StandardScaler + LogisticRegression)
      - "label_classes": list[str] of species names, in LabelEncoder order (0..n-1)
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        # Default model path
        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "model" / "iris_model.pkl"
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Train first with: python -m src.train"
            )

        artifact = load(self.model_path)
        self.pipeline = artifact["pipeline"]
        self.label_names: List[str] = list(artifact["label_classes"])

        # Map integer class -> species name (0 -> "setosa", etc.)
        self._id2name = {i: name for i, name in enumerate(self.label_names)}

        # Sanity: pipeline.classes_ are ints; ensure mapping covers them
        for k in self.pipeline.classes_:
            if int(k) not in self._id2name:
                raise ValueError(
                    f"Class {k} in pipeline not found in saved label names {self.label_names}"
                )

    # ---- helpers ----

    def _to_dataframe(
        self, rows: Iterable[Union[IrisIn, Dict[str, float], List[float], np.ndarray]]
    ) -> pd.DataFrame:
        """Convert various row formats to a DF with the proper column order."""
        prepared = []
        for r in rows:
            if isinstance(r, IrisIn):
                prepared.append([getattr(r, c) for c in FEATURE_COLUMNS])
            elif isinstance(r, dict):
                prepared.append([r[c] for c in FEATURE_COLUMNS])
            elif isinstance(r, (list, tuple, np.ndarray)):
                arr = list(r)
                if len(arr) != 4:
                    raise ValueError("Each row must have 4 values: " + ", ".join(FEATURE_COLUMNS))
                prepared.append(arr)
            else:
                raise TypeError(f"Unsupported row type: {type(r)}")
        return pd.DataFrame(prepared, columns=FEATURE_COLUMNS)

    # ---- public API ----

    def predict_one(self, x: Union[IrisIn, Dict[str, float], List[float], np.ndarray]) -> PredictionOut:
        df = self._to_dataframe([x])
        pred_int = int(self.pipeline.predict(df)[0])
        proba = self.pipeline.predict_proba(df)[0]
        classes_int = list(self.pipeline.classes_)

        species = self._id2name[pred_int]
        probs = {self._id2name[int(ci)]: float(p) for ci, p in zip(classes_int, proba)}
        return PredictionOut(species=species, probabilities=probs)

    def predict_many(
        self, X: Iterable[Union[IrisIn, Dict[str, float], List[float], np.ndarray]]
    ) -> List[PredictionOut]:
        df = self._to_dataframe(X)
        preds_int = self.pipeline.predict(df)
        probas = self.pipeline.predict_proba(df)
        classes_int = list(self.pipeline.classes_)
        out: List[PredictionOut] = []
        for pred_i, proba in zip(preds_int, probas):
            species = self._id2name[int(pred_i)]
            probs = {self._id2name[int(ci)]: float(p) for ci, p in zip(classes_int, proba)}
            out.append(PredictionOut(species=species, probabilities=probs))
        return out

    def predict_csv(self, csv_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        results = self.predict_many(df[FEATURE_COLUMNS].to_dict(orient="records"))
        # Merge predictions back
        pred_species = [r.species for r in results]
        df_out = df.copy()
        df_out["predicted_species"] = pred_species

        # Add probability columns
        prob_keys = list(results[0].probabilities.keys())
        for k in prob_keys:
            df_out[f"prob_{k}"] = [r.probabilities[k] for r in results]

        if output_path:
            output_path = Path(output_path)
            df_out.to_csv(output_path, index=False)
            print(f"Saved predictions to {output_path.resolve()}")

        return df_out

    def info(self) -> Dict:
        return {
            "model_path": str(self.model_path),
            "classes": self.label_names,
            "pipeline_type": type(self.pipeline).__name__,
            "supports_proba": hasattr(self.pipeline, "predict_proba"),
            "features": FEATURE_COLUMNS,
        }


# -------------------------
# CLI
# -------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Iris Predictor (CLI)")
    p.add_argument("--model", type=str, help="Path to model file (default: model/iris_model.pkl)")
    p.add_argument("--single", nargs=4, type=float,
                   metavar=("SEPAL_LEN", "SEPAL_W", "PETAL_LEN", "PETAL_W"),
                   help="Predict one sample: 4 numbers in order")
    p.add_argument("--csv", type=str, help="Path to CSV with feature columns")
    p.add_argument("--output", type=str, help="Where to save CSV predictions")
    return p


def main():
    args = _build_argparser().parse_args()
    predictor = IrisPredictor(args.model)

    print("Model info:", predictor.info())

    if args.single:
        sl, sw, pl, pw = args.single
        result = predictor.predict_one([sl, sw, pl, pw])
        print("\nPrediction:")
        print("  species:", result.species)
        print("  probabilities:", result.probabilities)
    elif args.csv:
        df = predictor.predict_csv(args.csv, args.output)
        print(f"\nPredicted {len(df)} rows")
        if not args.output:
            print("\nPreview:")
            print(df.head())
    else:
        print("\nNothing to do. Use --single or --csv. Try:")
        print("  python -m src.predict --single 5.1 3.5 1.4 0.2")
        print("  python -m src.predict --csv path/to/your.csv --output preds.csv")


if __name__ == "__main__":
    main()
