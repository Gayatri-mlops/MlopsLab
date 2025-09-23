import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


SEABORN_IRIS_RAW = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"


class DataProcessor:
    def __init__(self, data_path: Optional[str | os.PathLike] = None):
        """
        DataProcessor for the Iris dataset.

        Args:
            data_path: Optional path to a local CSV. If None, the loader will
                       download iris from a public URL / seaborn.
        """
        self.data_path = str(data_path) if data_path is not None else None
        self.feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        self.target_column = "species"

        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    # ---------------- I/O ---------------- #

    def load_data(self) -> pd.DataFrame:
        """Load Iris CSV either from local path or the web."""
        if self.data_path:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
            df = pd.read_csv(self.data_path)
            print(f"Loaded local dataset: {self.data_path} shape={df.shape}")
            return df

        # Try directly from Seaborn’s hosted CSV (GitHub)
        try:
            df = pd.read_csv(SEABORN_IRIS_RAW)
            print(f"Loaded dataset from {SEABORN_IRIS_RAW} shape={df.shape}")
            return df
        except Exception as e_url:
            print(f"Failed to fetch {SEABORN_IRIS_RAW}: {e_url}")

        # Fallback: seaborn.load_dataset (needs internet)
        try:
            import seaborn as sns  # ensure seaborn in requirements
            df = sns.load_dataset("iris")
            print("Loaded dataset via seaborn.load_dataset('iris')", f"shape={df.shape}")
            return df
        except Exception as e_sns:
            raise RuntimeError(
                "Could not load Iris dataset from the web or seaborn. "
                "Provide a local CSV path to DataProcessor(data_path=...)."
            ) from e_sns

    # -------------- Diagnostics ----------- #

    def get_data_info(self, df: pd.DataFrame) -> None:
        print("\n=== Dataset Info ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\n=== Numeric Summary ===")
        print(df[self.feature_columns].describe())
        if self.target_column in df.columns:
            print("\n=== Target Distribution ===")
            print(df[self.target_column].value_counts(dropna=False))

    def check_missing_values(self, df: pd.DataFrame) -> dict:
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        print("\n=== Missing Values ===")
        for c in df.columns:
            if missing[c] > 0:
                print(f"{c}: {missing[c]} ({pct[c]}%)")
            else:
                print(f"{c}: None")
        return missing.to_dict()

    # -------------- Cleaning / Prep ------- #

    def _select_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        required = self.feature_columns + [self.target_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        sub = df[required].copy()

        # Strip spaces in string columns
        for c in sub.select_dtypes(include="object").columns:
            sub[c] = sub[c].astype(str).str.strip()
        return sub

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        dfp = df.copy()

        # Drop rows with missing target
        if dfp[self.target_column].isnull().any():
            print("Dropping rows with missing target...")
            dfp = dfp.dropna(subset=[self.target_column])

        # Impute numeric features
        if dfp[self.feature_columns].isnull().any().any():
            print("Imputing numeric features with mean...")
            dfp[self.feature_columns] = self.imputer.fit_transform(dfp[self.feature_columns])

        return dfp.reset_index(drop=True)

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enc = df.copy()
        df_enc[self.target_column + "_encoded"] = self.label_encoder.fit_transform(
            df_enc[self.target_column]
        )
        print("\n=== Target Encoding Map ===")
        for i, cls in enumerate(self.label_encoder.classes_):
            print(f"{cls} -> {i}")
        return df_enc

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        X = df[self.feature_columns]
        y = df[self.target_column + "_encoded"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        print("\n=== Split ===")
        print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
        return X_tr, X_te, y_tr, y_te

    def scale_features(self, X_train, X_test=None):
        print("\n=== Scaling (StandardScaler) ===")
        Xtr = self.scaler.fit_transform(X_train)
        if X_test is not None:
            Xte = self.scaler.transform(X_test)
            print(f"Scaled shapes → Train: {Xtr.shape}, Test: {Xte.shape}")
            return Xtr, Xte
        print(f"Scaled shape → Train: {Xtr.shape}")
        return Xtr

    # -------------- Orchestrator ---------- #

    def process_full_pipeline(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features_flag: bool = True,
    ):
        """
        Load → inspect → clean → encode → split → (optional) scale.
        Returns: X_train, X_test, y_train, y_test
        """
        print("Starting Iris data processing pipeline...")
        df = self.load_data()
        self.get_data_info(df)
        self.check_missing_values(df)

        df = self._select_required_columns(df)
        df = self.handle_missing_values(df)
        df = self.encode_target(df)

        X_train, X_test, y_train, y_test = self.split_data(df, test_size, random_state)

        if scale_features_flag:
            X_train, X_test = self.scale_features(X_train, X_test)

        print("\nPipeline complete ✅")
        return X_train, X_test, y_train, y_test

    # -------------- Optional save --------- #

    def save_processed_data(
        self, X_train, X_test, y_train, y_test, output_dir: str = "processed_data"
    ):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X_train.npy", X_train)
        np.save(out / "X_test.npy", X_test)
        np.save(out / "y_train.npy", y_train)
        np.save(out / "y_test.npy", y_test)
        print(f"Saved arrays to {out.resolve()}")


if __name__ == "__main__":
    # Use local CSV:
    # processor = DataProcessor(r"C:\path\to\iris.csv")
    # Or auto-download:
    processor = DataProcessor()

    X_train, X_test, y_train, y_test = processor.process_full_pipeline()
    processor.save_processed_data(X_train, X_test, y_train, y_test)
