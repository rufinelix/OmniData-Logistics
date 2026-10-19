"""Predictive machine learning engine for transit delay forecasting.

Trains an XGBoost regression pipeline on historical shipment telemetry to predict
real-time actual delay hours based on trip geometry, carrier performance, and environmental indices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_shipments.csv"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "ml_delay_model.joblib"

FEATURE_NUMERIC = ["distance_km", "cargo_weight_kg", "weather_risk_index", "traffic_delay_minutes", "day_of_week"]
FEATURE_CATEGORICAL = ["carrier_id"]
TARGET_COL = "actual_delay_hours"

_CACHED_MODEL_PIPELINE: Pipeline | None = None


def build_pipeline() -> Pipeline:
    """Build a scikit-learn preprocessing and XGBoost regressor pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_CATEGORICAL),
        ],
        remainder="drop",
    )

    regressor = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target vector y from shipments DataFrame."""
    df_clean = df.copy()

    # Extract day of week if not already present
    if "day_of_week" not in df_clean.columns:
        if "creation_timestamp" in df_clean.columns:
            creation_dt = pd.to_datetime(df_clean["creation_timestamp"], errors="coerce")
            df_clean["day_of_week"] = creation_dt.dt.dayofweek.fillna(2).astype(int)
        else:
            df_clean["day_of_week"] = 2

    # Drop missing target values
    df_clean = df_clean.dropna(subset=[TARGET_COL])
    X = df_clean[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y = df_clean[TARGET_COL].astype(float)
    return X, y


def train_and_save_delay_model(
    data_path: Path | str | None = None,
    model_output_path: Path | str | None = None,
) -> Pipeline:
    """Train XGBoost model on shipment data and save artifact with joblib.

    Args:
        data_path: Path to synthetic_shipments.csv.
        model_output_path: Destination path for saved joblib pipeline.

    Returns:
        Pipeline: Trained scikit-learn pipeline.
    """
    in_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    out_path = Path(model_output_path) if model_output_path else DEFAULT_MODEL_PATH

    if not in_path.exists():
        raise FileNotFoundError(f"Shipments dataset not found at {in_path}")

    logger.info("Loading shipment training data from %s...", in_path)
    df = pd.read_csv(in_path)
    X, y = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    pipeline = build_pipeline()
    logger.info("Fitting XGBoost delay prediction pipeline on %d training rows...", len(X_train))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info("Evaluation metrics - MAE: %.3f hrs, R2: %.3f", mae, r2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    logger.info("Successfully persisted trained model to %s", out_path)

    global _CACHED_MODEL_PIPELINE
    _CACHED_MODEL_PIPELINE = pipeline
    return pipeline


def load_delay_model(model_path: Path | str | None = None) -> Pipeline:
    """Load the trained delay model from disk.

    Args:
        model_path: Path to serialized joblib file.

    Returns:
        Pipeline: Loaded scikit-learn pipeline.
    """
    global _CACHED_MODEL_PIPELINE
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if _CACHED_MODEL_PIPELINE is not None and model_path is None:
        return _CACHED_MODEL_PIPELINE

    if not path.exists():
        logger.warning("Serialized model not found at %s. Triggering on-demand training.", path)
        return train_and_save_delay_model(model_output_path=path)

    _CACHED_MODEL_PIPELINE = joblib.load(path)
    return _CACHED_MODEL_PIPELINE


def predict_delay(
    carrier_id: str,
    distance_km: float,
    cargo_weight_kg: float,
    weather_risk_index: float,
    traffic_delay_minutes: float,
    day_of_week: int = 2,
    model_path: Path | str | None = None,
) -> float:
    """Predict expected delay in hours for a prospective shipment quote.

    Args:
        carrier_id: Identifier of the carrier (e.g. 'CAR-001').
        distance_km: Route transit distance in kilometers.
        cargo_weight_kg: Payload weight in kilograms.
        weather_risk_index: Weather severity index (0.0 to 3.0+).
        traffic_delay_minutes: Anticipated congestion delay in minutes.
        day_of_week: Day of week index (0=Monday, 6=Sunday).
        model_path: Optional custom path to model artifact.

    Returns:
        float: Predicted delay in hours (positive = delayed, negative = early).
    """
    pipeline = load_delay_model(model_path)
    sample_df = pd.DataFrame([{
        "distance_km": float(distance_km),
        "cargo_weight_kg": float(cargo_weight_kg),
        "weather_risk_index": float(weather_risk_index),
        "traffic_delay_minutes": float(traffic_delay_minutes),
        "day_of_week": int(day_of_week),
        "carrier_id": str(carrier_id),
    }])

    pred = float(pipeline.predict(sample_df)[0])
    return round(pred, 2)


if __name__ == "__main__":
    train_and_save_delay_model()
