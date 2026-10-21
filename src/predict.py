"""
Prediction logic using the saved XGBoost model.
"""
import os
import json
import xgboost as xgb
import pandas as pd
from src.config import MODEL_PATH, MODEL_METADATA_PATH


class ModelNotFoundError(Exception):
    pass


def _load_booster_and_schema():
    if not os.path.exists(MODEL_PATH):
        raise ModelNotFoundError(
            "No trained model found. Please train a model first."
        )
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return booster, metadata["feature_columns"]


def predict_single(params: dict) -> float:
    """
    params: dict of {feature_name: value}
    """
    booster, feature_columns = _load_booster_and_schema()

    missing = [c for c in feature_columns if c not in params]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    row = pd.DataFrame([{col: params[col] for col in feature_columns}])
    dmatrix = xgb.DMatrix(row, feature_names=feature_columns)
    prediction = booster.predict(dmatrix)
    return float(prediction[0])


def get_required_features() -> list:
    _, feature_columns = _load_booster_and_schema()
    return feature_columns