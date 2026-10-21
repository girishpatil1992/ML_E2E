"""
Incremental XGBoost training.
If a previous model exists, it is loaded and training continues
(warm start) using ONLY the new data provided.
Otherwise, a fresh model is trained.
"""
import os
import json
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_PATH,
    MODEL_METADATA_PATH,
    DEFAULT_XGB_PARAMS,
    DEFAULT_NUM_BOOST_ROUND,
)
from src.data_handler import validate_training_data, split_features_target
from src.evaluate import evaluate_and_log


def _load_metadata() -> dict:
    if os.path.exists(MODEL_METADATA_PATH):
        with open(MODEL_METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_metadata(feature_columns: list, version: int):
    meta = {
        "feature_columns": feature_columns,
        "version": version,
        "xgb_params": DEFAULT_XGB_PARAMS,
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def train_model(
    df: pd.DataFrame,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    test_size: float = 0.2,
) -> dict:
    """
    Train (or incrementally continue training) the XGBoost model
    using the provided dataframe only.

    Returns a dict with evaluation metrics + metadata.
    """
    validate_training_data(df)
    X, y = split_features_target(df)
    feature_columns = list(X.columns)

    # Train/test split for evaluation purposes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    previous_model = None
    version = 1
    is_incremental = False

    metadata = _load_metadata()

    if os.path.exists(MODEL_PATH):
        # Load previous booster for warm start (continued training)
        previous_model = xgb.Booster()
        previous_model.load_model(MODEL_PATH)
        is_incremental = True
        version = metadata.get("version", 1) + 1

        # Ensure feature consistency; align new data to previous schema if possible
        prev_features = metadata.get("feature_columns", feature_columns)
        missing = [c for c in prev_features if c not in feature_columns]
        if missing:
            raise ValueError(
                f"New data is missing columns required by existing model: {missing}"
            )
        # Use the model's known feature order
        feature_columns = prev_features
        X_train = X_train[feature_columns]
        X_test = X_test[feature_columns]
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    booster = xgb.train(
        params=DEFAULT_XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        xgb_model=previous_model,  # None => fresh train; else continues training
        evals=[(dtrain, "train"), (dtest, "eval")],
        verbose_eval=False,
    )

    # Save model (overwrite with updated/continued model)
    booster.save_model(MODEL_PATH)
    _save_metadata(feature_columns, version)

    # Evaluate & log metrics history
    y_pred = booster.predict(dtest)
    metrics = evaluate_and_log(
        y_true=y_test,
        y_pred=y_pred,
        version=version,
        is_incremental=is_incremental,
        n_rows_used=df.shape[0],
    )

    return {
        "version": version,
        "is_incremental": is_incremental,
        "rows_used": df.shape[0],
        "feature_columns": feature_columns,
        "metrics": metrics,
    }