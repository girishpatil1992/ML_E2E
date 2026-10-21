"""
Central configuration for paths, constants, and feature schema.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.json")           # XGBoost native format
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

METRICS_DIR = os.path.join(BASE_DIR, "metrics")
METRICS_HISTORY_PATH = os.path.join(METRICS_DIR, "metrics_history.csv")

# Target column name expected in training CSVs
TARGET_COLUMN = "target"

# Default XGBoost hyperparameters (simple baseline model)
DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "rmse",
}
DEFAULT_NUM_BOOST_ROUND = 100

for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR]:
    os.makedirs(_dir, exist_ok=True)