"""
Handles loading, validating and preparing CSV data for training/prediction.
"""
import os
import glob
import shutil
import pandas as pd
from datetime import datetime
from src.config import TARGET_COLUMN, RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataValidationError(Exception):
    pass


def load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise DataValidationError("Uploaded CSV is empty.")
    return df


def validate_training_data(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(
            f"Training data must contain target column: '{TARGET_COLUMN}'"
        )
    if df.shape[0] < 5:
        raise DataValidationError("Need at least 5 rows to train.")
    if df.drop(columns=[TARGET_COLUMN]).select_dtypes(include="number").empty:
        raise DataValidationError("No numeric feature columns found.")


def split_features_target(df: pd.DataFrame):
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN]).select_dtypes(include="number")
    return X, y


def archive_processed_file(file_path: str) -> str:
    """Move a used training file from raw -> processed, timestamped."""
    filename = os.path.basename(file_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(PROCESSED_DATA_DIR, f"{ts}_{filename}")
    shutil.move(file_path, dest)
    return dest


def get_latest_raw_file() -> str | None:
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)