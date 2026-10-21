"""
Evaluation metrics: R2, MAE, MSE.
Tracks model evolution across retraining runs in a CSV history log.
"""
import os
import csv
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.config import METRICS_HISTORY_PATH

HISTORY_FIELDS = [
    "timestamp",
    "version",
    "is_incremental",
    "rows_used",
    "r2_score",
    "mae",
    "mse",
]


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "r2_score": round(float(r2_score(y_true, y_pred)), 5),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 5),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 5),
    }


def evaluate_and_log(y_true, y_pred, version, is_incremental, n_rows_used) -> dict:
    metrics = compute_metrics(y_true, y_pred)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "version": version,
        "is_incremental": is_incremental,
        "rows_used": n_rows_used,
        **metrics,
    }

    file_exists = os.path.exists(METRICS_HISTORY_PATH)
    with open(METRICS_HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return metrics


def get_metrics_history() -> list:
    if not os.path.exists(METRICS_HISTORY_PATH):
        return []
    with open(METRICS_HISTORY_PATH, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_latest_metrics() -> dict | None:
    history = get_metrics_history()
    return history[-1] if history else None