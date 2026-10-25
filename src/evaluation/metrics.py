"""
Metrics module for evaluating forecasting models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def mean_absolute_percentage_error_custom(y_true, y_pred):
    """Calculate MAPE only if no zeros in true values."""
    if (y_true == 0).any():
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate sMAPE."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Handle zeros in both true and pred
    diff = np.abs(y_true - y_pred)
    smape = np.where(denominator == 0, 0, diff / denominator)
    return np.mean(smape) * 100


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """Calculate MASE relative to naive forecast."""
    if y_train is None or len(y_train) < 2:
        return np.nan
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    if naive_mae == 0:
        return np.nan
    mae = mean_absolute_error(y_true, y_pred)
    return mae / naive_mae


def compute_all_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_train: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive set of forecasting metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training values (needed for MASE)
        
    Returns:
        Dict of metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error_custom(y_true, y_pred),
        "sMAPE": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred)
    }
    
    if y_train is not None:
        metrics["MASE"] = mean_absolute_scaled_error(y_true, y_pred, np.asarray(y_train))
        
    return metrics


def format_metrics_table(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Format metrics into a pandas DataFrame for publication.
    
    Args:
        results_dict: Dict with model names as keys and metric dicts as values.
        
    Returns:
        DataFrame with formatted metrics
    """
    df = pd.DataFrame(results_dict).T
    
    # Optionally round and format for presentation
    df = df.round(3)
    
    return df
