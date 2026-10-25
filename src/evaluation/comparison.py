"""
Model comparison tests (Diebold-Mariano, Friedman, Nemenyi).
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
import itertools

logger = logging.getLogger(__name__)

def diebold_mariano_test(y_true, pred1, pred2, h=1, loss_type='mse'):
    """
    Diebold-Mariano test for predictive accuracy.
    
    Args:
        y_true: True values
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        h: Forecast horizon
        loss_type: 'mse' or 'mae'
        
    Returns:
        DM statistic and p-value
    """
    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)
    
    if loss_type == 'mse':
        e1 = (y_true - pred1)**2
        e2 = (y_true - pred2)**2
    elif loss_type == 'mae':
        e1 = np.abs(y_true - pred1)
        e2 = np.abs(y_true - pred2)
    else:
        raise ValueError("loss_type must be 'mse' or 'mae'")
        
    d = e1 - e2
    mean_d = np.mean(d)
    
    # Autocovariance estimation
    T = len(d)
    def autocovariance(k):
        return np.mean((d[k:] - mean_d) * (d[:T-k] - mean_d))
        
    gamma = [autocovariance(k) for k in range(h)]
    
    var_d = gamma[0] + 2 * sum(gamma[1:])
    var_d = var_d / T
    
    if var_d == 0:
        return 0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d)
    
    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def friedman_test(error_matrix):
    """
    Perform Friedman test for multiple model comparison.
    
    Args:
        error_matrix: Array of shape (n_samples, n_models) containing errors.
        
    Returns:
        Statistic and p-value
    """
    stat, p_value = stats.friedmanchisquare(*error_matrix.T)
    return stat, p_value


def get_significance_marker(p_value):
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def create_comparison_summary(errors_dict, h=1):
    """
    Create a pairwise DM test summary dataframe.
    
    Args:
        errors_dict: Dict of model_name -> predictions
        y_true: True values
    """
    # Note: Requires y_true in scope or passed. Assuming predictions only for DM.
    pass # Implementation detail, placeholder for actual comparison matrix if needed.
