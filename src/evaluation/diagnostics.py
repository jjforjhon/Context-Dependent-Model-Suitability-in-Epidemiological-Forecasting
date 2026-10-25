"""
Diagnostic tests and plots for model residuals.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.graphics.gofplots as gofplots
import logging

logger = logging.getLogger(__name__)

def check_overfitting(train_rmse, test_rmse, threshold=0.5):
    """
    Check for potential overfitting based on train/test RMSE ratio.
    """
    ratio = train_rmse / test_rmse
    is_overfit = ratio < threshold
    if is_overfit:
        logger.warning(f"Potential overfitting detected! Train/Test RMSE ratio = {ratio:.3f} < {threshold}")
    return is_overfit, ratio


def residual_diagnostics(residuals, model_name="Model"):
    """
    Perform diagnostic tests on residuals.
    
    Args:
        residuals: Array of model residuals (y_true - y_pred)
        model_name: Name of the model
        
    Returns:
        Dict of diagnostic test results
    """
    residuals = np.asarray(residuals)
    
    # Ljung-Box test for autocorrelation (using default lags)
    lb_test = acorr_ljungbox(residuals, return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].min() # Min p-value across lags
    
    # Shapiro-Wilk test for normality
    sw_stat, sw_pvalue = stats.shapiro(residuals)
    
    # Durbin-Watson statistic
    dw_stat = durbin_watson(residuals)
    
    results = {
        "Ljung-Box_min_pvalue": lb_pvalue,
        "Shapiro-Wilk_pvalue": sw_pvalue,
        "Durbin-Watson": dw_stat
    }
    
    return results


def plot_residual_diagnostics(residuals, model_name="Model", save_path=None):
    """
    Generate standard diagnostic plots: ACF and Q-Q plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # ACF plot
    tsaplots.plot_acf(residuals, ax=axes[0], title=f"{model_name} Residual ACF")
    
    # Q-Q plot
    gofplots.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title(f"{model_name} Residual Q-Q Plot")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
