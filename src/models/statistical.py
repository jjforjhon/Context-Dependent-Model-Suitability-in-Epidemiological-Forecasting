"""
Statistical forecasting models for measles forecasting (SARIMA, ETS, Prophet).
Supports both multi-step and one-step-ahead iterative forecasting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

try:
    import pmdarima as pm
except ImportError:
    pm = None

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


# ============================================================
# ONE-STEP-AHEAD FUNCTIONS (for harmonized protocol)
# ============================================================

def fit_sarimax(y_hist, exog_hist, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Fit SARIMAX model on historical data with exogenous variables.
    
    Args:
        y_hist: Historical target values (numpy array)
        exog_hist: Historical exogenous variables (numpy array or None)
        order: ARIMA order (p,d,q)
        seasonal_order: Seasonal ARIMA order (P,D,Q,m)
        
    Returns:
        Fitted SARIMAX model
    """
    try:
        sarima = sm.tsa.statespace.SARIMAX(
            y_hist,
            exog=exog_hist,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarima_fit = sarima.fit(disp=False, method='lbfgs', maxiter=200)
        return sarima_fit
    except Exception as e:
        logger.error(f"SARIMAX fit failed: {e}")
        raise


def forecast_sarimax_one_step(model_fit, exog_next):
    """
    One-step-ahead forecast from fitted SARIMAX model.
    
    Args:
        model_fit: Fitted SARIMAX model
        exog_next: Exogenous variables for next step (numpy array, shape (1, n_exog) or None)
        
    Returns:
        Point forecast (float)
    """
    try:
        pred = model_fit.get_forecast(steps=1, exog=exog_next)
        # pred.predicted_mean may be numpy array or pandas Series
        pm = pred.predicted_mean
        return float(pm[0] if hasattr(pm, '__getitem__') else pm)
    except Exception as e:
        logger.error(f"SARIMAX one-step forecast failed: {e}")
        raise


def fit_ets(y_hist, seasonal_periods=12):
    """
    Fit ETS model on historical data.
    
    Args:
        y_hist: Historical target values (numpy array)
        seasonal_periods: Seasonal period
        
    Returns:
        Fitted ETS model
    """
    best_aic = np.inf
    best_fit = None
    best_params = {}
    
    trends = ['add', None]
    seasonals = ['add', None]
    
    for t in trends:
        for s in seasonals:
            try:
                if (t == 'mul' or s == 'mul') and (y_hist <= 0).any():
                    continue
                    
                model = ExponentialSmoothing(
                    y_hist,
                    trend=t,
                    seasonal=s,
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated"
                )
                fit_model = model.fit()
                if fit_model.aic < best_aic:
                    best_aic = fit_model.aic
                    best_fit = fit_model
                    best_params = {'trend': t, 'seasonal': s}
            except Exception:
                continue
    
    if best_fit is None:
        raise ValueError("ETS modeling failed for all parameter combinations")
    
    logger.info(f"Best ETS params: {best_params} with AIC: {best_aic}")
    return best_fit


def forecast_ets_one_step(model_fit):
    """
    One-step-ahead forecast from fitted ETS model.
    
    Args:
        model_fit: Fitted ETS model
        
    Returns:
        Point forecast (float)
    """
    try:
        pred = model_fit.forecast(steps=1)
        return float(pred.iloc[0] if hasattr(pred, 'iloc') else pred[0])
    except Exception as e:
        logger.error(f"ETS one-step forecast failed: {e}")
        raise


def fit_prophet(df_hist, regressor_cols=None):
    """
    Fit Prophet model on historical data with regressors.
    
    Args:
        df_hist: DataFrame with 'ds' and 'y' columns, plus regressor columns
        regressor_cols: List of regressor column names
        
    Returns:
        Fitted Prophet model
    """
    if Prophet is None:
        raise ImportError("Prophet is not installed")
    
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        if regressor_cols:
            for col in regressor_cols:
                model.add_regressor(col)
        
        model.fit(df_hist)
        return model
    except Exception as e:
        logger.error(f"Prophet fit failed: {e}")
        raise


def forecast_prophet_one_step(model_fit, df_future):
    """
    One-step-ahead forecast from fitted Prophet model.
    
    Args:
        model_fit: Fitted Prophet model
        df_future: DataFrame with 'ds' column and regressor columns for next step
        
    Returns:
        Point forecast (float)
    """
    try:
        forecast = model_fit.predict(df_future)
        return float(forecast['yhat'].iloc[0])
    except Exception as e:
        logger.error(f"Prophet one-step forecast failed: {e}")
        raise


def fit_arima(y_hist, exog_hist, order=(1,1,1)):
    """
    Fit ARIMA/ARIMAX model on historical data.
    
    Args:
        y_hist: Historical target values (numpy array)
        exog_hist: Historical exogenous variables (numpy array or None)
        order: ARIMA order (p,d,q)
        
    Returns:
        Fitted ARIMA/ARIMAX model
    """
    try:
        model = sm.tsa.ARIMA(
            y_hist,
            exog=exog_hist,
            order=order
        )
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        logger.error(f"ARIMA fit failed: {e}")
        raise


def forecast_arima_one_step(model_fit, exog_next):
    """
    One-step-ahead forecast from fitted ARIMA/ARIMAX model.
    
    Args:
        model_fit: Fitted ARIMA/ARIMAX model
        exog_next: Exogenous variables for next step (numpy array, shape (1, n_exog) or None)
        
    Returns:
        Point forecast (float)
    """
    try:
        pred = model_fit.get_forecast(steps=1, exog=exog_next)
        # pred.predicted_mean may be numpy array or pandas Series
        pm = pred.predicted_mean
        return float(pm[0] if hasattr(pm, '__getitem__') else pm)
    except Exception as e:
        logger.error(f"ARIMA one-step forecast failed: {e}")
        raise


# ============================================================
# MULTI-STEP FUNCTIONS (original, for reference)
# ============================================================

def fit_predict_sarima(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str, 
    exog_cols: Optional[list] = None
) -> Dict[str, Any]:
    """
    Fit and predict using SARIMA/SARIMAX (multi-step).
    
    Args:
        train_df: Training data
        test_df: Testing data
        target_col: Target column name
        exog_cols: Exogenous column names (e.g. Google Trends)
       
    Returns:
        Dict containing predictions, parameters, and metrics.
    """
    logger.info("Starting SARIMA modeling...")
    
    y_train = train_df[target_col]
    X_train = train_df[exog_cols] if exog_cols else None
    
    X_test = test_df[exog_cols] if exog_cols else None
    
    try:
        if pm is not None:
            model = pm.auto_arima(
                y_train, 
                exogenous=X_train,
                seasonal=True, m=12,
                suppress_warnings=True,
                error_action="ignore",
                random_state=RANDOM_SEED
            )
            order = model.order
            seasonal_order = model.seasonal_order
        else:
            # Fallback to manual selection
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)
           
        logger.info(f"Using SARIMA order: {order}, seasonal: {seasonal_order}")
       
        sarima = sm.tsa.statespace.SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarima_fit = sarima.fit(disp=False)
       
        preds = sarima_fit.get_forecast(steps=len(test_df), exog=X_test)
        pred_mean = preds.predicted_mean
        conf_int = preds.conf_int()
       
        train_preds = sarima_fit.fittedvalues
       
        return {
            "model_name": "SARIMAX" if exog_cols else "SARIMA",
            "predictions": pred_mean.values,
            "train_predictions": train_preds.values,
            "confidence_intervals": conf_int.values,
            "parameters": {"order": order, "seasonal_order": seasonal_order},
            "aic": sarima_fit.aic,
            "bic": sarima_fit.bic
        }
    except Exception as e:
        logger.error(f"SARIMA modeling failed: {e}")
        return None



def fit_predict_ets(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str
) -> Dict[str, Any]:
    """
    Fit and predict using Exponential Smoothing (ETS) (multi-step).
    
    Args:
        train_df: Training data
        test_df: Testing data
        target_col: Target column name
       
    Returns:
        Dict containing predictions and parameters.
    """
    logger.info("Starting ETS modeling...")
    y_train = train_df[target_col]
    
    best_aic = np.inf
    best_model = None
    best_fit = None
    best_params = {}
    
    trends = ['add', 'mul', None]
    seasonals = ['add', 'mul', None]
    
    for t in trends:
        for s in seasonals:
            try:
                # Additive or multiplicative seasonality requires strictly positive data for 'mul'
                # Let's handle it gracefully
                if (t == 'mul' or s == 'mul') and (y_train <= 0).any():
                    continue
                   
                model = ExponentialSmoothing(
                    y_train, 
                    trend=t, 
                    seasonal=s, 
                    seasonal_periods=12,
                    initialization_method="estimated"
                )
                fit_model = model.fit()
                if fit_model.aic < best_aic:
                    best_aic = fit_model.aic
                    best_model = model
                    best_fit = fit_model
                    best_params = {'trend': t, 'seasonal': s}
            except Exception:
                continue
               
    if best_fit is None:
        logger.error("ETS modeling failed for all parameter combinations.")
        return None
       
    logger.info(f"Best ETS params: {best_params} with AIC: {best_aic}")
    preds = best_fit.forecast(steps=len(test_df))
    train_preds = best_fit.fittedvalues
    
    return {
        "model_name": "ETS",
        "predictions": preds.values,
        "train_predictions": train_preds.values,
        "parameters": best_params,
        "aic": best_aic
    }


def fit_predict_prophet(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str,
    date_col: str,
    exog_cols: Optional[list] = None
) -> Dict[str, Any]:
    """
    Fit and predict using Prophet (multi-step).
    
    Args:
        train_df: Training data
        test_df: Testing data
        target_col: Target column name
        date_col: Datetime column name
        exog_cols: Exogenous column names
       
    Returns:
        Dict containing predictions and uncertainty intervals.
    """
    if Prophet is None:
        logger.error("Prophet is not installed.")
        return None
       
    logger.info("Starting Prophet modeling...")
    
    try:
        # Prepare data for Prophet
        prophet_train = pd.DataFrame({
            'ds': train_df[date_col],
            'y': train_df[target_col]
        })
       
        prophet_test = pd.DataFrame({
            'ds': test_df[date_col]
        })
       
        if exog_cols:
            for col in exog_cols:
                prophet_train[col] = train_df[col].values
                prophet_test[col] = test_df[col].values
               
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            changepoint_prior_scale=0.05  # Ideally tuned via CV
        )
       
        if exog_cols:
            for col in exog_cols:
                model.add_regressor(col)
               
        model.fit(prophet_train)
       
        # Predictions on train to get train_preds
        train_forecast = model.predict(prophet_train)
        train_preds = train_forecast['yhat'].values
       
        # Predictions on test
        test_forecast = model.predict(prophet_test)
        preds = test_forecast['yhat'].values
       
        return {
            "model_name": "Prophet",
            "predictions": preds,
            "train_predictions": train_preds,
            "uncertainty_intervals": {
                "lower": test_forecast['yhat_lower'].values,
                "upper": test_forecast['yhat_upper'].values
            },
            "parameters": {"changepoint_prior_scale": 0.05}
        }
    except Exception as e:
        logger.error(f"Prophet modeling failed: {e}")
        return None


import numpy as np
from scipy.stats import norm

def dm_test(actual, pred1, pred2, h=1, crit="MSE"):
    """Diebold-Mariano test for predictive accuracy."""
    actual = np.array(actual)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    
    if crit == "MSE":
        e1 = (actual - pred1)**2
        e2 = (actual - pred2)**2
    elif crit == "MAE":
        e1 = np.abs(actual - pred1)
        e2 = np.abs(actual - pred2)
    else:
        raise ValueError("crit must be MSE or MAE")
    
    d = e1 - e2
    mean_d = np.mean(d)
    
    n = len(d)
    gamma0 = np.var(d, ddof=1)
    gamma = [gamma0]
    
    for lag in range(1, h):
        if lag < n:
            cov = np.cov(d[lag:], d[:-lag])[0,1]
            gamma.append(cov)
        else:
            gamma.append(0)
    
    var_d = gamma[0] + 2 * sum(gamma[1:])
    
    if var_d <= 0:
        return np.nan, np.nan
    
    stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - norm.cdf(abs(stat)))
    
    return stat, p_value
