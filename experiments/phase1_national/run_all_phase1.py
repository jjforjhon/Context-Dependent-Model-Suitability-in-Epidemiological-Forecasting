"""
Phase 1 National Forecasting Experiment - Harmonized Protocol Version
All models evaluated under one-step-ahead rolling protocol with expanding window evaluation.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    CV_N_SPLITS, TEST_SIZE_PHASE1, FORECAST_HORIZON_PHASE1,
    RANDOM_STATE, LAG_FEATURES, ROLLING_WINDOWS,
    MODEL_GRIDS, BASE_FEATURES_PHASE1, TRENDS_FEATURES_PHASE1,
    TARGET_PHASE1, ROLLING_ORIGINS_PHASE1
)
from src.data.pipeline import DataPipeline
from src.models.ml_models import (
    fit_predict_xgboost, fit_predict_lightgbm, fit_predict_svr, fit_predict_mlp,
    perform_grid_search
)
from src.models.statistical import (
    fit_sarimax, forecast_sarimax_one_step,
    fit_ets, forecast_ets_one_step,
    fit_prophet, forecast_prophet_one_step
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Failure tracking
failure_log = []

def log_failure(model_name, step, origin, exception, context=""):
    """Log model failure with full context."""
    failure_log.append({
        "model": model_name,
        "origin": origin,
        "step": step,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback": traceback.format_exc(),
        "context": context,
        "timestamp": datetime.now().isoformat()
    })
    logger.error(f"Model {model_name} failed at origin {origin}, step {step}: {exception}")

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, sMAPE."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred)
    }

def drift_forecast(y_train, h):
    """Drift method: linear extrapolation from first to last training point."""
    if len(y_train) < 2:
        return np.full(h, y_train[-1] if len(y_train) > 0 else 0)
    slope = (y_train[-1] - y_train[0]) / (len(y_train) - 1)
    last = y_train[-1]
    return np.array([last + slope * (i + 1) for i in range(h)])

def seasonal_naive_forecast(y_train, h, seasonal_period=12):
    """Seasonal naive: repeat the value from same season last year."""
    if len(y_train) < seasonal_period:
        return np.full(h, y_train[-1] if len(y_train) > 0 else 0)
    last_season = y_train[-seasonal_period:]
    reps = int(np.ceil(h / seasonal_period))
    return np.tile(last_season, reps)[:h]

def zeros_forecast(y_train, h):
    """All-zeros baseline."""
    return np.zeros(h)

def rolling_origin_forecast(model_func, y_train, X_train, X_test, y_test, exog_train=None, exog_test=None, 
                            forecast_horizon=12, n_origins=5, model_name="model"):
    """
    Expanding window rolling-origin evaluation.
    Returns predictions for each origin and aggregate metrics.
    """
    n_test = len(y_test)
    all_predictions = []
    all_actuals = []
    origin_metrics = []
    
    # Calculate origin points - evenly spaced through test period
    if n_origins >= n_test:
        origin_points = list(range(n_test))
    else:
        origin_points = np.linspace(0, n_test - 1, n_origins, dtype=int).tolist()
    
    for origin_idx, origin in enumerate(origin_points):
        logger.info(f"  Origin {origin_idx+1}/{len(origin_points)}: test index {origin}")
        
        # Training data up to this origin
        y_hist = np.concatenate([y_train, y_test[:origin]]) if origin > 0 else y_train
        X_hist = np.vstack([X_train, X_test[:origin]]) if origin > 0 else X_train
        
        if exog_train is not None:
            exog_hist = np.vstack([exog_train, exog_test[:origin]]) if origin > 0 else exog_train
        else:
            exog_hist = None
        
        # Test window from this origin
        h_remaining = min(forecast_horizon, n_test - origin)
        y_true_window = y_test[origin:origin + h_remaining]
        X_test_window = X_test[origin:origin + h_remaining]
        
        if exog_test is not None:
            exog_test_window = exog_test[origin:origin + h_remaining]
        else:
            exog_test_window = None
        
        # One-step-ahead predictions for this window
        preds = []
        for step in range(h_remaining):
            try:
                # Current history for this step
                y_current = np.concatenate([y_hist, y_true_window[:step]]) if step > 0 else y_hist
                X_current = np.vstack([X_hist, X_test_window[:step]]) if step > 0 else X_hist
                if exog_hist is not None:
                    exog_current = np.vstack([exog_hist, exog_test_window[:step]]) if step > 0 else exog_hist
                else:
                    exog_current = None
                
                # Next step features (for ML models using lagged targets)
                X_next = X_test_window[step:step+1]
                               # Provide time T information exactly matching X_next for equivalent evaluation
                exog_next = exog_test_window[step:step+1] if exog_test_window is not None else None
                
                pred = model_func(
                    y_current, X_current, X_next, 
                    exog_current, exog_next,
                    model_name=f"{model_name}_origin{origin_idx}_step{step}"
                )
                preds.append(pred)
            except Exception as e:
                log_failure(model_name, step, origin_idx, e, f"Rolling origin prediction failed")
                preds.append(0.0)  # Fallback
        
        preds = np.array(preds)
        preds = np.maximum(preds, 0)  # Clip at zero
        
        all_predictions.extend(preds)
        all_actuals.extend(y_true_window)
        
        # Metrics for this origin
        if len(y_true_window) > 0:
            origin_metrics.append(calculate_metrics(y_true_window, preds))
    
    return {
        "predictions": np.array(all_predictions),
        "actuals": np.array(all_actuals),
        "origin_metrics": origin_metrics,
        "aggregate_metrics": calculate_metrics(np.array(all_actuals), np.array(all_predictions))
    }

# ============================================================
# MODEL WRAPPERS FOR ONE-STEP-AHEAD FORECASTING
# ============================================================

def ml_model_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", model_obj=None, feature_names=None):
    """One-step-ahead prediction for ML models using fitted model object."""
    if model_obj is None:
        raise ValueError("model_obj required for ML models")
    
    # Retrain iteratively to satisfy rolling origin requirements
    model_obj.fit(X_hist, y_hist)
    pred = model_obj.predict(X_next)
    
    # Save the model
    # model_name contains e.g. "XGBoost_Base_origin0_step0", we just save the latest state for the base architecture
    base_name = model_name.split("_origin")[0] if "_origin" in model_name else model_name
    model_path = Path("results/phase1/models") / f"Phase1_{base_name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, model_path)
    
    return float(pred[0])

def sarimax_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", order=(1,1,1), seasonal_order=(1,1,1,12)):
    """One-step-ahead SARIMAX forecast with exogenous variables."""
    try:
        model_fit = fit_sarimax(y_hist, exog_hist, order=order, seasonal_order=seasonal_order)
        pred = forecast_sarimax_one_step(model_fit, exog_next)
        base_name = model_name.split('_origin')[0] if '_origin' in model_name else model_name
        model_path = Path("results/phase1/models") / f"Phase1_{base_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_fit, model_path)
        return float(pred)
    except Exception as e:
        log_failure("SARIMAX", 0, 0, e, "SARIMAX one-step forecast failed")
        raise

def ets_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", seasonal_periods=12):
    """One-step-ahead ETS forecast (refit each step)."""
    try:
        model_fit = fit_ets(y_hist, seasonal_periods=seasonal_periods)
        pred = forecast_ets_one_step(model_fit)
        base_name = model_name.split('_origin')[0] if '_origin' in model_name else model_name
        model_path = Path("results/phase1/models") / f"Phase1_{base_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_fit, model_path)
        return float(pred)
    except Exception as e:
        log_failure("ETS", 0, 0, e, "ETS one-step forecast failed")
        raise

def prophet_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", feature_names=None):
    """One-step-ahead Prophet forecast with regressors."""
    try:
        # Build dataframe for Prophet
        dates_hist = pd.date_range(end=pd.Timestamp('2023-05-01'), periods=len(y_hist), freq='MS')
        df_hist = pd.DataFrame({'ds': dates_hist, 'y': y_hist})
        
        # Add regressors if available
        regressor_cols = []
        if exog_hist is not None and feature_names is not None:
            for i, name in enumerate(feature_names):
                if name.startswith('trends_') or name.startswith('temp_') or name.startswith('humidity_') or name.startswith('rainfall_'):
                    df_hist[name] = exog_hist[:, i] if exog_hist.ndim > 1 else exog_hist
                    regressor_cols.append(name)
        
        model_fit = fit_prophet(df_hist, regressor_cols=regressor_cols)
        
        # Next period
        next_date = dates_hist[-1] + pd.DateOffset(months=1)
        df_future = pd.DataFrame({'ds': [next_date]})
        if exog_next is not None and feature_names is not None:
            for i, name in enumerate(regressor_cols):
                df_future[name] = exog_next[:, feature_names.index(name)] if exog_next.ndim > 1 else exog_next
        
        pred = forecast_prophet_one_step(model_fit, df_future)
        base_name = model_name.split('_origin')[0] if '_origin' in model_name else model_name
        model_path = Path("results/phase1/models") / f"Phase1_{base_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_fit, model_path)
        return float(pred)
    except Exception as e:
        log_failure("Prophet", 0, 0, e, "Prophet one-step forecast failed")
        raise

def naive_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", baseline_type="zeros"):
    """One-step-ahead naive baselines."""
    h = 1
    if baseline_type == "drift":
        return float(drift_forecast(y_hist, h)[0])
    elif baseline_type == "seasonal_naive":
        return float(seasonal_naive_forecast(y_hist, h, seasonal_period=12)[0])
    else:  # zeros
        return 0.0

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("PHASE 1 NATIONAL FORECASTING - HARMONIZED PROTOCOL")
    logger.info("=" * 60)
    
    # Load processed data
    pipeline = DataPipeline()
    df = pipeline.load_processed_phase1()
    
    # Prepare features and target
    feature_cols_base = BASE_FEATURES_PHASE1
    feature_cols_trends = TRENDS_FEATURES_PHASE1
    target_col = TARGET_PHASE1
    
    # Drop rows with NaN in target
    df = df.dropna(subset=[target_col])
    
    # Split train/test chronologically
    train_size = len(df) - TEST_SIZE_PHASE1
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    X_train_base = df_train[feature_cols_base].fillna(0).values
    X_test_base = df_test[feature_cols_base].fillna(0).values
    X_train_trends = df_train[feature_cols_trends].fillna(0).values
    X_test_trends = df_test[feature_cols_trends].fillna(0).values
    
    logger.info(f"Train: {len(y_train)} months, Test: {len(y_test)} months")
    logger.info(f"Base features: {X_train_base.shape[1]}, Trends features: {X_train_trends.shape[1]}")
    
    # Results storage
    all_results = {}
    all_predictions = {}
    
    # ============================================================
    # ML MODELS WITH HYPERPARAMETER TUNING
    # ============================================================
    logger.info("\n--- ML Models (Base Features) ---")
    
    # Build ML pipelines
    ml_pipelines = {
        "XGBoost": Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(random_state=RANDOM_STATE, verbosity=0))
        ]),
        "LightGBM": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LGBMRegressor(random_state=RANDOM_STATE, verbose=-1))
        ]),
        "SVR": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR())
        ]),
        "MLP": Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(max_iter=1000, early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE))
        ])
    }
    
    for name, pipeline_obj in ml_pipelines.items():
        logger.info(f"\nTraining {name} (Base)...")
        
        param_grid = MODEL_GRIDS.get(name, {})
        
        # Hyperparameter tuning with expanded n_iter
        if param_grid:
            n_iter = min(50, max(20, len(list(param_grid.values())[0]) * len(param_grid)))
            search = RandomizedSearchCV(
                pipeline_obj, param_grid,
                n_iter=n_iter, cv=TimeSeriesSplit(n_splits=CV_N_SPLITS),
                scoring='neg_root_mean_squared_error',
                random_state=RANDOM_STATE, n_jobs=-1, verbose=0
            )
            search.fit(X_train_base, y_train)
            best_model = search.best_estimator_
            logger.info(f"  Best params: {search.best_params_}")
        else:
            best_model = clone(pipeline_obj)
            best_model.fit(X_train_base, y_train)
        
        # Rolling-origin evaluation
        def ml_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name=""):
            return ml_model_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, 
                                     model_name, model_obj=best_model)
        
        result = rolling_origin_forecast(
            ml_predict, y_train, X_train_base, X_test_base, y_test,
            forecast_horizon=FORECAST_HORIZON_PHASE1,
            n_origins=ROLLING_ORIGINS_PHASE1,
            model_name=f"{name}_Base"
        )
        
        all_results[f"{name}_Base"] = result["aggregate_metrics"]
        all_results[f"{name}_Base"]["origin_metrics"] = result["origin_metrics"]
        all_predictions[f"{name}_Base"] = result["predictions"]
        
        logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
        origin_rmses = [m['RMSE'] for m in result['origin_metrics']]
        logger.info(f"  Origin RMSEs: {[f'{x:.4f}' for x in origin_rmses]}")
    
    # ML Models with Trends features
    logger.info("\n--- ML Models (Trends Features) ---")
    for name, pipeline_obj in ml_pipelines.items():
        logger.info(f"\nTraining {name} (Trends)...")
        
        param_grid = MODEL_GRIDS.get(name, {})
        
        if param_grid:
            n_iter = min(50, max(20, len(list(param_grid.values())[0]) * len(param_grid)))
            search = RandomizedSearchCV(
                pipeline_obj, param_grid,
                n_iter=n_iter, cv=TimeSeriesSplit(n_splits=CV_N_SPLITS),
                scoring='neg_root_mean_squared_error',
                random_state=RANDOM_STATE, n_jobs=-1, verbose=0
            )
            search.fit(X_train_trends, y_train)
            best_model = search.best_estimator_
            logger.info(f"  Best params: {search.best_params_}")
        else:
            best_model = clone(pipeline_obj)
            best_model.fit(X_train_trends, y_train)
        
        def ml_predict_trends(y_hist, X_hist, X_next, exog_hist, exog_next, model_name=""):
            return ml_model_one_step(y_hist, X_hist, X_next, exog_hist, exog_next,
                                     model_name, model_obj=best_model)
        
        result = rolling_origin_forecast(
            ml_predict_trends, y_train, X_train_trends, X_test_trends, y_test,
            forecast_horizon=FORECAST_HORIZON_PHASE1,
            n_origins=ROLLING_ORIGINS_PHASE1,
            model_name=f"{name}_Trends"
        )
        
        all_results[f"{name}_Trends"] = result["aggregate_metrics"]
        all_results[f"{name}_Trends"]["origin_metrics"] = result["origin_metrics"]
        all_predictions[f"{name}_Trends"] = result["predictions"]
        
        logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
        origin_rmses = [m['RMSE'] for m in result['origin_metrics']]
        logger.info(f"  Origin RMSEs: {[f'{x:.4f}' for x in origin_rmses]}")
    
    # ============================================================
    # STATISTICAL MODELS - ONE-STEP-AHEAD WITH EXOG
    # ============================================================
    logger.info("\n--- Statistical Models (One-Step-Ahead) ---")
    
    # Prepare exogenous variables for statistical models
    exog_features = [c for c in feature_cols_trends if c not in ['year', 'month_sin', 'month_cos', 'is_covid']]
    exog_train = df_train[exog_features].fillna(0).values
    exog_test = df_test[exog_features].fillna(0).values
    
    # SARIMAX - one-step iterative with exog
    logger.info("\nTraining SARIMAX (one-step)...")
    try:
        def sarimax_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name=""):
            return sarimax_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name)
        
        result = rolling_origin_forecast(
            sarimax_predict, y_train, X_train_trends, X_test_trends, y_test,
            exog_train=exog_train, exog_test=exog_test,
            forecast_horizon=FORECAST_HORIZON_PHASE1,
            n_origins=ROLLING_ORIGINS_PHASE1,
            model_name="SARIMAX"
        )
        all_results["SARIMAX"] = result["aggregate_metrics"]
        all_results["SARIMAX"]["origin_metrics"] = result["origin_metrics"]
        all_predictions["SARIMAX"] = result["predictions"]
        logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
    except Exception as e:
        log_failure("SARIMAX", -1, -1, e, "SARIMAX setup failed")
        all_results["SARIMAX"] = {"RMSE": np.nan, "MAE": np.nan, "sMAPE": np.nan}
        all_predictions["SARIMAX"] = np.zeros(len(y_test))
    
    # ETS - one-step iterative (refit each step)
    logger.info("\nTraining ETS (one-step)...")
    try:
        def ets_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name=""):
            return ets_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name, seasonal_periods=12)
        
        result = rolling_origin_forecast(
            ets_predict, y_train, X_train_base, X_test_base, y_test,
            forecast_horizon=FORECAST_HORIZON_PHASE1,
            n_origins=ROLLING_ORIGINS_PHASE1,
            model_name="ETS"
        )
        all_results["ETS"] = result["aggregate_metrics"]
        all_results["ETS"]["origin_metrics"] = result["origin_metrics"]
        all_predictions["ETS"] = result["predictions"]
        logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
    except Exception as e:
        log_failure("ETS", -1, -1, e, "ETS setup failed")
        all_results["ETS"] = {"RMSE": np.nan, "MAE": np.nan, "sMAPE": np.nan}
        all_predictions["ETS"] = np.zeros(len(y_test))
    
    # Prophet - one-step iterative with regressors
    logger.info("\nTraining Prophet (one-step)...")
    try:
        def prophet_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name=""):
            return prophet_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name, exog_features)
        
        result = rolling_origin_forecast(
            prophet_predict, y_train, X_train_trends, X_test_trends, y_test,
            exog_train=exog_train, exog_test=exog_test,
            forecast_horizon=FORECAST_HORIZON_PHASE1,
            n_origins=ROLLING_ORIGINS_PHASE1,
            model_name="Prophet"
        )
        all_results["Prophet"] = result["aggregate_metrics"]
        all_results["Prophet"]["origin_metrics"] = result["origin_metrics"]
        all_predictions["Prophet"] = result["predictions"]
        logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
    except Exception as e:
        log_failure("Prophet", -1, -1, e, "Prophet setup failed")
        all_results["Prophet"] = {"RMSE": np.nan, "MAE": np.nan, "sMAPE": np.nan}
        all_predictions["Prophet"] = np.zeros(len(y_test))
    
    # ============================================================
    # NAIVE BASELINES
    # ============================================================
    logger.info("\n--- Naive Baselines ---")
    
    for baseline_name, baseline_type in [("Zeros", "zeros"), ("Seasonal_Naive", "seasonal_naive"), ("Drift", "drift")]:
        logger.info(f"\nEvaluating {baseline_name} baseline...")
        
        def naive_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name=""):
            return naive_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name, baseline_type)
        
        result = rolling_origin_forecast(
            naive_predict, y_train, X_train_base, X_test_base, y_test,
            forecast_horizon=FORECAST_HORIZON_PHASE1,
            n_origins=ROLLING_ORIGINS_PHASE1,
            model_name=baseline_name
        )
        
        all_results[baseline_name] = result["aggregate_metrics"]
        all_results[baseline_name]["origin_metrics"] = result["origin_metrics"]
        all_predictions[baseline_name] = result["predictions"]
        logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_dir = Path(__file__).parent.parent.parent / "results" / "phase1"
    metrics_dir = results_dir / "metrics"
    preds_dir = results_dir / "predictions"
    models_dir = results_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_rows = []
    for model_name, metrics in all_results.items():
        row = {"Model": model_name}
        row.update(metrics)
        if "origin_metrics" in metrics:
            # Add spread statistics
            origin_rmses = [m["RMSE"] for m in metrics["origin_metrics"] if not np.isnan(m["RMSE"])]
            if origin_rmses:
                row["RMSE_mean"] = np.mean(origin_rmses)
                row["RMSE_std"] = np.std(origin_rmses)
                row["RMSE_min"] = np.min(origin_rmses)
                row["RMSE_max"] = np.max(origin_rmses)
        metrics_rows.append(row)
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = metrics_dir / "phase1_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nSaved metrics to {metrics_path}")
    
    # Save predictions
    pred_df = pd.DataFrame({"Actual": result["actuals"]})
    for model_name, preds in all_predictions.items():
        pred_df[model_name] = preds
    
    preds_path = preds_dir / "phase1_preds.csv"
    pred_df.to_csv(preds_path, index=False)
    logger.info(f"Saved predictions to {preds_path}")
    
    # Save failure log
    failure_df = pd.DataFrame(failure_log)
    failure_path = metrics_dir / "phase1_failure_log.csv"
    failure_df.to_csv(failure_path, index=False)
    logger.info(f"Saved failure log to {failure_path} ({len(failure_log)} entries)")
    
    # Save model artifacts
    # (Would save fitted models here if needed)
    
    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 RESULTS SUMMARY (Harmonized One-Step Protocol)")
    logger.info("=" * 60)
    summary = metrics_df[["Model", "RMSE", "MAE", "sMAPE"]].sort_values("RMSE")
    logger.info(f"\n{summary.to_string(index=False)}")
    
    # ============================================================
    # GENERATE STATISTICAL ARTIFACTS (TABLES)
    # ============================================================
    try:
        from src.visualization.tables import table4_phase1_results, table7_phase1_ablation
        
        # 1. Generate Table 4 (Phase 1 Results)
        table4_phase1_results(metrics_df)
        
        # 2. Generate Table 7 (Ablation Study)
        base_df = metrics_df[metrics_df['Model'].str.endswith('_Base')].copy()
        base_df['Model'] = base_df['Model'].str.replace('_Base', '')
        trends_df = metrics_df[metrics_df['Model'].str.endswith('_Trends')].copy()
        trends_df['Model'] = trends_df['Model'].str.replace('_Trends', '')
        table7_phase1_ablation(base_df, trends_df)
        
        logger.info("Successfully generated Phase 1 metrics and Ablation artifacts.")
            
    except Exception as e:
        logger.error(f"Failed to generate statistical artifacts: {e}")
        
    # --- NEW CODE: GENERATE EDA TABLES (TABLE 1, 2, 3) ---
    try:
        logger.info("Generating EDA Tables (Tables 1, 2, 3)...")
        from src.visualization.tables import (
            table1_phase1_dataset_summary, table2_phase1_feature_description, table3_phase1_model_hyperparameters
        )
        from src.data.loader import (
            load_who_cases, load_climate_data, load_vaccination_data,
            load_population_data, load_google_trends
        )
        
        # Load raw datasets safely for Table 1
        df_cases_raw = load_who_cases(Path("data/Monthly_nation_measles/Measles Monthly Data 2012-2025 [WHO].xlsx"))
        df_climate_raw = load_climate_data(Path("data/monthly  division wise/climate/Division _Monthly_climate_2012-2025.csv"))
        df_vacc_raw = load_vaccination_data(Path("data/yearly vaccine covarage/bangladesh_national_coverage_2012_2025.csv"))
        df_pop_raw = load_population_data(Path("data/yearly population_division/bangladesh_population_density_2012_2026.csv"))
        df_trends_raw = load_google_trends(Path("data/google_trends/google_trends_monthly_2012_2025.csv"))
        
        # 1. Generate Table 1
        table1_phase1_dataset_summary(df_cases_raw, df_climate_raw, df_vacc_raw, df_pop_raw, df_trends_raw)
        
        # 2. Generate Table 2
        table2_phase1_feature_description(BASE_FEATURES_PHASE1)
        
        # 3. Generate Table 3
        table3_phase1_model_hyperparameters(MODEL_GRIDS)
        
        logger.info("Successfully generated EDA Tables (1, 2, 3).")
    except Exception as e:
        logger.error(f"Failed to generate EDA Tables: {e}")
        
    # ============================================================
    # GENERATE VISUALIZATION ARTIFACTS (FIGURES)
    # ============================================================
    try:
        from src.visualization.figures import (
            fig1_phase1_study_workflow, fig2_phase1_national_timeseries, fig3_phase1_seasonal_decomposition,
            fig5_phase1_actual_vs_predicted, fig6_phase1_model_comparison_bar, fig13_phase1_ablation_comparison
        )
        
        # fig1
        fig1_phase1_study_workflow()
        
        # fig2 & fig3
        fig2_phase1_national_timeseries(df)
        fig3_phase1_seasonal_decomposition(df)
        
        # fig5
        results_dict = {"actual": pd.Series(pred_df["Actual"].values)}
        for m in ["XGBoost_Trends", "ETS", "SARIMAX", "LightGBM_Base"]:
            if m in pred_df.columns:
                results_dict[m] = pd.Series(pred_df[m].values)
        fig5_phase1_actual_vs_predicted(results_dict)
        
        # fig6
        fig6_phase1_model_comparison_bar(metrics_df)
        
        # fig13
        fig13_phase1_ablation_comparison(base_df, trends_df)
        
        # --- NEW CODE: GENERATE DIAGNOSTIC FIGURES (FIG 4, 7, 8, 9) ---
        try:
            logger.info("Generating Diagnostic Figures (4, 7, 8, 9)...")
            from src.visualization.figures import (
                fig4_phase1_acf_pacf, fig7_phase1_feature_importance, fig8_phase1_shap_summary, fig9_phase1_residual_analysis
            )
            import joblib
            
            # 1. Fig 4: ACF/PACF of raw cases
            fig4_phase1_acf_pacf(df['measles_total'])
            
            # 2. Load trained XGBoost Base model for interpretability
            xgb_path = Path("results/phase1/models/Phase1_XGBoost_Base.joblib")
            if xgb_path.exists():
                xgb_pipeline = joblib.load(xgb_path)
                xgb_model = xgb_pipeline.named_steps['model']
                scaler = xgb_pipeline.named_steps['scaler']
                
                # Reconstruct scaled training data
                X_train_scaled = scaler.transform(df[feature_cols_base].fillna(0).values)
                X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols_base)
                
                # 3. Fig 7: Feature Importance
                fig7_phase1_feature_importance(xgb_model.feature_importances_, feature_cols_base)
                
                # 4. Fig 8: SHAP Summary
                fig8_phase1_shap_summary(xgb_model, X_train_df, feature_cols_base)
            else:
                logger.warning(f"Could not find {xgb_path} for SHAP/Feature Importance.")
                
            # 5. Fig 9: Residual Analysis
            if "XGBoost_Base" in pred_df.columns and "Actual" in pred_df.columns:
                fig9_phase1_residual_analysis(pred_df["Actual"], pred_df["XGBoost_Base"], "XGBoost_Base")
            
            logger.info("Successfully generated Diagnostic Figures.")
        except Exception as inner_e:
            logger.error(f"Failed to generate Diagnostic Figures: {inner_e}")

        logger.info("Successfully generated Phase 1 visualization figures.")
    except Exception as e:
        logger.error(f"Failed to generate Phase 1 figures: {e}")
    
    return all_results, all_predictions, failure_log

if __name__ == "__main__":
    main()