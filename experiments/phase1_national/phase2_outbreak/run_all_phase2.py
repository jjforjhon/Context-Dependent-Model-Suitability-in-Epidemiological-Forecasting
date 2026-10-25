"""
Phase 2 Outbreak Forecasting Experiment - Harmonized Protocol Version
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
    CV_N_SPLITS, TEST_SIZE_PHASE2, FORECAST_HORIZON_PHASE2,
    RANDOM_STATE, LAG_FEATURES, ROLLING_WINDOWS,
    MODEL_GRIDS, BASE_FEATURES_PHASE2, TRENDS_FEATURES_PHASE2,
    TARGET_PHASE2, ROLLING_ORIGINS_PHASE2
)
from src.data.pipeline import DataPipeline
from src.models.statistical import (
    fit_arima, forecast_arima_one_step,
    fit_ets, forecast_ets_one_step
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Failure tracking
failure_log = []

def log_failure(model_name, step, origin, division, exception, context=""):
    """Log model failure with full context."""
    failure_log.append({
        "model": model_name,
        "division": division,
        "origin": origin,
        "step": step,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback": traceback.format_exc(),
        "context": context,
        "timestamp": datetime.now().isoformat()
    })
    logger.error(f"Model {model_name} failed at division {division}, origin {origin}, step {step}: {exception}")

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

def seasonal_naive_forecast(y_train, h, seasonal_period=7):
    """Seasonal naive: repeat the value from same season last week."""
    if len(y_train) < seasonal_period:
        return np.full(h, y_train[-1] if len(y_train) > 0 else 0)
    last_season = y_train[-seasonal_period:]
    reps = int(np.ceil(h / seasonal_period))
    return np.tile(last_season, reps)[:h]

def zeros_forecast(y_train, h):
    """All-zeros baseline."""
    return np.zeros(h)

def rolling_origin_forecast(model_func, y_train, X_train, X_test, y_test, exog_train=None, exog_test=None, 
                            forecast_horizon=30, n_origins=5, model_name="model", division="division"):
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
        logger.info(f"  [{division}] Origin {origin_idx+1}/{len(origin_points)}: test index {origin}")
        
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
                
                # Next step features
                X_next = X_test_window[step:step+1]
                                # Provide time T information exactly matching X_next for equivalent evaluation
                exog_next = exog_test_window[step:step+1] if exog_test_window is not None else None
                
                pred = model_func(
                    y_current, X_current, X_next, 
                    exog_current, exog_next,
                    model_name=f"{model_name}_{division}_origin{origin_idx}_step{step}",
                    division=division
                )
                preds.append(pred)
            except Exception as e:
                log_failure(model_name, step, origin_idx, division, e, f"Rolling origin prediction failed")
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

def ml_model_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", model_obj=None, feature_names=None, division=None):
    """One-step-ahead prediction for ML models using fitted model object."""
    if model_obj is None:
        raise ValueError("model_obj required for ML models")
    
    # Retrain iteratively to satisfy rolling origin requirements
    model_obj.fit(X_hist, y_hist)
    pred = model_obj.predict(X_next)
    
    # Save the model
    base_name = model_name.split('_origin')[0] if '_origin' in model_name else model_name
    model_path = Path("results/phase2/models") / f"{base_name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, model_path)
    
    return float(pred[0])

def arimax_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None, order=(1,1,1)):
    """One-step-ahead ARIMAX forecast with exogenous variables."""
    try:
        model_fit = fit_arima(y_hist, exog_hist, order=order)
        pred = forecast_arima_one_step(model_fit, exog_next)
        base_name = model_name.split('_origin')[0] if '_origin' in model_name else model_name
        model_path = Path("results/phase2/models") / f"{base_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_fit, model_path)
        return float(pred)
    except Exception as e:
        log_failure("ARIMAX", 0, 0, division, e, "ARIMAX one-step forecast failed")
        raise

def ets_one_step_div(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None, seasonal_periods=7):
    """One-step-ahead ETS forecast (refit each step) for daily data."""
    try:
        model_fit = fit_ets(y_hist, seasonal_periods=seasonal_periods)
        pred = forecast_ets_one_step(model_fit)
        base_name = model_name.split('_origin')[0] if '_origin' in model_name else model_name
        model_path = Path("results/phase2/models") / f"{base_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_fit, model_path)
        return float(pred)
    except Exception as e:
        log_failure("ETS", 0, 0, division, e, "ETS one-step forecast failed")
        raise

def naive_one_step_div(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None, baseline_type="zeros"):
    """One-step-ahead naive baselines."""
    h = 1
    if baseline_type == "drift":
        return float(drift_forecast(y_hist, h)[0])
    elif baseline_type == "seasonal_naive":
        return float(seasonal_naive_forecast(y_hist, h, seasonal_period=7)[0])
    else:  # zeros
        return 0.0

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("PHASE 2 OUTBREAK FORECASTING - HARMONIZED PROTOCOL")
    logger.info("=" * 60)
    
    # Load processed data
    pipeline = DataPipeline()
    df = pipeline.load_processed_phase2()
    
    # Prepare features and target
    feature_cols = TRENDS_FEATURES_PHASE2  # Superset to extract all columns
    target_col = TARGET_PHASE2
    
    # Get unique divisions
    divisions = df['Division'].unique()
    logger.info(f"Divisions: {divisions}")
    
    all_results = {}
    all_predictions = {}
    all_actuals = {}
    division_results = {}
    
    for division in divisions:
        logger.info(f"\n{'='*60}")
        logger.info(f"DIVISION: {division}")
        logger.info(f"{'='*60}")
        
        df_div = df[df['Division'] == division].copy().sort_values('Date')
        
        # Drop rows with NaN in target or features
        df_div = df_div.dropna(subset=[target_col] + feature_cols)
        
        if len(df_div) < 30:
            logger.warning(f"Division {division}: insufficient data ({len(df_div)} rows), skipping")
            continue
        
        y = df_div[target_col].values
        X = df_div[feature_cols].fillna(0).values
        
                # Split train/test chronologically using master configuration
        train_size = len(df_div) - TEST_SIZE_PHASE2
        
        y_train = y[:train_size]
        y_test = y[train_size:]
        X_train = X[:train_size]
        X_test = X[train_size:]
        
        logger.info(f"Train: {len(y_train)} days, Test: {len(y_test)} days")
        
        # Subset Base features for Stage 3
        base_feature_idx = [feature_cols.index(c) for c in BASE_FEATURES_PHASE2]
        X_train_base = X_train[:, base_feature_idx]
        X_test_base = X_test[:, base_feature_idx]
        div_results = {}
        div_predictions = {}
        
        # ============================================================
        # STAGE 3: BASE CONFIGURATION EXECUTION LOOP
        # ============================================================
        logger.info(f"\n--- BASE CONFIGURATION MODELS for {division} ---")
        
        # 1. Base Feature Extraction
        base_exog_cols = [
            'Temperature_C', 'Humidity_Percent', 'Daily_Rainfall_mm',
            'Population', 'Area_sq_km', 'Density_per_sq_km', 'division_encoded'
        ]
        base_exog_idx = [feature_cols.index(c) for c in base_exog_cols if c in feature_cols]
        exog_train_base = X_train[:, base_exog_idx] if len(base_exog_idx) > 0 else None
        exog_test_base = X_test[:, base_exog_idx] if len(base_exog_idx) > 0 else None
        
        # Build Base ML pipelines
        ml_pipelines_base = {
            "XGBoost_Base": Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBRegressor(random_state=RANDOM_STATE, verbosity=0))
            ]),
            "LightGBM_Base": Pipeline([
                ('scaler', StandardScaler()),
                ('model', LGBMRegressor(random_state=RANDOM_STATE, verbose=-1))
            ])
        }
        
        for name, pipeline_obj in ml_pipelines_base.items():
            logger.info(f"\nTraining {name} for {division}...")
            
            # The hyperparameter grid keys are just "XGBoost" and "LightGBM"
            base_model_name = name.split('_')[0]
            param_grid = MODEL_GRIDS.get(base_model_name, {})
            
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
            
            def ml_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None):
                return ml_model_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, 
                                         model_name, model_obj=best_model, division=division)
            
            result = rolling_origin_forecast(
                ml_predict, y_train, X_train_base, X_test_base, y_test,
                forecast_horizon=FORECAST_HORIZON_PHASE2,
                n_origins=min(ROLLING_ORIGINS_PHASE2, len(y_test)),
                model_name=name,
                division=division
            )
            
            model_key = f"{name}_{division}"
            div_results[model_key] = result["aggregate_metrics"]
            div_results[model_key]["origin_metrics"] = result["origin_metrics"]
            div_predictions[model_key] = result["predictions"]
            logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
        
        # 2. ARIMAX Base
        logger.info(f"\nTraining ARIMAX_Base for {division}...")
        try:
            def arimax_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None):
                return arimax_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name, division=division)
            
            result = rolling_origin_forecast(
                arimax_predict, y_train, X_train_base, X_test_base, y_test,
                exog_train=exog_train_base, exog_test=exog_test_base,
                forecast_horizon=FORECAST_HORIZON_PHASE2,
                n_origins=min(ROLLING_ORIGINS_PHASE2, len(y_test)),
                model_name="ARIMAX_Base",
                division=division
            )
            model_key = f"ARIMAX_Base_{division}"
            div_results[model_key] = result["aggregate_metrics"]
            div_results[model_key]["origin_metrics"] = result["origin_metrics"]
            div_predictions[model_key] = result["predictions"]
            logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
        except Exception as e:
            log_failure("ARIMAX_Base", -1, -1, division, e, "ARIMAX Base setup failed")
            div_results[f"ARIMAX_Base_{division}"] = {"RMSE": np.nan, "MAE": np.nan, "sMAPE": np.nan}
            div_predictions[f"ARIMAX_Base_{division}"] = np.zeros(len(y_test))
            
        # 3. ETS Base (Univariate)
        logger.info(f"\nTraining ETS_Base (Univariate) for {division}...")
        try:
            def ets_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None):
                # Pass None for exog to strictly enforce univariate modeling
                return ets_one_step_div(y_hist, X_hist, X_next, None, None, model_name, division=division, seasonal_periods=7)
            
            result = rolling_origin_forecast(
                ets_predict, y_train, X_train_base, X_test_base, y_test,
                forecast_horizon=FORECAST_HORIZON_PHASE2,
                n_origins=min(ROLLING_ORIGINS_PHASE2, len(y_test)),
                model_name="ETS_Base",
                division=division
            )
            model_key = f"ETS_Base_{division}"
            div_results[model_key] = result["aggregate_metrics"]
            div_results[model_key]["origin_metrics"] = result["origin_metrics"]
            div_predictions[model_key] = result["predictions"]
            logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
        except Exception as e:
            log_failure("ETS_Base", -1, -1, division, e, "ETS Base setup failed")
            div_results[f"ETS_Base_{division}"] = {"RMSE": np.nan, "MAE": np.nan, "sMAPE": np.nan}
            div_predictions[f"ETS_Base_{division}"] = np.zeros(len(y_test))
        
        
        # ============================================================
        # STAGE 4: TRENDS CONFIGURATION EXECUTION LOOP
        # ============================================================
        logger.info(f"\n--- TRENDS CONFIGURATION MODELS for {division} ---")
        
        # 1. Trends Feature Extraction for ARIMAX
        trends_exog_cols = [
            'Temperature_C', 'Humidity_Percent', 'Daily_Rainfall_mm',
            'Population', 'Area_sq_km', 'Density_per_sq_km', 'division_encoded',
            'trends_lag_1', 'trends_lag_7', 'trends_lag_14'
        ]
        trends_exog_idx = [feature_cols.index(c) for c in trends_exog_cols if c in feature_cols]
        exog_train_trends = X_train[:, trends_exog_idx] if len(trends_exog_idx) > 0 else None
        exog_test_trends = X_test[:, trends_exog_idx] if len(trends_exog_idx) > 0 else None
        
        # Build Trends ML pipelines
        ml_pipelines_trends = {
            "XGBoost_Trends": Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBRegressor(random_state=RANDOM_STATE, verbosity=0))
            ]),
            "LightGBM_Trends": Pipeline([
                ('scaler', StandardScaler()),
                ('model', LGBMRegressor(random_state=RANDOM_STATE, verbose=-1))
            ])
        }
        
        for name, pipeline_obj in ml_pipelines_trends.items():
            logger.info(f"\nTraining {name} for {division}...")
            
            trends_model_name = name.split('_')[0]
            param_grid = MODEL_GRIDS.get(trends_model_name, {})
            
            if param_grid:
                n_iter = min(50, max(20, len(list(param_grid.values())[0]) * len(param_grid)))
                search = RandomizedSearchCV(
                    pipeline_obj, param_grid,
                    n_iter=n_iter, cv=TimeSeriesSplit(n_splits=CV_N_SPLITS),
                    scoring='neg_root_mean_squared_error',
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                logger.info(f"  Best params: {search.best_params_}")
            else:
                best_model = clone(pipeline_obj)
                best_model.fit(X_train, y_train)
            
            def ml_predict_trends(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None):
                return ml_model_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, 
                                         model_name, model_obj=best_model, division=division)
            
            result = rolling_origin_forecast(
                ml_predict_trends, y_train, X_train, X_test, y_test,
                forecast_horizon=FORECAST_HORIZON_PHASE2,
                n_origins=min(ROLLING_ORIGINS_PHASE2, len(y_test)),
                model_name=name,
                division=division
            )
            
            model_key = f"{name}_{division}"
            div_results[model_key] = result["aggregate_metrics"]
            div_results[model_key]["origin_metrics"] = result["origin_metrics"]
            div_predictions[model_key] = result["predictions"]
            logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
            
        # 2. ARIMAX Trends
        logger.info(f"\nTraining ARIMAX_Trends for {division}...")
        try:
            def arimax_predict_trends(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None):
                return arimax_one_step(y_hist, X_hist, X_next, exog_hist, exog_next, model_name, division=division)
            
            result = rolling_origin_forecast(
                arimax_predict_trends, y_train, X_train, X_test, y_test,
                exog_train=exog_train_trends, exog_test=exog_test_trends,
                forecast_horizon=FORECAST_HORIZON_PHASE2,
                n_origins=min(ROLLING_ORIGINS_PHASE2, len(y_test)),
                model_name="ARIMAX_Trends",
                division=division
            )
            model_key = f"ARIMAX_Trends_{division}"
            div_results[model_key] = result["aggregate_metrics"]
            div_results[model_key]["origin_metrics"] = result["origin_metrics"]
            div_predictions[model_key] = result["predictions"]
            logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
        except Exception as e:
            log_failure("ARIMAX_Trends", -1, -1, division, e, "ARIMAX Trends setup failed")
            div_results[f"ARIMAX_Trends_{division}"] = {"RMSE": np.nan, "MAE": np.nan, "sMAPE": np.nan}
            div_predictions[f"ARIMAX_Trends_{division}"] = np.zeros(len(y_test))
            
        # (ETS is intentionally skipped for Trends per methodology)
        
# ============================================================
        # NAIVE BASELINES
        # ============================================================
        logger.info(f"\n--- Naive Baselines for {division} ---")
        
        for baseline_name, baseline_type in [("Zeros", "zeros"), ("Seasonal_Naive", "seasonal_naive"), ("Drift", "drift")]:
            logger.info(f"\nEvaluating {baseline_name} baseline for {division}...")
            
            def naive_predict(y_hist, X_hist, X_next, exog_hist, exog_next, model_name="", division=None):
                return naive_one_step_div(y_hist, X_hist, X_next, exog_hist, exog_next, model_name, division=division, baseline_type=baseline_type)
            
            result = rolling_origin_forecast(
                naive_predict, y_train, X_train, X_test, y_test,
                forecast_horizon=FORECAST_HORIZON_PHASE2,
                n_origins=min(ROLLING_ORIGINS_PHASE2, len(y_test)),
                model_name=baseline_name,
                division=division
            )
            
            model_key = f"{baseline_name}_{division}"
            div_results[model_key] = result["aggregate_metrics"]
            div_results[model_key]["origin_metrics"] = result["origin_metrics"]
            div_predictions[model_key] = result["predictions"]
            logger.info(f"  Aggregate RMSE: {result['aggregate_metrics']['RMSE']:.4f}")
        
        division_results[division] = div_results
        all_predictions[division] = div_predictions
        all_actuals[division] = result["actuals"]
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_dir = Path(__file__).parent.parent.parent / "results" / "phase2"
    metrics_dir = results_dir / "metrics"
    preds_dir = results_dir / "predictions"
    models_dir = results_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-division metrics
    for division, results in division_results.items():
        metrics_rows = []
        for model_name, metrics in results.items():
            row = {"Model": model_name, "Division": division}
            row.update(metrics)
            if "origin_metrics" in metrics:
                origin_rmses = [m["RMSE"] for m in metrics["origin_metrics"] if not np.isnan(m["RMSE"])]
                if origin_rmses:
                    row["RMSE_mean"] = np.mean(origin_rmses)
                    row["RMSE_std"] = np.std(origin_rmses)
                    row["RMSE_min"] = np.min(origin_rmses)
                    row["RMSE_max"] = np.max(origin_rmses)
            metrics_rows.append(row)
        
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = metrics_dir / f"phase2_{division}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved {division} metrics to {metrics_path}")
    
    # Save aggregate metrics
    all_metrics_rows = []
    for division, results in division_results.items():
        for model_name, metrics in results.items():
            row = {"Model": model_name, "Division": division}
            row.update(metrics)
            if "origin_metrics" in metrics:
                origin_rmses = [m["RMSE"] for m in metrics["origin_metrics"] if not np.isnan(m["RMSE"])]
                if origin_rmses:
                    row["RMSE_mean"] = np.mean(origin_rmses)
                    row["RMSE_std"] = np.std(origin_rmses)
                    row["RMSE_min"] = np.min(origin_rmses)
                    row["RMSE_max"] = np.max(origin_rmses)
            all_metrics_rows.append(row)
    
    all_metrics_df = pd.DataFrame(all_metrics_rows)
    agg_metrics_path = metrics_dir / "phase2_metrics.csv"
    all_metrics_df.to_csv(agg_metrics_path, index=False)
    logger.info(f"\nSaved aggregate metrics to {agg_metrics_path}")
    
    # Save predictions
    for division, preds in all_predictions.items():
        pred_df = pd.DataFrame()
        pred_df["Actual"] = all_actuals[division]
        
        for model_name, pred_vals in preds.items():
            pred_df[model_name] = pred_vals
        
        preds_path = preds_dir / f"phase2_{division}_preds.csv"
        pred_df.to_csv(preds_path, index=False)
        logger.info(f"Saved {division} predictions to {preds_path}")
    
    # Save failure log
    failure_df = pd.DataFrame(failure_log)
    failure_path = metrics_dir / "phase2_failure_log.csv"
    failure_df.to_csv(failure_path, index=False)
    logger.info(f"Saved failure log to {failure_path} ({len(failure_log)} entries)")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 RESULTS SUMMARY (Harmonized One-Step Protocol)")
    logger.info("=" * 60)
    summary = all_metrics_df[["Division", "Model", "RMSE", "MAE", "sMAPE"]].sort_values(["Division", "RMSE"])
    logger.info(f"\n{summary.to_string(index=False)}")
    
    # ============================================================
    # STAGE 6: GENERATE BUILT-IN STATISTICAL ARTIFACTS (TABLES)
    # ============================================================
    try:
        from src.visualization.tables import (
            table5_phase2_results, 
            table8_phase2_XGBoost_Base_results,
            table9_phase2_LightGBM_Base_results, 
            table13_phase2_ARIMAX_Base_results,
            table14_phase2_ETS_Base_results,
            table10_phase2_XGBoost_Trends_results,
            table11_phase2_LightGBM_Trends_results,
            table12_phase2_ARIMAX_Trends_results
        )
        
        table5_phase2_results(all_metrics_df)
        
        # --- BASE TABLES ---
        xgb_base = all_metrics_df[all_metrics_df['Model'].str.startswith('XGBoost_Base')].copy()
        if not xgb_base.empty:
            xgb_base['Model'] = 'XGBoost_Base'
            table8_phase2_XGBoost_Base_results(xgb_base)
            
        lgb_base = all_metrics_df[all_metrics_df['Model'].str.startswith('LightGBM_Base')].copy()
        if not lgb_base.empty:
            lgb_base['Model'] = 'LightGBM_Base'
            table9_phase2_LightGBM_Base_results(lgb_base)
            
        ari_base = all_metrics_df[all_metrics_df['Model'].str.startswith('ARIMAX_Base')].copy()
        if not ari_base.empty:
            ari_base['Model'] = 'ARIMAX_Base'
            table13_phase2_ARIMAX_Base_results(ari_base)
            
        ets_base = all_metrics_df[all_metrics_df['Model'].str.startswith('ETS_Base')].copy()
        if not ets_base.empty:
            ets_base['Model'] = 'ETS_Base'
            table14_phase2_ETS_Base_results(ets_base)
            
        # --- TRENDS TABLES ---
        xgb_trends = all_metrics_df[all_metrics_df['Model'].str.startswith('XGBoost_Trends')].copy()
        if not xgb_trends.empty:
            xgb_trends['Model'] = 'XGBoost_Trends'
            table10_phase2_XGBoost_Trends_results(xgb_trends)
            
        lgb_trends = all_metrics_df[all_metrics_df['Model'].str.startswith('LightGBM_Trends')].copy()
        if not lgb_trends.empty:
            lgb_trends['Model'] = 'LightGBM_Trends'
            table11_phase2_LightGBM_Trends_results(lgb_trends)
            
        ari_trends = all_metrics_df[all_metrics_df['Model'].str.startswith('ARIMAX_Trends')].copy()
        if not ari_trends.empty:
            ari_trends['Model'] = 'ARIMAX_Trends'
            table12_phase2_ARIMAX_Trends_results(ari_trends)
            
        logger.info("Successfully generated Phase 2 built-in statistical artifacts.")
    except Exception as e:
        logger.error(f"Failed to generate Phase 2 artifacts: {e}")
        
    # ============================================================
    # GENERATE VISUALIZATION ARTIFACTS (FIGURES)
    # ============================================================
    try:
        from src.visualization.figures import (
            fig10_phase2_division_forecasts, fig11_phase2_heatmap_division_model
        )
        
        # fig10 requires a dict: {division: {"df": df, "best_model": str}}
        # But for simplicity we can just pass the bare minimum or construct it from all_predictions
        fig10_results = {}
        for div, preds_dict in all_predictions.items():
            if not preds_dict: continue
            # Find best model for this div
            div_metrics = all_metrics_df[all_metrics_df['Division'] == div]
            best_model = div_metrics.loc[div_metrics['RMSE'].idxmin()]['Model'] if not div_metrics.empty else list(preds_dict.keys())[0]
            
            # Reconstruct df
            df_div = df[df['Division'] == div].sort_values('Date').dropna(subset=[target_col]).copy()
            y_actual = df_div[target_col].values[-len(preds_dict[best_model]):]
            dates = df_div['Date'].values[-len(preds_dict[best_model]):]
            
            plot_df = pd.DataFrame({"Date": dates, "Actual": y_actual, "Prediction": preds_dict[best_model][:len(y_actual)]})
            fig10_results[div] = {"df": plot_df, "best_model": best_model}
            
        fig10_phase2_division_forecasts(fig10_results)
        
        # fig11
        pivot_df = all_metrics_df.pivot(index="Division", columns="Model", values="RMSE")
        fig11_phase2_heatmap_division_model(pivot_df)
        
        logger.info("Successfully generated Phase 2 visualization figures.")
    except Exception as e:
        logger.error(f"Failed to generate Phase 2 figures: {e}")
        
    return all_results, all_predictions, failure_log

if __name__ == "__main__":
    main()
