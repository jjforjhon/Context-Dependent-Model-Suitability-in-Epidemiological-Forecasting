"""
Machine Learning forecasting models for measles forecasting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

def perform_grid_search(model, param_grid, X_train, y_train):
    """Utility to perform grid search with TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=5)
    gsearch = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    gsearch.fit(X_train, y_train)
    return gsearch.best_estimator_, gsearch.best_params_


def fit_predict_xgboost(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    logger.info("Starting XGBoost modeling...")
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBRegressor(random_state=RANDOM_SEED)
    best_model, best_params = perform_grid_search(xgb, param_grid, X_train, y_train)
    
    preds = best_model.predict(X_test)
    train_preds = best_model.predict(X_train)
    
    return {
        "model_name": "XGBoost",
        "predictions": preds,
        "train_predictions": train_preds,
        "best_params": best_params,
        "feature_importances": best_model.feature_importances_
    }


def fit_predict_lightgbm(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    logger.info("Starting LightGBM modeling...")
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [5, 10, 20],  # Equivalent to min_child_weight
        'subsample': [0.8, 1.0]
    }
    
    lgb = LGBMRegressor(random_state=RANDOM_SEED, verbose=-1)
    best_model, best_params = perform_grid_search(lgb, param_grid, X_train, y_train)
    
    preds = best_model.predict(X_test)
    train_preds = best_model.predict(X_train)
    
    return {
        "model_name": "LightGBM",
        "predictions": preds,
        "train_predictions": train_preds,
        "best_params": best_params,
        "feature_importances": best_model.feature_importances_
    }


def fit_predict_svr(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    logger.info("Starting SVR modeling...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.01, 0.1, 0.5]
    }
    
    svr = SVR(kernel='rbf')
    best_model, best_params = perform_grid_search(svr, param_grid, X_train_scaled, y_train)
    
    preds = best_model.predict(X_test_scaled)
    train_preds = best_model.predict(X_train_scaled)
    
    return {
        "model_name": "SVR",
        "predictions": preds,
        "train_predictions": train_preds,
        "best_params": best_params,
        "feature_importances": None
    }


def fit_predict_mlp(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    logger.info("Starting MLP modeling...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'hidden_layer_sizes': [(32, 16), (64, 32), (32,)]
    }
    
    mlp = MLPRegressor(
        max_iter=1000, 
        early_stopping=True, 
        validation_fraction=0.15,
        random_state=RANDOM_SEED
    )
    best_model, best_params = perform_grid_search(mlp, param_grid, X_train_scaled, y_train)
    
    preds = best_model.predict(X_test_scaled)
    train_preds = best_model.predict(X_train_scaled)
    
    return {
        "model_name": "MLP",
        "predictions": preds,
        "train_predictions": train_preds,
        "best_params": best_params,
        "feature_importances": None
    }


def fit_predict_rf(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    logger.info("Starting Random Forest modeling...")
    
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 15, None]
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    best_model, best_params = perform_grid_search(rf, param_grid, X_train, y_train)
    
    preds = best_model.predict(X_test)
    train_preds = best_model.predict(X_train)
    
    return {
        "model_name": "RandomForest",
        "predictions": preds,
        "train_predictions": train_preds,
        "best_params": best_params,
        "feature_importances": best_model.feature_importances_
    }
