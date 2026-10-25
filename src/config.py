"""
Central configuration for the measles forecasting project.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(os.environ.get("PROJECT_ROOT", "placeholder_path"))
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random seed
RANDOM_SEED = 42
RANDOM_STATE = 42

# Cross-validation
CV_N_SPLITS = 3

# Phase 1: Monthly National
TEST_SIZE_PHASE1 = 24  # months
FORECAST_HORIZON_PHASE1 = 24
ROLLING_ORIGINS_PHASE1 = 5

# Phase 2: Daily Division
TEST_SIZE_PHASE2 = 21  # days
FORECAST_HORIZON_PHASE2 = 21
ROLLING_ORIGINS_PHASE2 = 5

# Feature Engineering
LAG_FEATURES = list(range(1, 13))
ROLLING_WINDOWS = [3, 6, 12]

# Phase 1 Features
BASE_FEATURES_PHASE1 = [
    'year', 'month_sin', 'month_cos',
    'cases_lag_1', 'cases_lag_2', 'cases_lag_3', 'cases_lag_6', 'cases_lag_12',
    'cases_rmean_3', 'cases_rstd_3', 'cases_rmean_6', 'cases_rstd_6', 'cases_rmean_12',
    'Temperature_C', 'Humidity_Percent', 'Monthly_Avg_Rainfall_mm_per_day',
    'National_MR1_Coverage', 'National_MR2_Coverage',
    'Population', 'is_covid'
]

TRENDS_FEATURES_PHASE1 = BASE_FEATURES_PHASE1 + [
    'trends_lag_1', 'trends_lag_2', 'trends_lag_3', 'trends_rmean_3', 'trends_change', 'trends_spike'
]

TARGET_PHASE1 = 'measles_total'

# Phase 2 Features
BASE_FEATURES_PHASE2 = [
    'cases_lag_1', 'cases_lag_3', 'cases_lag_7', 'cases_lag_14',
    'cases_rmean_7', 'cases_rmean_14', 'cases_rstd_7',
    'cumulative_cases_since_start', 'growth_rate',
    'dow', 'dow_sin', 'dow_cos',
    'Temperature_C', 'Humidity_Percent', 'Daily_Rainfall_mm',
    'Population', 'Area_sq_km', 'Density_per_sq_km', 'division_encoded'
]

TRENDS_FEATURES_PHASE2 = BASE_FEATURES_PHASE2 + [
    'trends_lag_1', 'trends_lag_7', 'trends_lag_14'
]

TARGET_PHASE2 = 'Cases'

# Model Hyperparameter Grids
MODEL_GRIDS = {
    "XGBoost": {
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__n_estimators": [100, 300, 500],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    },
    "LightGBM": {
        "model__num_leaves": [31, 50, 100],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__n_estimators": [100, 300, 500],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    },
    "SVR": {
        "model__C": [0.1, 1.0, 10.0, 100.0],
        "model__epsilon": [0.01, 0.1, 0.2],
        "model__kernel": ["rbf"]
    },
    "MLP": {
        "model__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "model__learning_rate_init": [0.001, 0.01, 0.1],
        "model__max_iter": [500, 1000],
        "model__alpha": [0.0001, 0.001, 0.01]
    }
}

# Metrics
METRICS = ["rmse", "mae", "mape", "r2"]

# Plotting Settings
PLOT_CONFIG = {
    "dpi": 300,
    "font": "Arial",
    "figsize_default": (10, 6)
}

# Color Palette (colorblind-safe)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#F0E442", "#56B4E9", "#D55E00", "#CC79A7"]