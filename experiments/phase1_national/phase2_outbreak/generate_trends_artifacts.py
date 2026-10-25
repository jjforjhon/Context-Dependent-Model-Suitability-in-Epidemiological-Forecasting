import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from src.config import DATA_DIR, RESULTS_DIR
from src.visualization import figures, tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate():
    # 1. Generate Tables from metrics CSV
    metrics_csv = RESULTS_DIR / "phase2" / "metrics" / "phase2_metrics.csv"
    if not metrics_csv.exists():
        logger.error("Phase 2 metrics CSV not found!")
        return
        
    df = pd.read_csv(metrics_csv)
    
    xgb_trends = df[df["Model"] == "XGBoost"].copy()
    lgb_trends = df[df["Model"] == "LightGBM"].copy()
    arima_trends = df[df["Model"] == "ARIMA"].copy()
    
    tables.table10_phase2_XGBoost_Trends_results(xgb_trends)
    tables.table11_phase2_LightGBM_Trends_results(lgb_trends)
    tables.table12_phase2_ARIMAX_Trends_results(arima_trends)
    
    # 2. Generate Figures from saved models and data
    merged_path = DATA_DIR / "processed" / "phase2_daily_division.csv"
    if not merged_path.exists():
        logger.error("Merged data not found!")
        return
        
    df_merged = pd.read_csv(merged_path)
    df_merged['Date'] = pd.to_datetime(df_merged['Date'])
    df_merged = df_merged.sort_values(by=['Division', 'Date']).reset_index(drop=True)
    
    div_results_xgb = {}
    div_results_lgb = {}
    div_results_arima = {}
    
    divisions = sorted(df_merged['Division'].unique())
    preds_dir = RESULTS_DIR / "phase2" / "predictions"
    
    for div in divisions:
        # XGBoost
        xgb_pred_path = preds_dir / f"{div}_XGBoost_Trends_predictions.csv"
        if xgb_pred_path.exists():
            df_xgb = pd.read_csv(xgb_pred_path)
            div_results_xgb[div] = {"df": df_xgb}
            
        # LightGBM
        lgb_pred_path = preds_dir / f"{div}_LightGBM_Trends_predictions.csv"
        if lgb_pred_path.exists():
            df_lgb = pd.read_csv(lgb_pred_path)
            div_results_lgb[div] = {"df": df_lgb}
            
        # ARIMAX
        arima_pred_path = preds_dir / f"{div}_ARIMA_Trends_predictions.csv"
        if arima_pred_path.exists():
            df_arima = pd.read_csv(arima_pred_path)
            div_results_arima[div] = {"df": df_arima}
            
    figures.fig16_phase2_XGBoost_Trends_division_forecasts(div_results_xgb)
    figures.fig17_phase2_LightGBM_Trends_division_forecasts(div_results_lgb)
    figures.fig18_phase2_ARIMAX_Trends_division_forecasts(div_results_arima)

if __name__ == "__main__":
    generate()
