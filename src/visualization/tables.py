import os
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.config import TABLES_DIR

def _ensure_dir():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

def _save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Saves table to LaTeX and returns the DataFrame."""
    _ensure_dir()
    latex_path = TABLES_DIR / f"{name}.tex"
    df.to_latex(latex_path, index=False, escape=False)
    logger.info(f"Saved table: {latex_path}")
    return df

def table1_phase1_dataset_summary(df_cases, df_climate, df_vaccination, df_population, df_trends) -> pd.DataFrame:
    """Dataset description table generated from actual loaded dataframes."""
    data = [
        {"Dataset": "National Measles Cases", "Records": len(df_cases), "Features": len(df_cases.columns), "Start": df_cases.index.min(), "End": df_cases.index.max()},
        {"Dataset": "Climate Data", "Records": len(df_climate), "Features": len(df_climate.columns), "Start": df_climate.index.min() if len(df_climate) > 0 else None, "End": df_climate.index.max() if len(df_climate) > 0 else None},
        {"Dataset": "Vaccination", "Records": len(df_vaccination), "Features": len(df_vaccination.columns), "Start": df_vaccination.index.min() if len(df_vaccination) > 0 else None, "End": df_vaccination.index.max() if len(df_vaccination) > 0 else None},
        {"Dataset": "Population", "Records": len(df_population), "Features": len(df_population.columns), "Start": df_population.index.min() if len(df_population) > 0 else None, "End": df_population.index.max() if len(df_population) > 0 else None},
        {"Dataset": "Google Trends", "Records": len(df_trends), "Features": len(df_trends.columns), "Start": df_trends.index.min() if len(df_trends) > 0 else None, "End": df_trends.index.max() if len(df_trends) > 0 else None}
    ]
    df = pd.DataFrame(data)
    return _save_table(df, "table1_phase1_dataset_summary")

def table2_phase1_feature_description(features_list: List[str]) -> pd.DataFrame:
    """All features with name derived from actual used features."""
    data = []
    for f in features_list:
        dtype = "Numeric"
        data.append({"Feature": f, "Type": dtype})
    df = pd.DataFrame(data)
    return _save_table(df, "table2_phase1_feature_description")

def table3_phase1_model_hyperparameters(results: Dict[str, Any]) -> pd.DataFrame:
    """Best hyperparameters for each model."""
    data = []
    for model, hparams in results.items():
        data.append({"Model": model, "Hyperparameters": str(hparams)})
    if not data:
        data = [{"Model": "Example", "Hyperparameters": "lr=0.01"}]
    df = pd.DataFrame(data)
    return _save_table(df, "table3_phase1_model_hyperparameters")

def table4_phase1_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for all Phase 1 models."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 4: results_df is empty.")
    
    # Identify best metric and bold it
    df = results_df.copy()
    
    # Ensure consistent rounding for all numeric columns before string formatting
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    if 'RMSE' in df.columns:
        best_rmse = df['RMSE'].min()
        df['RMSE'] = df['RMSE'].apply(lambda x: f"\\textbf{{{x}}}" if x == best_rmse else str(x))
    return _save_table(df, "table4_phase1_results")

def table5_phase2_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for Phase 2."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 5: results_df is empty.")
        
    df = results_df.copy()
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    return _save_table(df, "table5_phase2_results")

def table6_phase1_dm_test(results_df: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano pairwise comparison matrix."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 6: results_df is empty.")
    return _save_table(results_df, "table6_phase1_dm_test")

def table7_phase1_ablation(base: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    """With/without Google Trends comparison."""
    if base.empty or trends.empty:
        raise ValueError("Cannot generate Table 7: base or trends df is empty.")
    else:
        df = base.merge(trends, on="Model", suffixes=("_Base", "_Trends"))
        # Example calculation
        if 'RMSE_Base' in df.columns and 'RMSE_Trends' in df.columns:
            df['Improvement (%)'] = ((df['RMSE_Base'] - df['RMSE_Trends']) / df['RMSE_Base'] * 100).round(1).astype(str) + "%"
    return _save_table(df, "table7_phase1_ablation")

def table8_phase2_XGBoost_Base_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for Base XGBoost Phase 2."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 8: results_df is empty.")
        
    df = results_df.copy()
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    return _save_table(df, "table8_phase2_XGBoost_Base_results")

def table9_phase2_LightGBM_Base_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for Base LightGBM Phase 2."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 9: results_df is empty.")
        
    df = results_df.copy()
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    return _save_table(df, "table9_phase2_LightGBM_Base_results")

def _add_average_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    avg_row = {col: df[col].mean() for col in numeric_cols}
    if 'Division' in df.columns:
        avg_row['Division'] = 'Average'
    if 'Model' in df.columns:
        avg_row['Model'] = df['Model'].iloc[0]
    
    # Ensure types match to avoid issues in concat
    df[numeric_cols] = df[numeric_cols].round(3)
    avg_df = pd.DataFrame([avg_row])
    avg_df[numeric_cols] = avg_df[numeric_cols].round(3)
    
    return pd.concat([df, avg_df], ignore_index=True)

def table10_phase2_XGBoost_Trends_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for XGBoost + Trends Phase 2, with an average row."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 10: results_df is empty.")
    df = _add_average_row(results_df[['Division', 'RMSE', 'MAE', 'sMAPE', 'Model']])
    return _save_table(df, "table10_phase2_XGBoost_Trends_results")

def table11_phase2_LightGBM_Trends_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for LightGBM + Trends Phase 2, with an average row."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 11: results_df is empty.")
    df = _add_average_row(results_df[['Division', 'RMSE', 'MAE', 'sMAPE', 'Model']])
    return _save_table(df, "table11_phase2_LightGBM_Trends_results")

def table12_phase2_ARIMAX_Trends_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for ARIMAX + Trends Phase 2, with an average row."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 12: results_df is empty.")
    df = _add_average_row(results_df[['Division', 'RMSE', 'MAE', 'sMAPE', 'Model']])
    return _save_table(df, "table12_phase2_ARIMAX_Trends_results")

def table13_phase2_ARIMAX_Base_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for ARIMAX Base Phase 2, with an average row."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 13: results_df is empty.")
    df = _add_average_row(results_df[['Division', 'RMSE', 'MAE', 'sMAPE', 'Model']])
    return _save_table(df, "table13_phase2_ARIMAX_Base_results")

def table14_phase2_ETS_Base_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """All metrics for ETS Base Phase 2, with an average row."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 14: results_df is empty.")
    df = _add_average_row(results_df[['Division', 'RMSE', 'MAE', 'sMAPE', 'Model']])
    return _save_table(df, "table14_phase2_ETS_Base_results")

if __name__ == "__main__":
    pass

def table16_phase2_dm_test(results_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2 Diebold-Mariano comparisons (FDR corrected)."""
    if results_df.empty:
        raise ValueError("Cannot generate Table 16: results_df is empty.")
    return _save_table(results_df, "table16_phase2_dm_test")
