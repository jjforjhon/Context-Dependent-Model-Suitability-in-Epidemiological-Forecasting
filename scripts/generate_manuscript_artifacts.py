import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.visualization.figures import (
    fig10_phase2_division_forecasts, fig11_phase2_heatmap_division_model,
    fig14_phase2_XGBoost_Base_division_forecasts, fig15_phase2_LightGBM_Base_division_forecasts,
    fig16_phase2_XGBoost_Trends_division_forecasts, fig17_phase2_LightGBM_Trends_division_forecasts,
    fig18_phase2_ARIMAX_Trends_division_forecasts, fig19_phase2_ARIMAX_Base_division_forecasts,
    fig20_phase2_ETS_Base_division_forecasts
)
from src.models.statistical import dm_test
from src.visualization.tables import _save_table, table6_phase1_dm_test, table16_phase2_dm_test

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Phase 2 Manuscript Artifact Orchestrator...")
    
    results_dir = Path("results/phase2")
    metrics_path = results_dir / "metrics" / "phase2_metrics.csv"
    preds_dir = results_dir / "predictions"
    
    if not metrics_path.exists():
        logger.error(f"Cannot find {metrics_path}. Please run run_all_phase2.py first.")
        return
        
    all_metrics = pd.read_csv(metrics_path)
    divisions = all_metrics['Division'].unique()
    
    # ---------------------------------------------------------
    # 1. GENERATE INDIVIDUAL FORECAST VISUALIZATIONS
    # ---------------------------------------------------------
    plot_map = {
        "XGBoost_Base": fig14_phase2_XGBoost_Base_division_forecasts,
        "LightGBM_Base": fig15_phase2_LightGBM_Base_division_forecasts,
        "ARIMAX_Base": fig19_phase2_ARIMAX_Base_division_forecasts,
        "ETS_Base": fig20_phase2_ETS_Base_division_forecasts,
        "XGBoost_Trends": fig16_phase2_XGBoost_Trends_division_forecasts,
        "LightGBM_Trends": fig17_phase2_LightGBM_Trends_division_forecasts,
        "ARIMAX_Trends": fig18_phase2_ARIMAX_Trends_division_forecasts,
    }
    
    for model_col, plot_func in plot_map.items():
        results_dict = {}
        for div in divisions:
            pred_file = preds_dir / f"phase2_{div}_preds.csv"
            if not pred_file.exists():
                continue
            
            df_pred = pd.read_csv(pred_file)
            pred_col = f"{model_col}_{div}"
            if pred_col not in df_pred.columns:
                continue
            
            if 'Date' not in df_pred.columns:
                df_pred['Date'] = pd.date_range(start="2026-07-06", periods=len(df_pred), freq="D")
            
            plot_df = pd.DataFrame({
                "Date": df_pred['Date'],
                "Actual": df_pred['Actual'],
                "Prediction": df_pred[pred_col]
            })
            results_dict[div] = {"df": plot_df}
            
        if results_dict:
            logger.info(f"Generating plot for {model_col}...")
            try:
                plot_func(results_dict)
            except Exception as e:
                logger.error(f"Plot failed for {model_col}: {e}")

    # ---------------------------------------------------------
    # 2. GENERATE BASE-VS-TRENDS COMPARISON (TABLE 15)
    # ---------------------------------------------------------
    logger.info("Generating Table 15 (Base vs Trends)...")
    base_metrics = all_metrics[all_metrics['Model'].str.contains('_Base')].copy()
    trends_metrics = all_metrics[all_metrics['Model'].str.contains('_Trends')].copy()
    
    base_metrics['Core_Model'] = base_metrics['Model'].apply(lambda x: x.split('_Base')[0])
    trends_metrics['Core_Model'] = trends_metrics['Model'].apply(lambda x: x.split('_Trends')[0])
    
    merged = pd.merge(
        base_metrics, trends_metrics,
        on=['Division', 'Core_Model'],
        suffixes=('_Base', '_Trends')
    )
    
    if not merged.empty:
        merged['RMSE_Delta'] = merged['RMSE_Trends'] - merged['RMSE_Base']
        table15_df = merged[['Division', 'Core_Model', 'RMSE_Base', 'RMSE_Trends', 'RMSE_Delta']]
        _save_table(table15_df, "table15_phase2_Base_vs_Trends_Comparison")
    
    # ---------------------------------------------------------
    # 3. GENERATE WINNER ANALYSES (FIG 21, 22)
    # ---------------------------------------------------------
    logger.info("Generating Winner Analyses...")
    import matplotlib.pyplot as plt
    
    def _draw_winner_bar(df, title, filename):
        df_clean = df.copy()
        # Remove the division name from the end of the model string if it exists
        # e.g., "XGBoost_Base_Barishal" -> "XGBoost_Base"
        df_clean['Core_Model'] = df_clean.apply(
            lambda row: row['Model'].replace(f"_{row['Division']}", "") if row['Model'].endswith(f"_{row['Division']}") else row['Model'], 
            axis=1
        )
        winners = df_clean.loc[df_clean.groupby('Division')['RMSE'].idxmin()]
        counts = winners['Core_Model'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind='bar', color='#0072B2', ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Number of Divisions Won")
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        
        fig_dir = Path("results/figures")
        fig_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_dir / f"{filename}.pdf")
        fig.savefig(fig_dir / f"{filename}.png")
        plt.close(fig)

    if not base_metrics.empty:
        _draw_winner_bar(base_metrics, "Phase 2 Division Winners (Base Models Only)", "fig21_phase2_base_winners")
        
    if not all_metrics.empty:
        _draw_winner_bar(all_metrics, "Phase 2 Division Winners (All Models)", "fig22_phase2_overall_winners")
        
    # ---------------------------------------------------------
    # 4. GENERATE HEATMAPS (FIG 11)
    # ---------------------------------------------------------
    logger.info("Generating Heatmap (Fig 11)...")
    if not all_metrics.empty:
        pivot_df = all_metrics.pivot(index="Division", columns="Model", values="RMSE")
        try:
            fig11_phase2_heatmap_division_model(pivot_df)
        except Exception as e:
            logger.error(f"Fig11 failed: {e}")


    # ---------------------------------------------------------
    # 5. STATISTICAL TESTING (DM TESTS)
    # ---------------------------------------------------------
    logger.info("Computing Statistical Tests (Phase 1 & Phase 2)...")
    
    # --- PHASE 1 DM TEST ---
    phase1_preds_path = Path("results/phase1/predictions/phase1_preds.csv")
    if phase1_preds_path.exists():
        p1_preds = pd.read_csv(phase1_preds_path)
        actual1 = p1_preds["Actual"].values
        
        models_to_compare = [
            ("XGBoost_Trends", "ETS"),
            ("SVR_Trends", "ETS"),
            ("MLP_Base", "ETS"),
            ("Drift", "ETS"),
            ("XGBoost_Trends", "SARIMAX"),
            ("SVR_Trends", "SARIMAX"),
            ("Drift", "Zeros")
        ]
        
        dm_results_p1 = []
        for m1, m2 in models_to_compare:
            if m1 in p1_preds.columns and m2 in p1_preds.columns:
                pred1 = p1_preds[m1].values
                pred2 = p1_preds[m2].values
                stat, p = dm_test(actual1, pred1, pred2, h=1)
                dm_results_p1.append({
                    "Model_1": m1, "Model_2": m2,
                    "DM_stat": stat, "p_value": p, "Significant": p < 0.05
                })
        
        if dm_results_p1:
            table6_phase1_dm_test(pd.DataFrame(dm_results_p1))
            logger.info("Generated Table 6 (Phase 1 DM Test).")
            
    # --- PHASE 2 DM TEST ---
    if not all_metrics.empty and preds_dir.exists():
        dm_results_p2 = []
        divisions = all_metrics["Division"].unique()
        
        for div in divisions:
            div_metrics = all_metrics[all_metrics["Division"] == div]
            pred_file = preds_dir / f"phase2_{div}_preds.csv"
            
            if not pred_file.exists():
                continue
                
            p2_preds = pd.read_csv(pred_file)
            if "Actual" not in p2_preds.columns:
                continue
            actual2 = p2_preds["Actual"].values
            
            # Find best ML and best Statistical model for this division
            ml_models = [m for m in div_metrics["Model"] if "XGBoost" in m or "LightGBM" in m or "SVR" in m or "MLP" in m]
            stat_models = [m for m in div_metrics["Model"] if "ARIMAX" in m or "ETS" in m or "Prophet" in m]
            
            if not ml_models or not stat_models:
                continue
                
            best_ml = min(ml_models, key=lambda m: div_metrics[div_metrics["Model"] == m]["RMSE"].values[0])
            best_stat = min(stat_models, key=lambda m: div_metrics[div_metrics["Model"] == m]["RMSE"].values[0])
            
            # Remove division suffix from column names if present
            # E.g. XGBoost_Base_Dhaka -> XGBoost_Base
            best_ml_col = best_ml
            best_stat_col = best_stat
            
            # Wait, run_all_phase2.py saves column names as 'XGBoost_Base' or 'XGBoost_Base_Dhaka'?
            # Let's check columns inside p2_preds!
            # Usually they are just 'XGBoost_Base' inside the division CSV.
            if best_ml not in p2_preds.columns:
                best_ml_col = best_ml.replace(f"_{div}", "")
            if best_stat not in p2_preds.columns:
                best_stat_col = best_stat.replace(f"_{div}", "")
                
            if best_ml_col in p2_preds.columns and best_stat_col in p2_preds.columns:
                pred1 = p2_preds[best_ml_col].values
                pred2 = p2_preds[best_stat_col].values
                
                stat, p = dm_test(actual2, pred1, pred2, h=1)
                dm_results_p2.append({
                    "Division": div,
                    "Best_ML": best_ml_col,
                    "Best_Stat": best_stat_col,
                    "DM_stat": stat,
                    "p_value": p
                })
                
        if dm_results_p2:
            p2_df = pd.DataFrame(dm_results_p2)
            # Apply Benjamini-Hochberg FDR
            p_values = p2_df["p_value"].values
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            m = len(p_values)
            
            significant = np.zeros(m, dtype=bool)
            for i in range(m):
                threshold = (i + 1) / m * 0.05
                if sorted_p[i] <= threshold:
                    significant[sorted_indices[i]] = True
                    
            p2_df["Significant_FDR"] = significant
            table16_phase2_dm_test(p2_df)
            logger.info("Generated Table 16 (Phase 2 DM Test with FDR).")

    logger.info("All supplementary artifacts generated successfully!")

if __name__ == "__main__":
    main()
