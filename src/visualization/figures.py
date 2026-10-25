import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.api import qqplot

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Publication quality settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Colorblind-safe palette (Okabe-Ito)
CB_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']
from src.config import FIGURES_DIR
FIG_DIR = FIGURES_DIR

def _ensure_dir():
    """Ensure the figures directory exists."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def _save_fig(fig: Figure, name: str):
    """Save figure in both PNG and PDF formats."""
    _ensure_dir()
    for ext in ["png", "pdf"]:
        filepath = FIG_DIR / f"{name}.{ext}"
        fig.savefig(filepath)
        logger.info(f"Saved figure: {filepath}")

def fig1_phase1_study_workflow():
    """Generates the Methodology flowchart."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis("off")
    # Draw simple blocks as representation
    ax.text(0.5, 0.8, "Data Collection", ha="center", va="center", bbox=dict(boxstyle="round", fc="w"))
    ax.text(0.5, 0.6, "Phase 1: National Monthly Forecasting\n(SARIMA, ETS, Prophet, XGBoost, LightGBM, SVR, MLP)", ha="center", va="center", bbox=dict(boxstyle="round", fc="w"))
    ax.text(0.5, 0.4, "Phase 2: Division Daily Forecasting\n(ARIMA, ETS, XGBoost, LightGBM)", ha="center", va="center", bbox=dict(boxstyle="round", fc="w"))
    ax.text(0.5, 0.2, "Evaluation & Comparison", ha="center", va="center", bbox=dict(boxstyle="round", fc="w"))
    
    # Arrows
    ax.annotate("", xy=(0.5, 0.7), xytext=(0.5, 0.75), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.5, 0.5), xytext=(0.5, 0.55), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.5, 0.3), xytext=(0.5, 0.35), arrowprops=dict(arrowstyle="->"))
    
    fig.tight_layout()
    _save_fig(fig, "fig1_phase1_study_workflow")
    plt.close(fig)

def fig2_phase1_national_timeseries(df: pd.DataFrame):
    """Monthly cases 2012-2025 with outbreak year annotations."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if 'Date' in df.columns and 'measles_total' in df.columns:
        ax.plot(pd.to_datetime(df['Date']), df['measles_total'], color=CB_PALETTE[0], label="Measles Cases")
        ax.set_ylabel("Cases")
        ax.set_xlabel("Year")
        ax.legend()
    fig.tight_layout()
    _save_fig(fig, "fig2_phase1_national_timeseries")
    plt.close(fig)

def fig3_phase1_seasonal_decomposition(df: pd.DataFrame):
    """Additive decomposition."""
    fig, axes = plt.subplots(4, 1, figsize=(7, 7), sharex=True)
    if 'Date' in df.columns and 'measles_total' in df.columns:
        series = pd.Series(df['measles_total'].values, index=pd.to_datetime(df['Date']))
        if len(series) >= 24:
            decomp = seasonal_decompose(series, model='additive', period=12)
            axes[0].plot(decomp.observed, color=CB_PALETTE[0])
            axes[0].set_ylabel("Observed")
            axes[1].plot(decomp.trend, color=CB_PALETTE[1])
            axes[1].set_ylabel("Trend")
            axes[2].plot(decomp.seasonal, color=CB_PALETTE[2])
            axes[2].set_ylabel("Seasonal")
            axes[3].plot(decomp.resid, color=CB_PALETTE[3], marker='o', linestyle='none')
            axes[3].set_ylabel("Residual")
    fig.tight_layout()
    _save_fig(fig, "fig3_phase1_seasonal_decomposition")
    plt.close(fig)

def fig4_phase1_acf_pacf(series: pd.Series):
    """ACF and PACF side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    if len(series) > 0:
        plot_acf(series.dropna(), ax=axes[0], color=CB_PALETTE[0])
        plot_pacf(series.dropna(), ax=axes[1], color=CB_PALETTE[1])
    fig.tight_layout()
    _save_fig(fig, "fig4_phase1_acf_pacf")
    plt.close(fig)

def fig5_phase1_actual_vs_predicted(results_dict: Dict[str, pd.Series]):
    """Top models overlaid on actual."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if "actual" in results_dict:
        ax.plot(results_dict["actual"].index, results_dict["actual"].values, label="Actual", color='black', linewidth=1.5)
        
    for i, (model_name, preds) in enumerate(results_dict.items()):
        if model_name != "actual" and i < len(CB_PALETTE):
            ax.plot(preds.index, preds.values, label=model_name, color=CB_PALETTE[i], linestyle='--')
            
    ax.set_ylabel("Cases")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "fig5_phase1_actual_vs_predicted")
    plt.close(fig)

def fig6_phase1_model_comparison_bar(metrics_df: pd.DataFrame):
    """Grouped bar chart of RMSE/MAE/sMAPE for all models."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if not metrics_df.empty and set(['Model', 'RMSE', 'MAE', 'sMAPE']).issubset(metrics_df.columns):
        metrics_df.set_index('Model')[['RMSE', 'MAE', 'sMAPE']].plot(kind='bar', ax=ax, color=CB_PALETTE[:3])
        ax.set_ylabel("Error Metric Value")
        plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    _save_fig(fig, "fig6_phase1_model_comparison_bar")
    plt.close(fig)

def fig7_phase1_feature_importance(importances: np.ndarray, feature_names: List[str]):
    """Horizontal bar chart (top 15 features)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(importances) > 0:
        indices = np.argsort(importances)[-15:]
        ax.barh(range(len(indices)), importances[indices], align='center', color=CB_PALETTE[0])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(np.array(feature_names)[indices])
        ax.set_xlabel("Importance")
    fig.tight_layout()
    _save_fig(fig, "fig7_phase1_feature_importance")
    plt.close(fig)

def fig8_phase1_shap_summary(model: Any, X: pd.DataFrame, feature_names: List[str]):
    """Genuine SHAP summary plot."""
    try:
        import shap
        # Ensure we have a figure object to manage
        fig = plt.figure(figsize=(7, 5))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        _save_fig(plt.gcf(), "fig8_phase1_shap_summary")
        plt.close()
    except Exception as e:
        logger.error(f"SHAP plot failed: {e}")

def fig9_phase1_residual_analysis(y_true: pd.Series, y_pred: pd.Series, model_name: str):
    """2x2 grid (residuals vs time, histogram, ACF, Q-Q)."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    residuals = y_true - y_pred
    
    # Time
    axes[0,0].plot(residuals.index, residuals.values, color=CB_PALETTE[0])
    axes[0,0].set_title("Residuals vs Time")
    
    # Histogram
    sns.histplot(residuals.dropna(), kde=True, ax=axes[0,1], color=CB_PALETTE[1])
    axes[0,1].set_title("Histogram")
    
    # ACF
    plot_acf(residuals.dropna(), ax=axes[1,0], color=CB_PALETTE[2])
    axes[1,0].set_title("ACF")
    
    # QQ
    qqplot(residuals.dropna(), line='s', ax=axes[1,1])
    axes[1,1].set_title("Q-Q Plot")
    
    fig.suptitle(f"Residual Analysis: {model_name}")
    fig.tight_layout()
    _save_fig(fig, "fig9_phase1_residual_analysis")
    plt.close(fig)

def fig10_phase2_division_forecasts(results: Dict[str, dict]):
    """8-panel grid showing forecasts per division with dynamic best models."""
    fig, axes = plt.subplots(4, 2, figsize=(7, 9), sharex=True)
    axes = axes.flatten()
    for i, (div, info) in enumerate(results.items()):
        if i < 8:
            df = info["df"]
            best_model = info["best_model"]
            if 'Date' in df.columns and 'Actual' in df.columns and 'Prediction' in df.columns:
                axes[i].plot(pd.to_datetime(df['Date']), df['Actual'], label='Actual', color='black')
                axes[i].plot(pd.to_datetime(df['Date']), df['Prediction'], label=f'Forecast (Best: {best_model})', color=CB_PALETTE[0], linestyle='--')
            axes[i].set_title(div)
            axes[i].legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig10_phase2_division_forecasts")
    plt.close(fig)

def fig11_phase2_heatmap_division_model(metrics_df: pd.DataFrame):
    """Model x Division performance heatmap."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if not metrics_df.empty:
        # Fill missing values (if a model failed for a division) with a noticeable flag or just let seaborn handle NaNs
        sns.heatmap(metrics_df, annot=True, cmap="YlGnBu", ax=ax, fmt=".2f", mask=metrics_df.isna())
        ax.set_title("RMSE by Division and Model")
    fig.tight_layout()
    _save_fig(fig, "fig11_phase2_heatmap_division_model")
    plt.close(fig)

def fig12_phase1_google_trends_overlay(df_cases: pd.DataFrame, df_trends: pd.DataFrame):
    """Dual-axis time series overlay."""
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    if 'date' in df_cases.columns and 'cases' in df_cases.columns:
        ax1.plot(pd.to_datetime(df_cases['date']), df_cases['cases'], color=CB_PALETTE[0], label="Cases")
        ax1.set_ylabel("Cases", color=CB_PALETTE[0])
        ax1.tick_params(axis='y', labelcolor=CB_PALETTE[0])
        
        ax2 = ax1.twinx()
        if 'date' in df_trends.columns and 'trend' in df_trends.columns:
            ax2.plot(pd.to_datetime(df_trends['date']), df_trends['trend'], color=CB_PALETTE[1], label="Google Trends", linestyle='--')
            ax2.set_ylabel("Trend Index", color=CB_PALETTE[1])
            ax2.tick_params(axis='y', labelcolor=CB_PALETTE[1])
            
    fig.tight_layout()
    _save_fig(fig, "fig12_phase1_google_trends_overlay")
    plt.close(fig)

def fig13_phase1_ablation_comparison(base_metrics: pd.DataFrame, trends_metrics: pd.DataFrame):
    """Bar chart comparing with/without Google Trends."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if not base_metrics.empty and not trends_metrics.empty and 'Model' in base_metrics.columns:
        merged = base_metrics.merge(trends_metrics, on="Model", suffixes=("_Base", "_Trends"))
        if 'RMSE_Base' in merged.columns and 'RMSE_Trends' in merged.columns:
            merged.set_index('Model')[['RMSE_Base', 'RMSE_Trends']].plot(kind='bar', ax=ax, color=[CB_PALETTE[0], CB_PALETTE[1]])
            ax.set_ylabel("RMSE")
            plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    _save_fig(fig, "fig13_phase1_ablation_comparison")
    plt.close(fig)

def fig14_phase2_XGBoost_Base_division_forecasts(results: Dict[str, dict]):
    """8-panel grid showing forecasts per division specifically for Base XGBoost."""
    fig, axes = plt.subplots(4, 2, figsize=(7, 9), sharex=True)
    axes = axes.flatten()
    for i, (div, info) in enumerate(results.items()):
        if i < 8:
            df = info["df"]
            # Force the label to explicitly state this is XGBoost Base regardless of best_model string
            if 'Date' in df.columns and 'Actual' in df.columns and 'Prediction' in df.columns:
                axes[i].plot(pd.to_datetime(df['Date']), df['Actual'], label='Actual', color='black')
                axes[i].plot(pd.to_datetime(df['Date']), df['Prediction'], label='Forecast (XGBoost_Base)', color=CB_PALETTE[0], linestyle='--')
            axes[i].set_title(div)
            axes[i].legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig14_phase2_XGBoost_Base_division_forecasts")
    plt.close(fig)

def fig15_phase2_LightGBM_Base_division_forecasts(results: Dict[str, dict]):
    """8-panel grid showing forecasts per division specifically for Base LightGBM."""
    fig, axes = plt.subplots(4, 2, figsize=(7, 9), sharex=True)
    axes = axes.flatten()
    for i, (div, info) in enumerate(results.items()):
        if i < 8:
            df = info["df"]
            if 'Date' in df.columns and 'Actual' in df.columns and 'Prediction' in df.columns:
                axes[i].plot(pd.to_datetime(df['Date']), df['Actual'], label='Actual', color='black')
                axes[i].plot(pd.to_datetime(df['Date']), df['Prediction'], label='Forecast (LightGBM_Base)', color=CB_PALETTE[1], linestyle='--')
            axes[i].set_title(div)
            axes[i].legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig15_phase2_LightGBM_Base_division_forecasts")
    plt.close(fig)

def fig16_phase2_XGBoost_Trends_division_forecasts(results: Dict[str, dict]):
    """8-panel grid showing forecasts per division specifically for XGBoost + Trends."""
    fig, axes = plt.subplots(4, 2, figsize=(7, 9), sharex=True)
    axes = axes.flatten()
    for i, (div, info) in enumerate(results.items()):
        if i < 8:
            df = info["df"]
            if 'Date' in df.columns and 'Actual' in df.columns and 'Prediction' in df.columns:
                axes[i].plot(pd.to_datetime(df['Date']), df['Actual'], label='Actual', color='black')
                axes[i].plot(pd.to_datetime(df['Date']), df['Prediction'], label='Forecast (XGBoost + Google Trends)', color=CB_PALETTE[0], linestyle='--')
            axes[i].set_title(div)
            axes[i].legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig16_phase2_XGBoost_Trends_division_forecasts")
    plt.close(fig)

def fig17_phase2_LightGBM_Trends_division_forecasts(results: Dict[str, dict]):
    """8-panel grid showing forecasts per division specifically for LightGBM + Trends."""
    fig, axes = plt.subplots(4, 2, figsize=(7, 9), sharex=True)
    axes = axes.flatten()
    for i, (div, info) in enumerate(results.items()):
        if i < 8:
            df = info["df"]
            if 'Date' in df.columns and 'Actual' in df.columns and 'Prediction' in df.columns:
                axes[i].plot(pd.to_datetime(df['Date']), df['Actual'], label='Actual', color='black')
                axes[i].plot(pd.to_datetime(df['Date']), df['Prediction'], label='Forecast (LightGBM + Google Trends)', color=CB_PALETTE[1], linestyle='--')
            axes[i].set_title(div)
            axes[i].legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig17_phase2_LightGBM_Trends_division_forecasts")
    plt.close(fig)

def fig18_phase2_ARIMAX_Trends_division_forecasts(results: Dict[str, dict]):
    """8-panel grid showing forecasts per division specifically for ARIMAX + Trends."""
    fig, axes = plt.subplots(4, 2, figsize=(7, 9), sharex=True)
    axes = axes.flatten()
    for i, (div, info) in enumerate(results.items()):
        if i < 8:
            df = info["df"]
            if 'Date' in df.columns and 'Actual' in df.columns and 'Prediction' in df.columns:
                axes[i].plot(pd.to_datetime(df['Date']), df['Actual'], label='Actual', color='black')
                axes[i].plot(pd.to_datetime(df['Date']), df['Prediction'], label='Forecast (ARIMAX + Google Trends)', color=CB_PALETTE[2], linestyle='--')
            axes[i].set_title(div)
            axes[i].legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig18_phase2_ARIMAX_Trends_division_forecasts")
    plt.close(fig)

def fig19_phase2_ARIMAX_Base_division_forecasts(results: Dict[str, dict]):
    """8-panel grid showing forecasts per division specifically for ARIMAX Base."""
    fig, axes = plt.subplots(4, 2, figsize=(7, 9), sharex=True)
    axes = axes.flatten()
    for i, (div, info) in enumerate(results.items()):
        if i < 8:
            df = info["df"]
            if 'Date' in df.columns and 'Actual' in df.columns and 'Prediction' in df.columns:
                axes[i].plot(pd.to_datetime(df['Date']), df['Actual'], label='Actual', color='black')
                axes[i].plot(pd.to_datetime(df['Date']), df['Prediction'], label='Forecast (ARIMAX_Base)', color=CB_PALETTE[2], linestyle='--')
            axes[i].set_title(div)
            axes[i].legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig19_phase2_ARIMAX_Base_division_forecasts")
    plt.close(fig)

def fig20_phase2_ETS_Base_division_forecasts(results: Dict[str, dict]):
    """8-panel grid showing forecasts per division specifically for ETS Base."""
    fig, axes = plt.subplots(4, 2, figsize=(7, 9), sharex=True)
    axes = axes.flatten()
    for i, (div, info) in enumerate(results.items()):
        if i < 8:
            df = info["df"]
            if 'Date' in df.columns and 'Actual' in df.columns and 'Prediction' in df.columns:
                axes[i].plot(pd.to_datetime(df['Date']), df['Actual'], label='Actual', color='black')
                axes[i].plot(pd.to_datetime(df['Date']), df['Prediction'], label='Forecast (ETS_Base)', color=CB_PALETTE[2], linestyle='--')
            axes[i].set_title(div)
            axes[i].legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig20_phase2_ETS_Base_division_forecasts")
    plt.close(fig)

if __name__ == "__main__":
    pass
