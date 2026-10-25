"""
End-to-end data pipeline.
"""
import logging
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.loader import (load_who_cases, load_division_cases, load_climate_data, 
                             load_population_data, load_vaccination_data, load_google_trends)
from src.data.cleaner import (remove_trailing_empty_rows, standardize_division_names, 
                              validate_date_formats, handle_daily_outliers, handle_missing_climate_values)
from src.data.feature_engineering import build_phase1_features, build_phase2_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    """Data pipeline class for loading processed data."""
    
    def __init__(self):
        self.processed_dir = PROCESSED_DATA_DIR
    
    def load_processed_phase1(self):
        """Load processed Phase 1 data."""
        path = self.processed_dir / "phase1_monthly_national.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            raise FileNotFoundError(f"Processed Phase 1 data not found at {path}")
    
    def load_processed_phase2(self):
        """Load processed Phase 2 data."""
        path = self.processed_dir / "phase2_daily_division.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            raise FileNotFoundError(f"Processed Phase 2 data not found at {path}")

def generate_data_quality_report(df: pd.DataFrame, phase: str):
    """Generates a data quality report."""
    logger.info(f"--- Data Quality Report: {phase} ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info("-------------------------------------")

def run_pipeline():
    """Runs the full data pipeline."""
    logger.info("Starting data pipeline...")
    
    try:
        # Load actual datasets
        logger.info("Loading datasets...")
        df_cases = pd.read_excel(RAW_DATA_DIR / 'Measles Monthly Data 2012-2025 [WHO].xlsx')
        df_cases_div = pd.read_csv(RAW_DATA_DIR / 'daily_division_2026.csv')
        df_clim_monthly = pd.read_csv(RAW_DATA_DIR / 'Division _Monthly_climate_2012-2025.csv')
        df_clim_daily = pd.read_csv(RAW_DATA_DIR / 'bangladesh_climate_daily_2026.csv')
        df_pop = pd.read_csv(RAW_DATA_DIR / 'bangladesh_population_density_2012_2025.csv')
        df_vacc = pd.read_csv(RAW_DATA_DIR / 'bangladesh_national_coverage_2012_2025.csv')
        df_trends_monthly = pd.read_csv(RAW_DATA_DIR / 'google_trends_monthly_2012_2025.csv')
        df_trends_daily = pd.read_csv(RAW_DATA_DIR / 'google_trends_daily_2026.csv')
        
        # Transformations
        logger.info("Transforming datasets...")
        # WHO data
        df_cases['Date'] = pd.to_datetime(dict(year=df_cases['year'], month=df_cases['month'], day=1))
        df_cases.set_index('Date', inplace=True)
        
        # Monthly climate
        df_clim_monthly_agg = df_clim_monthly.groupby('Year_Month', as_index=False).agg({
            'Temperature_C': 'mean',
            'Humidity_Percent': 'mean',
            'Monthly_Avg_Rainfall_mm_per_day': 'mean'
        })
        df_clim_monthly_agg['Date'] = pd.to_datetime(df_clim_monthly_agg['Year_Month'].str.replace('_', '-') + '-01')
        
        # Vaccination - interpolate yearly to monthly
        df_vacc_monthly = df_vacc.copy()
        df_vacc_monthly['Date'] = pd.to_datetime(df_vacc_monthly['Year'].astype(str) + '-01-01')
        df_vacc_monthly = df_vacc_monthly.set_index('Date').resample('MS').interpolate().reset_index()
        df_vacc_monthly.drop(columns=['Year'], inplace=True)
        
        # Population - aggregate to national monthly
        df_pop_national = df_pop.groupby('Year')[['Population']].sum().reset_index()
        df_pop_monthly = df_pop_national.copy()
        df_pop_monthly['Date'] = pd.to_datetime(df_pop_monthly['Year'].astype(str) + '-01-01')
        df_pop_monthly = df_pop_monthly.set_index('Date').resample('MS').interpolate().reset_index()
        df_pop_monthly.drop(columns=['Year'], inplace=True)
        
        # Google Trends monthly
        df_trends_monthly['Date'] = pd.to_datetime(df_trends_monthly['Year_Month'].str.replace('_', '-') + '-01')
        
        # Phase 1
        df_phase1_base = df_cases.reset_index() 
        df_phase1_base = pd.merge(df_phase1_base, df_clim_monthly_agg.drop(columns=['Year_Month']), on='Date', how='left')
        df_phase1_base = pd.merge(df_phase1_base, df_trends_monthly, on='Date', how='left')
        df_phase1_base = pd.merge(df_phase1_base, df_vacc_monthly, on='Date', how='left')
        df_phase1_base = pd.merge(df_phase1_base, df_pop_monthly, on='Date', how='left')
        df_phase1_base.set_index('Date', inplace=True)
        
        # Phase 2
        df_cases_div['Date'] = pd.to_datetime(df_cases_div['Date'])
        df_cases_div['Year'] = df_cases_div['Date'].dt.year
        df_cases_div['Division'] = df_cases_div['Division'].str.strip().str.title()
        df_clim_daily['Division'] = df_clim_daily['Division'].str.strip().str.title()
        df_clim_daily['Date'] = pd.to_datetime(df_clim_daily['Date'])
        df_trends_daily['Date'] = pd.to_datetime(df_trends_daily['Date'])
        df_pop_div = df_pop.copy()
        df_pop_div['Division'] = df_pop_div['Division'].str.strip().str.title()
        
        # Fix for 2026 Phase 2 data: propagate 2025 population figures to 2026
        df_pop_2026 = df_pop_div[df_pop_div['Year'] == 2025].copy()
        df_pop_2026['Year'] = 2026
        df_pop_div = pd.concat([df_pop_div, df_pop_2026], ignore_index=True)
        
        df_phase2_base = pd.merge(df_cases_div, df_clim_daily, on=['Date', 'Division'], how='left')
        df_phase2_base = pd.merge(df_phase2_base, df_trends_daily, on='Date', how='left')
        df_phase2_base = pd.merge(df_phase2_base, df_pop_div[['Year', 'Division', 'Population', 'Area_sq_km', 'Density_per_sq_km']], on=['Year', 'Division'], how='left')
        
        # Feature Engineering
        phase1_df = build_phase1_features(df_phase1_base, df_clim_monthly_agg, df_vacc_monthly, df_pop_monthly, df_trends_monthly)
        phase2_df = build_phase2_features(df_phase2_base, df_clim_daily, df_trends_daily)
        
        # Save
        phase1_path = PROCESSED_DATA_DIR / "phase1_monthly_national.csv"
        phase2_path = PROCESSED_DATA_DIR / "phase2_daily_division.csv"
        
        phase1_df.to_csv(phase1_path)
        phase2_df.to_csv(phase2_path, index=False)
        
        logger.info(f"Saved Phase 1 processed data to {phase1_path}")
        logger.info(f"Saved Phase 2 processed data to {phase2_path}")
        
        # Reports
        generate_data_quality_report(phase1_df, "Phase 1")
        generate_data_quality_report(phase2_df, "Phase 2")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Data pipeline completed.")

if __name__ == "__main__":
    run_pipeline()