"""
Data cleaning module.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def remove_trailing_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Removes trailing empty rows from the DataFrame."""
    initial_len = len(df)
    df_clean = df.dropna(how='all')
    logger.info(f"Removed {initial_len - len(df_clean)} trailing empty rows.")
    return df_clean

def standardize_division_names(df: pd.DataFrame, div_col: str = 'Division') -> pd.DataFrame:
    """Standardizes division names."""
    if div_col in df.columns:
        df[div_col] = df[div_col].astype(str).str.strip().str.title()
        logger.info(f"Standardized division names in '{div_col}'.")
    return df

def validate_date_formats(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """Validates date formats."""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        invalid = df[date_col].isna().sum()
        if invalid > 0:
            logger.warning(f"Found {invalid} unparseable dates in '{date_col}'.")
    return df

def handle_daily_outliers(df: pd.DataFrame, target_col: str = 'Cases', div_col: str = 'Division') -> pd.DataFrame:
    """Flags outliers (> mean + 3*std) per division."""
    df['is_outlier'] = 0
    if div_col in df.columns and target_col in df.columns:
        for div, group in df.groupby(div_col):
            mean = group[target_col].mean()
            std = group[target_col].std()
            thresh = mean + 3 * std
            outlier_idx = group[group[target_col] > thresh].index
            df.loc[outlier_idx, 'is_outlier'] = 1
        logger.info(f"Flagged {df['is_outlier'].sum()} outliers.")
    return df

def handle_missing_climate_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing climate values via forward fill."""
    nas_before = df.isna().sum().sum()
    df_clean = df.ffill()
    nas_after = df_clean.isna().sum().sum()
    logger.info(f"Filled {nas_before - nas_after} missing climate values with forward fill.")
    return df_clean

def convert_excel_to_csv(excel_path: Path, csv_path: Path):
    """Converts WHO Excel to clean CSV."""
    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False)
    logger.info(f"Converted {excel_path} to {csv_path}")

def validate_data_types(df: pd.DataFrame, type_mapping: dict) -> pd.DataFrame:
    """Validates and enforces data types."""
    for col, dtype in type_mapping.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.error(f"Failed to cast {col} to {dtype}: {e}")
    logger.info("Validated data types.")
    return df
