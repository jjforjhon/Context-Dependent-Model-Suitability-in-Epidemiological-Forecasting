"""
Data loading module.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def load_who_cases(file_path: Path) -> pd.DataFrame:
    """
    Loads WHO Excel measles data and converts to DataFrame with proper datetime index.
    """
    try:
        logger.info(f"Loading WHO cases from {file_path}")
        df = pd.read_excel(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load WHO cases: {e}")
        raise

def load_division_cases(file_path: Path) -> pd.DataFrame:
    """
    Loads daily division case data.
    """
    try:
        logger.info(f"Loading division cases from {file_path}")
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        logger.error(f"Failed to load division cases: {e}")
        raise

def load_climate_data(file_path: Path) -> pd.DataFrame:
    """
    Loads climate data (monthly 2012-2025, monthly 2026, daily 2026).
    """
    try:
        logger.info(f"Loading climate data from {file_path}")
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        logger.error(f"Failed to load climate data: {e}")
        raise

def load_population_data(file_path: Path) -> pd.DataFrame:
    """
    Loads population density data.
    """
    try:
        logger.info(f"Loading population data from {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to load population data: {e}")
        raise

def load_vaccination_data(file_path: Path) -> pd.DataFrame:
    """
    Loads vaccination coverage data.
    """
    try:
        logger.info(f"Loading vaccination data from {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to load vaccination data: {e}")
        raise

def load_google_trends(file_path: Path) -> pd.DataFrame:
    """
    Loads Google Trends data (monthly and daily).
    """
    try:
        logger.info(f"Loading Google Trends data from {file_path}")
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        logger.error(f"Failed to load Google Trends data: {e}")
        raise

def validate_loaded_data(df: pd.DataFrame, expected_cols: list) -> bool:
    """
    Validates data after loading (checks expected columns).
    """
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns during validation: {missing_cols}")
        return False
    return True
