"""
Feature engineering module.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_phase1_features(df_cases: pd.DataFrame, df_climate: pd.DataFrame, 
                          df_vaccination: pd.DataFrame, df_population: pd.DataFrame, 
                          df_trends: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features for monthly national forecasting (Phase 1).
    """
    logger.info("Building Phase 1 features...")
    df = df_cases.copy()
    
    # 1. Temporal features
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year'] = df.index.year
    
    # 2. Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f'cases_lag_{lag}'] = df['measles_total'].shift(lag)
        
    # 3. Rolling features
    for window in [3, 6, 12]:
        df[f'cases_rmean_{window}'] = df['measles_total'].shift(1).rolling(window=window).mean()
        if window in [3, 6]:
            df[f'cases_rstd_{window}'] = df['measles_total'].shift(1).rolling(window=window).std()
            
    # Mocking merges since specific columns aren't provided in full context, but adding expected structure.
    # Climate lags
    for lag in [1, 2]:
        if 'Temperature_C' in df.columns:
            df[f'temp_lag_{lag}'] = df['Temperature_C'].shift(lag)
        if 'Humidity_Percent' in df.columns:
            df[f'humidity_lag_{lag}'] = df['Humidity_Percent'].shift(lag)
    if 'Monthly_Avg_Rainfall_mm_per_day' in df.columns:
        df['rainfall_lag_1'] = df['Monthly_Avg_Rainfall_mm_per_day'].shift(1)
        
    # Google Trends features
    if 'trends_combined' in df.columns:
        for lag in [1, 2, 3]:
            df[f'trends_lag_{lag}'] = df['trends_combined'].shift(lag)
        df['trends_rmean_3'] = df['trends_combined'].shift(1).rolling(window=3).mean()
        df['trends_change'] = df['trends_combined'].shift(1).diff()
        df['trends_spike'] = (df['trends_combined'].shift(1) > 2 * df['trends_rmean_3'].shift(1)).astype(int)

    # COVID indicator
    # Assumes DatetimeIndex
    df['is_covid'] = ((df.index >= '2020-03-01') & (df.index <= '2021-12-31')).astype(int)
    
    logger.info("Phase 1 features created successfully.")
    return df

def build_phase2_features(df_cases_div: pd.DataFrame, df_climate_daily: pd.DataFrame, 
                          df_trends_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features for daily division forecasting (Phase 2).
    """
    logger.info("Building Phase 2 features...")
    df = df_cases_div.copy()
    
    # Needs to group by division for proper lags
    if 'Division' not in df.columns:
        logger.warning("'Division' column not found, lags might be incorrect across division boundaries.")
        
    # Lags
    for lag in [1, 3, 7, 14]:
        df[f'cases_lag_{lag}'] = df.groupby('Division')['Cases'].shift(lag)
        
    # Rolling
    for window in [7, 14]:
        df[f'cases_rmean_{window}'] = df.groupby('Division')['Cases'].transform(lambda x: x.shift(1).rolling(window).mean())
    df['cases_rstd_7'] = df.groupby('Division')['Cases'].transform(lambda x: x.shift(1).rolling(7).std())
    
    # Cumulative
    df['cumulative_cases_since_start'] = df.groupby('Division')['Cases'].transform(lambda x: x.shift(1).cumsum())
    
    # Growth rate
    cases_t1 = df.groupby('Division')['Cases'].shift(1)
    cases_t2 = df.groupby('Division')['Cases'].shift(2)
    # Handle division by zero and cap at 10
    growth = np.where(cases_t2 == 0, 0, cases_t1 / cases_t2)
    df['growth_rate'] = np.clip(growth, a_min=None, a_max=10)
    
    # Day of week
    if 'Date' in df.columns:
        df['dow'] = df['Date'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        
    # Climate and Trends lags would also be merged and shifted per division
    # (Assuming climate and trends merged prior or done here)
    climate_col_mapping = {
        'Temperature_C': 'temperature',
        'Humidity_Percent': 'humidity',
        'Daily_Rainfall_mm': 'rainfall'
    }
    
    for real_col, prefix in climate_col_mapping.items():
        if real_col in df.columns:
            df[f'{prefix}_lag_1'] = df.groupby('Division')[real_col].shift(1)
            df[f'{prefix}_lag_7'] = df.groupby('Division')[real_col].shift(7)
            df[f'{prefix}_lag_14'] = df.groupby('Division')[real_col].shift(14)
            
    if 'trends_combined' in df.columns:
        df['trends_lag_1'] = df.groupby('Division')['trends_combined'].shift(1)
        df['trends_lag_7'] = df.groupby('Division')['trends_combined'].shift(7)
        df['trends_lag_14'] = df.groupby('Division')['trends_combined'].shift(14)

    # Division encoded
    if 'Division' in df.columns:
        df['division_encoded'] = df['Division'].astype('category').cat.codes
        
    logger.info("Phase 2 features created successfully.")
    return df
