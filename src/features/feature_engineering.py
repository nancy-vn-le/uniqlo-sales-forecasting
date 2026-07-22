"""Feature engineering for the forecasting models."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all model features from raw division_daily + daily_sales joined dataframe.
    Returns a new DataFrame with added feature columns.
    """
    df = df.copy().sort_values(['division_code', 'date']).reset_index(drop=True)

    # Date-based features
    df['dow_num'] = df['date'].dt.dayofweek          # 0=Mon, 6=Sun
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.day_of_year
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    # Event one-hot encoding
    for ev in ['Arigato', 'Christmas', 'EOFY', 'Vivid', 'LongWeekend']:
        df[f'event_{ev}'] = (df['event_flag'] == ev).astype(int)

    # Boolean → int
    for col in ['is_public_holiday', 'promo_flag']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Lag features per division
    for lag in [7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby('division_code')['sales_amt'].shift(lag)

    # Rolling means per division
    for window in [7, 14]:
        df[f'rolling_{window}_mean'] = (
            df.groupby('division_code')['sales_amt']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # YoY same-day comparison
    df['lag_364'] = df.groupby('division_code')['sales_amt'].shift(364)

    # Cold snap flag
    df['cold_snap'] = (df['temperature_index'] < 15).astype(int)

    return df


def prepare_prophet_df(df: pd.DataFrame, div_code: int) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a single division's data in Prophet format with event regressors."""
    EVENT_COLS = ['Arigato', 'Christmas', 'EOFY', 'Vivid', 'LongWeekend']
    d = df[df['division_code'] == div_code][
        ['date', 'sales_amt', 'event_flag']
    ].copy()
    d = d.rename(columns={'date': 'ds', 'sales_amt': 'y'})
    for ev in EVENT_COLS:
        d[ev] = (d['event_flag'] == ev).astype(int)
    return d, EVENT_COLS
