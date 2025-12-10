from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_series(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """
    基础清洗：去重、按日期排序、缺失值前向填充。
    """
    cleaned = (
        df[[date_col, target_col]]
        .drop_duplicates(subset=[date_col])
        .sort_values(by=date_col)
        .copy()
    )
    cleaned[target_col] = cleaned[target_col].interpolate().ffill().bfill()
    return cleaned.reset_index(drop=True)


def train_test_split_ts(
    df: pd.DataFrame, test_size: float = 0.2, shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    时间序列拆分，默认保持时间顺序。
    """
    if shuffle:
        return train_test_split(df, test_size=test_size, shuffle=True, random_state=42)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def create_lag_features(
    series: pd.Series, lags: int = 7, horizon: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    将单变量时间序列转换为监督学习格式，用于树模型/XGBoost。
    """
    data = {}
    for i in range(1, lags + 1):
        data[f"lag_{i}"] = series.shift(i)
    X = pd.DataFrame(data)
    y = series.shift(-horizon)
    mask = ~(X.isna().any(axis=1) | y.isna())
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

