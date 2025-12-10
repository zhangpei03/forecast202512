from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .evaluation import Metrics, compute_metrics
from .preprocess import create_lag_features


@dataclass
class ModelResult:
    name: str
    y_true: pd.Series
    y_pred: pd.Series
    metrics: Metrics

    def summary(self) -> dict:
        return {"name": self.name, "metrics": self.metrics.to_dict()}


def _iterative_predict(
    model, history: List[float], steps: int, lags: int, feature_names: List[str]
) -> List[float]:
    """
    滚动预测，确保测试集使用模型自身预测的历史。
    """
    preds: List[float] = []
    hist = list(history)
    for _ in range(steps):
        window = hist[-lags:]
        X_pred = pd.DataFrame([window], columns=feature_names)
        y_hat = float(model.predict(X_pred)[0])
        preds.append(y_hat)
        hist.append(y_hat)
    return preds


def train_linear_regression(
    train: pd.DataFrame, test: pd.DataFrame, target_col: str, lags: int = 7
) -> ModelResult:
    train_series = train[target_col].reset_index(drop=True)
    test_series = test[target_col].reset_index(drop=True)

    X_train, y_train = create_lag_features(train_series, lags=lags)
    feature_names = list(X_train.columns)
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = _iterative_predict(
        model, history=train_series.tolist(), steps=len(test_series), lags=lags, feature_names=feature_names
    )
    y_pred = pd.Series(preds, index=test.index)
    metrics = compute_metrics(test_series, y_pred.reset_index(drop=True))
    return ModelResult(name="LinearRegression", y_true=test_series, y_pred=y_pred, metrics=metrics)


def train_xgboost(
    train: pd.DataFrame, test: pd.DataFrame, target_col: str, lags: int = 7
) -> ModelResult:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("需要安装 xgboost 以运行该模型") from exc

    train_series = train[target_col].reset_index(drop=True)
    test_series = test[target_col].reset_index(drop=True)

    X_train, y_train = create_lag_features(train_series, lags=lags)
    feature_names = list(X_train.columns)
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = _iterative_predict(
        model, history=train_series.tolist(), steps=len(test_series), lags=lags, feature_names=feature_names
    )
    y_pred = pd.Series(preds, index=test.index)
    metrics = compute_metrics(test_series, y_pred.reset_index(drop=True))
    return ModelResult(name="XGBoost", y_true=test_series, y_pred=y_pred, metrics=metrics)


def train_prophet(
    train: pd.DataFrame, test: pd.DataFrame, date_col: str, target_col: str
) -> ModelResult:
    try:
        from prophet import Prophet
    except Exception as exc:
        raise RuntimeError("需要安装 prophet 以运行该模型") from exc

    train_df = train.rename(columns={date_col: "ds", target_col: "y"})
    test_df = test.rename(columns={date_col: "ds", target_col: "y"})
    m = Prophet(
        seasonality_mode="additive",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.add_seasonality(name="weekly", period=7, fourier_order=3)
    m.fit(train_df)

    future = m.make_future_dataframe(periods=len(test_df), freq=pd.infer_freq(train_df["ds"]) or "D")
    forecast = m.predict(future)
    y_pred = forecast.tail(len(test_df))["yhat"].reset_index(drop=True)
    metrics = compute_metrics(test_df["y"].reset_index(drop=True), y_pred)
    return ModelResult(name="Prophet", y_true=test_df["y"], y_pred=y_pred, metrics=metrics)


def train_arima(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> ModelResult:
    try:
        import pmdarima as pm
    except Exception as exc:
        raise RuntimeError("需要安装 pmdarima 以运行该模型") from exc

    train_series = train[target_col].reset_index(drop=True)
    test_series = test[target_col].reset_index(drop=True)
    model = pm.auto_arima(train_series, seasonal=False, suppress_warnings=True, stepwise=True)
    preds = model.predict(n_periods=len(test_series))
    y_pred = pd.Series(preds, index=test.index)
    metrics = compute_metrics(test_series, y_pred.reset_index(drop=True))
    return ModelResult(name="ARIMA", y_true=test_series, y_pred=y_pred, metrics=metrics)


# ---------- 未来预测（30 天等外推） ----------


def forecast_future_linear(
    data: pd.DataFrame, target_col: str, lags: int = 7, steps: int = 30
) -> List[float]:
    series = data[target_col].reset_index(drop=True)
    X, y = create_lag_features(series, lags=lags)
    model = LinearRegression()
    model.fit(X, y)
    feature_names = list(X.columns)
    preds = _iterative_predict(
        model, history=series.tolist(), steps=steps, lags=lags, feature_names=feature_names
    )
    return preds


def forecast_future_xgboost(
    data: pd.DataFrame, target_col: str, lags: int = 7, steps: int = 30
) -> List[float]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("需要安装 xgboost 以运行该模型") from exc

    series = data[target_col].reset_index(drop=True)
    X, y = create_lag_features(series, lags=lags)
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)
    feature_names = list(X.columns)
    preds = _iterative_predict(
        model, history=series.tolist(), steps=steps, lags=lags, feature_names=feature_names
    )
    return preds


def forecast_future_prophet(
    data: pd.DataFrame, date_col: str, target_col: str, steps: int = 30, freq: str = "D"
) -> List[float]:
    try:
        from prophet import Prophet
    except Exception as exc:
        raise RuntimeError("需要安装 prophet 以运行该模型") from exc

    df = data.rename(columns={date_col: "ds", target_col: "y"})
    m = Prophet(
        seasonality_mode="additive",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.add_seasonality(name="weekly", period=7, fourier_order=3)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq=freq)
    forecast = m.predict(future)
    return forecast.tail(steps)["yhat"].tolist()


def forecast_future_arima(
    data: pd.DataFrame, target_col: str, steps: int = 30
) -> List[float]:
    try:
        import pmdarima as pm
    except Exception as exc:
        raise RuntimeError("需要安装 pmdarima 以运行该模型") from exc

    series = data[target_col].reset_index(drop=True)
    model = pm.auto_arima(series, seasonal=False, suppress_warnings=True, stepwise=True)
    preds = model.predict(n_periods=steps)
    return preds.tolist()

