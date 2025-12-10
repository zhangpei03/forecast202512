from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas.tseries.frequencies import to_offset
from tqdm import tqdm

from .config import Settings
from .data_loader import load_excel
from .evaluation import Metrics
from .llm_agent import analyze_forecast_with_llm
from .models import (
    ModelResult,
    forecast_future_arima,
    forecast_future_prophet,
    forecast_future_linear,
    forecast_future_xgboost,
    train_arima,
    train_prophet,
    train_linear_regression,
    train_xgboost,
)
from .preprocess import clean_series, train_test_split_ts
from .visualization import build_metrics_table, plot_forecasts


@dataclass
class PipelineOutput:
    settings: Settings
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    results: List[ModelResult]
    chart_path: Path
    llm_analysis: str | None
    metrics_table: List[Dict]
    future_dates: List[pd.Timestamp]
    future_predictions: Dict[str, List[float]]
    excel_path: Path


def export_predictions_excel(
    output_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    future_dates: List[pd.Timestamp],
    date_col: str,
    target_col: str,
    results: List[ModelResult],
    future_predictions: Dict[str, List[float]],
) -> Path:
    """
    导出包含训练集、测试集预测以及未来30天预测的Excel。
    训练集部分不包含预测，仅保留实际值以保证完整时间轴。
    """
    train_part = train_df[[date_col, target_col]].copy()
    train_part = train_part.rename(columns={date_col: "date", target_col: "actual"})
    for res in results:
        train_part[f"{res.name}_pred"] = None

    test_part = test_df[[date_col, target_col]].copy()
    test_part = test_part.rename(columns={date_col: "date", target_col: "actual"})
    for res in results:
        col = f"{res.name}_pred"
        test_part[col] = res.y_pred.reset_index(drop=True)

    fut = pd.DataFrame({"date": future_dates, "actual": [None] * len(future_dates)})
    for res in results:
        col = f"{res.name}_pred"
        preds = future_predictions.get(res.name, [float("nan")] * len(future_dates))
        fut[col] = preds

    export_df = pd.concat(
        [train_part.reset_index(drop=True), test_part.reset_index(drop=True), fut],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_excel(output_path, index=False)
    return output_path


class ForecastPipeline:
    """
    端到端流水线：加载数据 -> 预处理 -> 多模型预测 -> 评估/可视化 -> LLM 分析。
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self) -> PipelineOutput:
        cfg = self.settings
        data = load_excel(cfg.data_path, cfg.date_column, cfg.target_column)
        data = clean_series(data, cfg.date_column, cfg.target_column)
        train_df, test_df = train_test_split_ts(data, test_size=0.2)
        freq = pd.infer_freq(data[cfg.date_column]) or "D"
        future_dates = pd.date_range(
            start=data[cfg.date_column].iloc[-1] + to_offset(freq),
            periods=30,
            freq=freq,
        )

        models = [
            ("XGBoost", lambda: train_xgboost(train_df, test_df, cfg.target_column)),
            ("LinearRegression", lambda: train_linear_regression(train_df, test_df, cfg.target_column)),
            ("Prophet", lambda: train_prophet(train_df, test_df, cfg.date_column, cfg.target_column)),
            ("ARIMA", lambda: train_arima(train_df, test_df, cfg.target_column)),
        ]

        results: List[ModelResult] = []
        for name, runner in tqdm(models, desc="模型训练/预测"):
            try:
                res = runner()
                results.append(res)
            except Exception as exc:
                # 将失败模型记录占位，便于后续调试
                placeholder = ModelResult(
                    name=f"{name}-failed",
                    y_true=test_df[cfg.target_column].reset_index(drop=True),
                    y_pred=pd.Series([float("nan")] * len(test_df)),
                    metrics=Metrics(mae=float("inf"), rmse=float("inf"), mape=float("inf")),
                )
                results.append(placeholder)
                print(f"{name} 失败: {exc}")

        # 生成未来 30 天预测
        future_preds: Dict[str, List[float]] = {}
        for name, func in [
            ("XGBoost", lambda: forecast_future_xgboost(data, cfg.target_column)),
            ("LinearRegression", lambda: forecast_future_linear(data, cfg.target_column)),
            ("Prophet", lambda: forecast_future_prophet(data, cfg.date_column, cfg.target_column, steps=30, freq=freq)),
            ("ARIMA", lambda: forecast_future_arima(data, cfg.target_column)),
        ]:
            try:
                future_preds[name] = func()
            except Exception as exc:  # noqa: BLE001
                print(f"{name} 未来预测失败: {exc}")
                future_preds[name] = [float("nan")] * len(future_dates)

        chart_path = plot_forecasts(
            train=train_df,
            test=test_df,
            date_col=cfg.date_column,
            target_col=cfg.target_column,
            results=results,
            output_path=cfg.output_dir / "forecast_chart.html",
            future_dates=future_dates,
            future_predictions=future_preds,
        )

        metrics_table = build_metrics_table(results)
        llm_analysis = analyze_forecast_with_llm(
            metrics=metrics_table,
            context="包含数据清洗、训练集/测试集划分，以及基础可视化。",
            settings=cfg,
        )

        excel_path = export_predictions_excel(
            output_path=cfg.output_dir / "predictions.xlsx",
            train_df=train_df,
            test_df=test_df,
            future_dates=list(future_dates),
            date_col=cfg.date_column,
            target_col=cfg.target_column,
            results=results,
            future_predictions=future_preds,
        )

        return PipelineOutput(
            settings=cfg,
            train_df=train_df,
            test_df=test_df,
            results=results,
            chart_path=chart_path,
            llm_analysis=llm_analysis,
            metrics_table=metrics_table,
            future_dates=list(future_dates),
            future_predictions=future_preds,
            excel_path=excel_path,
        )

