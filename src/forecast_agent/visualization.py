from pathlib import Path
from typing import Dict, List

import pandas as pd
from pyecharts.charts import Line
from pyecharts import options as opts

from .models import ModelResult


def plot_forecasts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str,
    target_col: str,
    results: List[ModelResult],
    output_path: Path,
    future_dates: List[pd.Timestamp] | None = None,
    future_predictions: Dict[str, List[float]] | None = None,
) -> Path:
    """
    输出折线图，含真实值、测试集预测与未来预测。
    """
    future_dates = [] if future_dates is None else list(future_dates)
    future_predictions = future_predictions or {}
    x_axis = test[date_col].dt.strftime("%Y-%m-%d").tolist() + [
        d.strftime("%Y-%m-%d") for d in future_dates
    ]

    line = Line(init_opts=opts.InitOpts(width="1400px", height="700px"))
    line.add_xaxis(x_axis)

    real_series = test[target_col].tolist() + [None] * len(future_dates)
    line.add_yaxis(
        series_name="真实值",
        y_axis=real_series,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )

    for res in results:
        future_list = future_predictions.get(res.name, [None] * len(future_dates))
        y_axis = res.y_pred.tolist() + future_list
        line.add_yaxis(
            series_name=res.name,
            y_axis=y_axis,
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
        )
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="预测对比（含未来外推）"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts()],
        yaxis_opts=opts.AxisOpts(name=target_col),
        xaxis_opts=opts.AxisOpts(name=date_col, type_="category"),
        legend_opts=opts.LegendOpts(type_="scroll"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line.render(str(output_path))
    return output_path


def build_metrics_table(results: List[ModelResult]) -> List[Dict]:
    return [
        {"model": r.name, **r.metrics.to_dict()}
        for r in sorted(results, key=lambda r: r.metrics.rmse)
    ]

