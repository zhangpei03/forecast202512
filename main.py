import argparse
from pathlib import Path

from src.forecast_agent.config import load_settings, Settings
from src.forecast_agent.pipeline import ForecastPipeline
from src.forecast_agent.visualization import build_metrics_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="时间序列预测流水线")
    parser.add_argument("--data-path", type=Path, help="Excel 数据路径")
    parser.add_argument("--date-column", type=str, help="日期列名")
    parser.add_argument("--target-column", type=str, help="目标数值列名")
    parser.add_argument("--output-dir", type=Path, help="输出目录，包含图表/报告")
    return parser.parse_args()


def merge_settings(args: argparse.Namespace) -> Settings:
    cfg = load_settings()
    if args.data_path:
        cfg.data_path = args.data_path
    if args.date_column:
        cfg.date_column = args.date_column
    if args.target_column:
        cfg.target_column = args.target_column
    if args.output_dir:
        cfg.output_dir = args.output_dir
    return cfg


def main() -> None:
    args = parse_args()
    cfg = merge_settings(args)
    pipeline = ForecastPipeline(cfg)
    output = pipeline.run()

    print("=== 评估指标（按 RMSE 升序）===")
    for row in output.metrics_table:
        print(row)
    if output.llm_analysis:
        print("\n=== LLM 分析 ===")
        print(output.llm_analysis)
    else:
        print("\n未生成 LLM 分析，可能缺少 DEEPSEEK_API_KEY/DEEPSEEK_API_URL。")
    print(f"\n可视化已输出: {output.chart_path}")


if __name__ == "__main__":
    main()

