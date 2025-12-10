from pathlib import Path
import pandas as pd


def load_excel(path: Path, date_column: str, target_column: str) -> pd.DataFrame:
    """
    加载 Excel 数据，确保日期列为 datetime，按日期排序。
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")

    df = pd.read_excel(path)
    if date_column not in df.columns or target_column not in df.columns:
        raise ValueError(
            f"数据缺少必要列: {date_column} 或 {target_column}，现有列: {list(df.columns)}"
        )

    df = df[[date_column, target_column]].copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df.sort_values(by=date_column, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

