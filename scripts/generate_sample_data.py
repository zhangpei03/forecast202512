from pathlib import Path
import numpy as np
import pandas as pd


def generate(path: Path, periods: int = 180) -> None:
    rng = pd.date_range(start="2024-01-01", periods=periods, freq="D")
    trend = np.linspace(50, 120, periods)
    season = 8 * np.sin(np.linspace(0, 6.28, periods))
    noise = np.random.normal(0, 3, periods)
    values = trend + season + noise
    df = pd.DataFrame({"date": rng, "value": values})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    print(f"样例数据已写入: {path}")


if __name__ == "__main__":
    generate(Path("data/sample_data.xlsx"))

