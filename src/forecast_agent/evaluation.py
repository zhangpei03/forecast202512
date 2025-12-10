from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float

    def to_dict(self) -> Dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "mape": self.mape}


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Metrics:
    y_true, y_pred = y_true.reset_index(drop=True), y_pred.reset_index(drop=True)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return Metrics(mae=mae, rmse=rmse, mape=mape)

