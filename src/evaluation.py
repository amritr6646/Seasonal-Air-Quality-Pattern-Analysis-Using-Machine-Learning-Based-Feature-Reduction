from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R²": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
    }


def comparison_table(records: list[dict[str, object]]) -> pd.DataFrame:
    columns = ["Season", "Model", "Feature Count", "MAE", "RMSE", "R²", "Training Time (s)"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records).reindex(columns=columns)