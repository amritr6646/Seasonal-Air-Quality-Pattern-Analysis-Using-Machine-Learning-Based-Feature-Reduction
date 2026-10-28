from __future__ import annotations

import pandas as pd

from src.preprocessing import SEASONS


def seasonal_means(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.DataFrame()
    return (
        frame.groupby("Season", observed=False)[available]
        .mean()
        .reindex(SEASONS)
        .rename_axis("Season")
        .reset_index()
    )


def monthly_means(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.DataFrame()
    monthly = frame.set_index("Timestamp")[available].resample("MS").mean()
    return monthly.rename_axis("Month").reset_index()