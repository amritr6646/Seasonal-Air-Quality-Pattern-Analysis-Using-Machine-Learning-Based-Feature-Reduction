from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "patna_aqi_weather_final_merged.csv"
TARGET = "PM2.5"
SEASONS = [
    "Winter",
    "Summer / Pre-Monsoon",
    "Monsoon",
    "Autumn / Post-Monsoon",
]
POLLUTANTS = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
MONTH_TO_SEASON = {
    11: "Winter",
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Summer / Pre-Monsoon",
    4: "Summer / Pre-Monsoon",
    5: "Summer / Pre-Monsoon",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Autumn / Post-Monsoon",
}


def _canonical_column(column: str) -> str:
    normalized = re.sub(r"\s+", " ", str(column).strip()).lower()
    exact = {
        "time": "Timestamp",
        "timestamp": "Timestamp",
        "date": "Timestamp",
        "datetime": "Timestamp",
        "pm2_5 (μg/m³)": "PM2.5",
        "pm2.5": "PM2.5",
        "pm2_5": "PM2.5",
        "pm10 (μg/m³)": "PM10",
        "pm10": "PM10",
        "carbon_monoxide (μg/m³)": "CO",
        "carbon monoxide": "CO",
        "co": "CO",
        "nitrogen_dioxide (μg/m³)": "NO2",
        "nitrogen dioxide": "NO2",
        "no2": "NO2",
        "sulphur_dioxide (μg/m³)": "SO2",
        "sulfur_dioxide (μg/m³)": "SO2",
        "sulphur dioxide": "SO2",
        "sulfur dioxide": "SO2",
        "so2": "SO2",
        "ozone (μg/m³)": "O3",
        "ozone": "O3",
        "o3": "O3",
        "temperature_2m (°c)": "Temperature",
        "temperature": "Temperature",
        "relative_humidity_2m (%)": "Humidity",
        "relative humidity": "Humidity",
        "humidity": "Humidity",
        "precipitation (mm)": "Rainfall",
        "precipitation": "Rainfall",
        "rainfall": "Rainfall",
        "surface_pressure (hpa)": "Pressure",
        "surface pressure": "Pressure",
        "pressure": "Pressure",
        "wind_speed_10m (km/h)": "Wind Speed",
        "wind speed": "Wind Speed",
        "wind_direction_10m (°)": "Wind Direction",
        "wind direction": "Wind Direction",
        "dayofweek": "DayOfWeek",
    }
    if normalized in exact:
        return exact[normalized]
    if normalized == "season":
        return "Source Season"
    return str(column).strip()


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the real CSV, normalize known labels, and derive seasons from time."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    raw = pd.read_csv(source_path)
    original_rows = len(raw)
    raw.columns = [_canonical_column(column) for column in raw.columns]
    timestamp_column = next(
        (column for column in raw.columns if column == "Timestamp"), None
    )
    if timestamp_column is None:
        raise ValueError("The dataset must contain a date/time column (for example, 'time').")

    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    invalid_dates = int(raw["Timestamp"].isna().sum())
    if invalid_dates:
        raw = raw.loc[raw["Timestamp"].notna()].copy()
    if raw.empty:
        raise ValueError("No valid timestamp rows remain after parsing the dataset.")

    source_season = raw.get("Source Season")
    derived_season = raw["Timestamp"].dt.month.map(MONTH_TO_SEASON)
    season_mismatches = 0
    if source_season is not None:
        normalized_source = source_season.astype("string").str.strip().str.lower()
        normalized_derived = derived_season.astype("string").str.strip().str.lower()
        season_mismatches = int((normalized_source != normalized_derived).sum())
        raw = raw.drop(columns=["Source Season"])
    raw["Season"] = derived_season
    raw = raw.sort_values("Timestamp").reset_index(drop=True)

    for column in raw.columns:
        if column not in {"Timestamp", "Season", "DayOfWeek"}:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    metadata: dict[str, object] = {
        "source_path": str(source_path),
        "source_rows": original_rows,
        "invalid_dates_removed": invalid_dates,
        "season_label_mismatches": season_mismatches,
        "duplicate_rows": int(raw.duplicated().sum()),
        "period_start": raw["Timestamp"].min(),
        "period_end": raw["Timestamp"].max(),
    }
    return raw, metadata


def available_pollutants(frame: pd.DataFrame) -> list[str]:
    return [column for column in POLLUTANTS if column in frame.columns]


def numeric_features(frame: pd.DataFrame, target: str = TARGET) -> list[str]:
    excluded = {target, "Timestamp", "Season", "DayOfWeek"}
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded and frame[column].notna().any()
    ]


def season_rows(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    if season == "All Seasons":
        return frame.copy()
    return frame.loc[frame["Season"] == season].copy()