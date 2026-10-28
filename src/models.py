from __future__ import annotations

from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluation import regression_metrics
from src.feature_selection import chronological_split, select_features
from src.preprocessing import SEASONS, TARGET


def evaluate_season(
    frame: pd.DataFrame,
    season: str,
    features: list[str],
    method: str,
    threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seasonal = frame.loc[frame["Season"] == season].copy()
    x_train, x_test, y_train, y_test = chronological_split(seasonal, features)
    selected, importance = select_features(x_train, y_train, method, threshold)
    if not selected:
        raise ValueError("Feature selection returned no usable predictors.")

    records: list[dict[str, object]] = []
    predictions: dict[str, object] = {
        "actual": y_test.reset_index(drop=True),
        "timestamp": seasonal.loc[x_test.index, "Timestamp"].reset_index(drop=True),
        "selected_features": selected,
        "selector_scores": importance,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    for model_name, model_features in (("All Features", features), ("Reduced Features", selected)):
        model = RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        start = perf_counter()
        model.fit(x_train[model_features], y_train)
        elapsed = perf_counter() - start
        predicted = model.predict(x_test[model_features])
        if model_name == "All Features":
            predictions["importance"] = pd.DataFrame(
                {"Feature": model_features, "Score": model.feature_importances_}
            ).sort_values("Score", ascending=False)
        row: dict[str, object] = {
            "Season": season,
            "Model": model_name,
            "Feature Count": len(model_features),
            "Training Time (s)": elapsed,
            **regression_metrics(y_test, predicted),
        }
        records.append(row)
        predictions[model_name] = predicted
    return records, predictions


def evaluate_all(
    frame: pd.DataFrame,
    features: list[str],
    method: str,
    threshold: float,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], dict[str, str]]:
    records: list[dict[str, object]] = []
    prediction_data: dict[str, dict[str, object]] = {}
    errors: dict[str, str] = {}
    if TARGET not in frame.columns:
        return records, prediction_data, {"Dataset": "PM2.5 is not available; modeling is disabled."}

    for season in SEASONS:
        try:
            season_records, details = evaluate_season(frame, season, features, method, threshold)
            records.extend(season_records)
            prediction_data[season] = details
        except (ValueError, KeyError) as error:
            errors[season] = str(error)
    return records, prediction_data, errors