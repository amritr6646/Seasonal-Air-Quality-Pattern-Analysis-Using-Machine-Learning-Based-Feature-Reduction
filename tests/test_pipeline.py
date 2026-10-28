from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import comparison_table
from src.feature_selection import METHODS, selection_report
from src.models import evaluate_all
from src.preprocessing import (
    DEFAULT_DATA_PATH,
    SEASONS,
    TARGET,
    load_dataset,
    numeric_features,
)


def test_actual_dataset_is_loaded_and_seasons_are_derived():
    frame, metadata = load_dataset(DEFAULT_DATA_PATH)

    assert len(frame) == 29_899
    assert TARGET in frame.columns
    assert set(frame["Season"].dropna()) == set(SEASONS)
    assert metadata["invalid_dates_removed"] == 0
    assert metadata["duplicate_rows"] == 0
    assert metadata["season_label_mismatches"] == 11_760
    assert frame.loc[frame["Timestamp"].dt.month == 10, "Season"].eq("Autumn / Post-Monsoon").all()


def test_each_feature_selection_method_returns_observed_columns():
    frame, _ = load_dataset(DEFAULT_DATA_PATH)
    features = numeric_features(frame)
    winter = frame.loc[frame["Season"] == "Winter"]

    for method in METHODS:
        report = selection_report(winter, features, method, 0.85)
        assert report["selected_features"]
        assert set(report["selected_features"]).issubset(features)
        assert set(report["removed_features"]).issubset(features)
        assert not set(report["selected_features"]) & set(report["removed_features"])


def test_models_return_real_results_for_every_season():
    frame, _ = load_dataset(DEFAULT_DATA_PATH)
    records, predictions, errors = evaluate_all(
        frame,
        numeric_features(frame),
        "Random Forest Importance",
        0.85,
    )
    results = comparison_table(records)

    assert results.shape[0] == 8
    assert set(results["Season"]) == set(SEASONS)
    assert set(results["Model"]) == {"All Features", "Reduced Features"}
    assert results[["MAE", "RMSE", "R²"]].notna().all().all()
    assert not errors
    assert set(predictions) == set(SEASONS)
    assert all(len(details["actual"]) == details["test_rows"] for details in predictions.values())
    assert all("importance" in details for details in predictions.values())