from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

from src.preprocessing import TARGET


METHODS = ["Correlation", "RFE", "Random Forest Importance"]


def chronological_split(
    frame: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    minimum_rows: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    usable = frame.dropna(subset=[target]).sort_values("Timestamp")
    if len(usable) < minimum_rows:
        raise ValueError(f"At least {minimum_rows} target observations are required; found {len(usable)}.")
    split_at = int(len(usable) * 0.8)
    if split_at < 2 or len(usable) - split_at < 2:
        raise ValueError("Not enough rows for chronological training and test partitions.")

    train = usable.iloc[:split_at]
    test = usable.iloc[split_at:]
    x_train = _impute(train[features], train[features].median(numeric_only=True))
    x_test = _impute(test[features], train[features].median(numeric_only=True))
    return x_train, x_test, train[target].astype(float), test[target].astype(float)


def _impute(values: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    filled = values.copy()
    for column in filled.columns:
        median = medians.get(column, np.nan)
        filled[column] = filled[column].fillna(median if pd.notna(median) else 0.0)
    return filled


def _forest(seed: int = 42, trees: int = 100) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def select_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    threshold: float = 0.85,
) -> tuple[list[str], pd.DataFrame]:
    """Select features using training data only; return selections and scores."""
    features = list(x_train.columns)
    if not features:
        return [], pd.DataFrame(columns=["Feature", "Score"])

    if method == "Correlation":
        nonconstant_features = [feature for feature in features if x_train[feature].nunique(dropna=True) > 1]
        correlations = x_train[nonconstant_features].corr().abs().fillna(0.0)
        target_corr = x_train.apply(lambda values: values.corr(y_train)).abs().fillna(0.0)
        keep: list[str] = []
        for feature in sorted(features, key=lambda item: (-target_corr[item], features.index(item))):
            if all(
                feature not in correlations.index
                or retained not in correlations.columns
                or correlations.loc[feature, retained] < threshold
                for retained in keep
            ):
                keep.append(feature)
        scores = pd.DataFrame({"Feature": features, "Score": target_corr.values})
        return keep or [features[0]], scores.sort_values("Score", ascending=False)

    selected_count = max(1, ceil(len(features) / 2))
    if method == "RFE":
        selector = RFE(
            estimator=_forest(trees=40),
            n_features_to_select=selected_count,
            step=1,
        )
        selector.fit(x_train, y_train)
        ranks = pd.DataFrame({"Feature": features, "Score": selector.ranking_})
        chosen = ranks.loc[ranks["Score"] == 1, "Feature"].tolist()
        return chosen, ranks.sort_values("Score")

    model = _forest(trees=120)
    model.fit(x_train, y_train)
    scores = pd.DataFrame({"Feature": features, "Score": model.feature_importances_})
    chosen = scores.nlargest(selected_count, "Score")["Feature"].tolist()
    return chosen, scores.sort_values("Score", ascending=False)


def selection_report(
    frame: pd.DataFrame,
    features: list[str],
    method: str,
    threshold: float,
) -> dict[str, object]:
    x_train, _, y_train, _ = chronological_split(frame, features)
    selected, scores = select_features(x_train, y_train, method, threshold)
    return {
        "original_features": features,
        "selected_features": selected,
        "removed_features": [feature for feature in features if feature not in selected],
        "scores": scores,
        "train_rows": len(x_train),
    }