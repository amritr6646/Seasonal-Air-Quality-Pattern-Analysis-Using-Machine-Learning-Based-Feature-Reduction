# Seasonal Air Quality Pattern Analysis

Streamlit prototype for exploring the user's actual Patna, Bihar air-quality and weather CSV, comparing seasonal patterns, selecting season-specific features, and evaluating Random Forest PM2.5 models.

## Selected dataset

The app uses `patna_aqi_weather_final_merged.csv`, already present in this workspace. It contains 29,899 hourly rows from 2022-08-04 05:00 through 2025-12-31 23:00, 19 source columns, no duplicate rows, no invalid timestamps, and no missing values. It includes PM2.5, PM10, CO, NO2, SO2, O3, and the available weather measurements. This was selected over `patna_aqi_weather_cleaned.csv` because it has 24 more rows and avoids the two supplied lagged PM2.5 columns, which could leak recent target values into a model comparison.

The merged CSV's pre-existing `Season` values disagree with the required month-based season rules on 11,760 rows. The app reports that data-quality issue and derives season from the parsed timestamp instead. The source CSV is not changed. The actual recorded period begins on August 4, not August 1.

## Run

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app expects the selected CSV beside `app.py`. It does not generate, alter, or substitute project data.

## Pages

- **Dashboard:** dataset metrics, workflow, derived seasonal PM2.5 means, and a compact OpenStreetMap street map centered on Patna with native zoom controls.
- **Dataset Overview:** shape, previews, types, missingness, duplicates, and descriptive statistics.
- **Seasonal Analysis:** seasonal means, monthly trends, distributions, and PM2.5/PM10 trends.
- **Correlation Analysis:** season filter, Pearson heatmap, and threshold-based correlated pairs.
- **Feature Reduction:** correlation filtering, Random Forest RFE, and Random Forest importance, run independently per season.
- **Machine Learning:** PM2.5 Random Forest models using all versus selected predictors, evaluated on later chronological observations.
- **Model Comparison:** seasonal and across-season metric comparisons.
- **Final Insights:** measured per-season importance, selected features, feature reduction, model choice, and metrics.

## Method notes

- Seasons are derived from month: Winter (Nov-Feb), Summer / Pre-Monsoon (Mar-May), Monsoon (Jun-Sep), Autumn / Post-Monsoon (Oct).
- The target is PM2.5. If that field is absent, modeling stops rather than silently switching targets.
- Candidate predictors are numeric source columns other than PM2.5. They include the present pollutant, weather, and calendar measurements; no columns are invented.
- Each season is sorted by time and split 80/20, with the latest 20% held out. Median imputation, feature selection, and model fitting use training observations only.
- Correlation selection retains the feature with the strongest absolute training correlation with PM2.5 first, then removes a candidate only when it is correlated at or above the chosen absolute threshold with an already retained feature. If the data do not contain redundant pairs at that threshold, this method may select most or all predictors.
- RFE and Random Forest importance select half of the candidate predictors (rounded up), using training data. This is a declared selection setting, not a result claim.
- Metrics (MAE, RMSE, R², and fit time) are calculated from each season's held-out period. Reduced features are not presumed to perform better.
- The CSV has no missing values; if that changes, descriptive summaries retain available measurements and model predictors are imputed from training-window medians.
- The Patna map loads labeled street/place tiles from OpenStreetMap and therefore needs an internet connection to show the map imagery and labels; map zoom and pan are interactive.

## Verify

```bash
pytest -q
```

Tests load the actual CSV and validate timestamp-derived seasons, feature-selection behavior, and model result structure. All displayed model scores are recalculated by the app from this CSV.