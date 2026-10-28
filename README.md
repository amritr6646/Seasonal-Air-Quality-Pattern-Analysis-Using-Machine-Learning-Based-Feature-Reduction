# Seasonal Air Quality Pattern Analysis

A Streamlit app for exploring Patna, Bihar air-quality and weather data, comparing seasonal patterns, selecting season-specific features, and evaluating Random Forest PM2.5 models.

**Author:** [YOUR NAME]

## Dataset

The app uses `patna_aqi_weather_final_merged.csv`, which is included in this repository.

- **Rows:** 29,899 hourly records
- **Period:** 4 Aug 2022 to 31 Dec 2025 (data starts on 4 Aug 2022)
- **Columns:** 19, including PM2.5, PM10, CO, NO2, SO2, O3, and weather measurements
- **Quality:** no duplicate rows, no invalid timestamps, no missing values
- **Source:** [ADD DATA SOURCE, e.g. CPCB / weather API name and link]

This file was chosen over `patna_aqi_weather_cleaned.csv` because it has 24 more rows and does not include the two lagged PM2.5 columns, which could leak recent target values into the model comparison.

**Known data issue:** the `Season` column in the CSV disagrees with the month-based season rules below on 11,760 rows. The app reports this and derives the season from the timestamp instead. The source CSV is not modified.

## Run

```bash
git clone https://github.com/amritr6646/Seasonal-Air-Quality-Pattern-Analysis-Using-Machine-Learning-Based-Feature-Reduction
cd Seasonal-Air-Quality-Pattern-Analysis-Using-Machine-Learning-Based-Feature-Reduction

python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
streamlit run app.py
```

The app expects the CSV in the same folder as `app.py`. It does not generate, alter, or substitute project data.

## Pages

- **Dashboard:** dataset metrics, workflow, derived seasonal PM2.5 means, and a compact OpenStreetMap street map centered on Patna with zoom controls.
- **Dataset Overview:** shape, previews, types, missing values, duplicates, and descriptive statistics.
- **Seasonal Analysis:** seasonal means, monthly trends, distributions, and PM2.5/PM10 trends.
- **Correlation Analysis:** season filter, Pearson heatmap, and threshold-based correlated pairs.
- **Feature Reduction:** correlation filtering, Random Forest RFE, and Random Forest importance, run independently for each season.
- **Machine Learning:** PM2.5 Random Forest models using all predictors versus selected predictors, evaluated on later chronological observations.
- **Model Comparison:** seasonal and across-season metric comparisons.
- **Final Insights:** per-season feature importance, selected features, feature reduction, model choice, and metrics.

## Results

- Highest average PM2.5 season: [SEASON, VALUE]
- Lowest average PM2.5 season: [SEASON, VALUE]
- Best feature set / method: [e.g. all features vs RFE, with R² and RMSE]

![App screenshot](screenshots/dashboard.png)

## Method notes

- **Seasons** are derived from the month:
  - Winter: Nov to Feb
  - Summer / Pre-Monsoon: Mar to May
  - Monsoon: Jun to Sep
  - Autumn / Post-Monsoon: Oct only
  - Note: Autumn is limited to October in this project. The standard IMD post-monsoon season is Oct to Dec.
- **Target** is PM2.5. If that column is missing, modeling stops instead of silently switching targets.
- **Candidate predictors** are all numeric columns other than PM2.5 (pollutants, weather, and calendar fields). No columns are invented.
- **Train/test split:** each season is sorted by time and split 80/20, with the latest 20% held out. Median imputation, feature selection, and model fitting use training data only.
- **Correlation selection** keeps the feature most correlated with PM2.5 first, then removes a candidate only if it is correlated at or above the chosen threshold with an already retained feature. If the data has no redundant pairs at that threshold, most or all predictors may be kept.
- **RFE and Random Forest importance** each select half of the candidate predictors (rounded up) using training data. The 50% size is a fixed setting, not a result.
- **Metrics** (MAE, RMSE, R², fit time) are calculated on each season's held-out period. Reduced feature sets are not assumed to perform better.
- **Missing values:** the CSV has none. If that changes, descriptive summaries use the available measurements and model predictors are imputed with training-window medians.
- **Map:** the Patna map loads tiles from OpenStreetMap, so it needs an internet connection. Zoom and pan are interactive.

## Limitations

- The held-out 20% of each season is its latest period only, so scores may not represent every year.
- Same-hour pollutants such as PM10 are strong predictors of PM2.5, which can make models look better than they would in real forecasting.
- Results come from a single city and one monitoring dataset.

## Tests

```bash
pytest -q
```

The tests load the actual CSV and validate timestamp-derived seasons, feature-selection behavior, and model result structure. All displayed model scores are recalculated by the app from this CSV.

## License

[ADD LICENSE, e.g. MIT]