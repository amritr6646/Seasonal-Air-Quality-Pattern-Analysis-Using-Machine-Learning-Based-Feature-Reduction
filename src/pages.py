from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src.evaluation import comparison_table
from src.feature_selection import METHODS, selection_report
from src.models import evaluate_all
from src.preprocessing import SEASONS, TARGET, season_rows
from src.seasonal_analysis import monthly_means, seasonal_means


@st.cache_data(show_spinner="Calculating season-specific feature selections...")
def get_selection(frame: pd.DataFrame, features: tuple[str, ...], method: str, threshold: float):
    reports, errors = {}, {}
    for season in SEASONS:
        try:
            reports[season] = selection_report(frame.loc[frame["Season"] == season], list(features), method, threshold)
        except (ValueError, KeyError) as error:
            errors[season] = str(error)
    return reports, errors


@st.cache_data(show_spinner="Training season-specific Random Forest models...")
def get_evaluations(frame: pd.DataFrame, features: tuple[str, ...], method: str, threshold: float):
    return evaluate_all(frame, list(features), method, threshold)


def _heading(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">PATNA AIR QUALITY / ANALYSIS</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def render_dashboard(frame: pd.DataFrame, metadata: dict[str, object], pollutants: list[str], features: list[str]) -> None:
    start = pd.Timestamp(metadata["period_start"]).strftime("%b %Y")
    end = pd.Timestamp(metadata["period_end"]).strftime("%b %Y")
    st.markdown(
        '<div class="hero"><div class="eyebrow">B.TECH MAJOR PROJECT · PATNA, BIHAR</div>'
        '<div class="hero-title">Seasonal Air Quality Pattern Analysis</div>'
        f'<div class="hero-sub">Machine learning-based feature reduction · {start} – {end} · Hourly observations</div></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    metrics = [
        ("Total records", f"{len(frame):,}", "observed hourly rows"),
        ("Total features", str(frame.shape[1]), f"{len(features)} numeric predictors"),
        ("Pollutants", str(len(pollutants)), "present in source CSV"),
        ("Seasons", str(frame["Season"].nunique()), "derived from month"),
    ]
    for column, (label, value, note) in zip(columns, metrics):
        with column:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.subheader("Project workflow")
    steps = ["Dataset", "Preprocessing", "Season classification", "Seasonal analysis", "Feature reduction", "Random Forest", "Model comparison", "Final insights"]
    workflow = '<div class="workflow">'
    for index, step in enumerate(steps):
        workflow += ('<span class="workflow-arrow">→</span>' if index else "")
        workflow += f'<span class="workflow-step">{step}</span>'
    st.markdown(workflow + "</div>", unsafe_allow_html=True)

    st.subheader("Study location")
    map_column, details_column = st.columns([1.2, 1])
    with map_column:
        location_map = folium.Map(
            location=[25.60, 85.10], zoom_start=6, tiles="OpenStreetMap",
            control_scale=True, zoom_control=True, height=340,
        )
        folium.Marker(
            [25.60, 85.10], tooltip="Patna",
            popup="Patna, Bihar, India · 25.60°N, 85.10°E",
            icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
        ).add_to(location_map)
        st_folium(location_map, height=340, use_container_width=True, returned_objects=[], key="patna-location-map")
    with details_column:
        st.markdown("**Patna, Bihar, India**")
        st.caption("25.60°N, 85.10°E · approximate project coordinates")
        st.caption("OpenStreetMap shows nearby place labels. Use `+` and `−` or scroll to zoom.")

    if metadata["season_label_mismatches"]:
        st.warning(
            f"{metadata['season_label_mismatches']:,} source season labels disagree with the required month rules. "
            "Season is recalculated from each timestamp."
        )
    st.subheader("Seasonal PM2.5 profile")
    if TARGET in frame:
        summary = seasonal_means(frame, [TARGET]).dropna()
        chart_column, table_column = st.columns([1.5, 1])
        with chart_column:
            chart = px.bar(summary, x="Season", y=TARGET, color="Season", text_auto=".1f", title="Mean PM2.5 by derived season")
            chart.update_layout(showlegend=False, xaxis_title=None, yaxis_title="PM2.5 (dataset units)")
            st.plotly_chart(chart, use_container_width=True)
        with table_column:
            st.dataframe(summary, hide_index=True, use_container_width=True)
            st.caption("Calculated from non-missing observations in the selected CSV.")


def render_overview(frame: pd.DataFrame, metadata: dict[str, object]) -> None:
    _heading("Dataset overview", "A transparent profile of the chosen source file and preprocessing decisions.")
    st.caption(f"Selected dataset: `{Path(str(metadata['source_path'])).name}` · {metadata['source_path']}")
    columns = st.columns(4)
    values = [f"{len(frame):,}", str(frame.shape[1]), str(metadata["duplicate_rows"]), str(metadata["invalid_dates_removed"])]
    for column, label, value in zip(columns, ["Rows", "Columns", "Duplicate rows", "Invalid dates removed"], values):
        with column:
            st.metric(label, value)
    if metadata["season_label_mismatches"]:
        st.warning(f"{metadata['season_label_mismatches']:,} source season labels did not match month-based seasons; timestamps are used instead.")
    tabs = st.tabs(["Data preview", "Column types", "Missing values", "Statistics"])
    with tabs[0]:
        st.dataframe(frame.head(25), hide_index=True, use_container_width=True)
        st.caption(f"Showing the first 25 of {len(frame):,} processed rows.")
    with tabs[1]:
        types = pd.DataFrame({"Column": frame.columns, "Data type": [str(dtype) for dtype in frame.dtypes], "Non-null rows": frame.notna().sum().values})
        st.dataframe(types, hide_index=True, use_container_width=True)
    with tabs[2]:
        missing = frame.isna().sum().rename("Missing values").to_frame()
        missing["Missing %"] = (missing["Missing values"] / max(len(frame), 1) * 100).round(2)
        st.dataframe(missing, use_container_width=True)
        st.caption("Summaries use available values. Modeling fills predictor gaps from training medians.")
    with tabs[3]:
        st.dataframe(frame.select_dtypes(include="number").describe().T, use_container_width=True)


def render_seasonal(frame: pd.DataFrame, pollutants: list[str], features: list[str]) -> None:
    _heading("Seasonal analysis", "Compare pollutant and meteorological patterns across Patna’s four month-defined seasons.")
    available = [column for column in dict.fromkeys(pollutants + features) if column in frame.columns]
    selected = st.selectbox("Parameter", available, index=available.index(TARGET) if TARGET in available else 0)
    season_summary = seasonal_means(frame, [selected]).dropna()
    month_summary = monthly_means(frame, [selected]).dropna()

    for tab, name in zip(st.tabs(["Seasonal comparison", "Monthly pattern", "Distributions", "PM trends"]), ["season", "month", "distribution", "trends"]):
        with tab:
            if name == "season":
                if season_summary.empty:
                    st.info("No usable observations are available for this parameter.")
                else:
                    chart = px.bar(season_summary, x="Season", y=selected, color="Season", text_auto=".2f", title=f"Seasonal mean · {selected}")
                    chart.update_layout(showlegend=False, xaxis_title=None, yaxis_title=f"{selected} · source units")
                    st.plotly_chart(chart, use_container_width=True)
                    st.dataframe(season_summary, hide_index=True, use_container_width=True)
            elif name == "month":
                chart = px.line(month_summary, x="Month", y=selected, markers=True, title=f"Monthly mean · {selected}")
                st.plotly_chart(chart, use_container_width=True)
            elif name == "distribution":
                chart = px.box(frame.dropna(subset=[selected]), x="Season", y=selected, color="Season", points="outliers", category_orders={"Season": SEASONS}, title=f"Observed distribution · {selected}")
                st.plotly_chart(chart, use_container_width=True)
            else:
                targets = [column for column in ["PM2.5", "PM10"] if column in frame.columns]
                if targets:
                    trend = monthly_means(frame, targets).melt(id_vars="Month", var_name="Pollutant", value_name="Monthly mean")
                    st.plotly_chart(px.line(trend, x="Month", y="Monthly mean", color="Pollutant", markers=True, title="Monthly particulate matter trend"), use_container_width=True)
                else:
                    st.info("Neither PM2.5 nor PM10 is available in this dataset.")


def render_correlation(frame: pd.DataFrame, features: list[str], pollutants: list[str]) -> None:
    _heading("Correlation analysis", "Measure pairwise linear association and flag redundant predictors within the selected time group.")
    season = st.selectbox("Season", ["All Seasons", *SEASONS], key="correlation-season")
    threshold = st.select_slider("Redundancy threshold · absolute Pearson r", [0.80, 0.85, 0.90, 0.95], value=0.85)
    scoped = season_rows(frame, season)
    columns = list(dict.fromkeys(column for column in [*pollutants, *features] if column in scoped.columns))
    numeric = scoped[columns].select_dtypes(include="number")
    if len(numeric.columns) < 2:
        st.info("At least two numeric parameters are needed to calculate correlations.")
        return
    corr = numeric.corr()
    heatmap = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True, colorbar_title="Pearson r", hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
    heatmap.update_layout(title=f"Pearson correlation · {season}", height=max(520, len(corr.columns) * 35))
    st.plotly_chart(heatmap, use_container_width=True)
    pairs = [
        {"Feature 1": left, "Feature 2": right, "Pearson r": corr.loc[left, right], "Absolute r": abs(corr.loc[left, right])}
        for index, left in enumerate(corr.columns)
        for right in corr.columns[index + 1:]
        if pd.notna(corr.loc[left, right]) and abs(corr.loc[left, right]) >= threshold
    ]
    st.subheader("Highly correlated pairs")
    if pairs:
        st.dataframe(pd.DataFrame(pairs).sort_values("Absolute r", ascending=False), hide_index=True, use_container_width=True)
        st.caption("Correlation indicates overlapping linear information, not causation.")
    else:
        st.info(f"No feature pairs meet |r| ≥ {threshold:.2f} for {season.lower()}.")


def render_feature_reduction(frame: pd.DataFrame, features: list[str]) -> None:
    _heading("Feature reduction", "Feature selection is fit on each season’s training window, never its test period.")
    method = st.selectbox("Selection method", METHODS, key="feature-method")
    threshold = st.select_slider("Correlation threshold", [0.80, 0.85, 0.90, 0.95], value=0.85, key="feature-threshold")
    season = st.selectbox("Season", SEASONS, key="feature-season")
    reports, errors = get_selection(frame, tuple(features), method, threshold)
    if season in errors:
        st.warning(f"{season}: {errors[season]}")
        return
    report = reports[season]
    left, right = st.columns(2)
    left.metric("Original feature count", len(report["original_features"]))
    right.metric("Selected feature count", len(report["selected_features"]))
    st.caption(f"{method} · {report['train_rows']:,} training observations")
    left, right = st.columns(2)
    with left:
        st.subheader("Selected features")
        st.dataframe(pd.DataFrame({"Feature": report["selected_features"]}), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Removed features")
        st.dataframe(pd.DataFrame({"Feature": report["removed_features"] or ["None"]}), hide_index=True, use_container_width=True)
    chart = px.bar(report["scores"], x="Score", y="Feature", orientation="h", title="Training-window feature score / ranking")
    chart.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(350, len(report["scores"]) * 24))
    st.plotly_chart(chart, use_container_width=True)
    with st.expander("Compare all methods for this season"):
        for candidate_method in METHODS:
            candidate = report if candidate_method == method else get_selection(frame, tuple(features), candidate_method, threshold)[0].get(season)
            if candidate:
                st.write(f"**{candidate_method}:** {len(candidate['selected_features'])} of {len(features)} features")
                st.caption(", ".join(candidate["selected_features"]))


def render_machine_learning(frame: pd.DataFrame, features: list[str]):
    _heading("Machine learning", "Season-specific Random Forest regression with an 80/20 chronological holdout.")
    st.caption("Predict PM2.5 from other numeric columns. Feature selection and imputation use training data only.")
    if TARGET not in frame:
        st.error("PM2.5 is missing. No alternate target will be selected.")
        return pd.DataFrame(), {}, {}, METHODS[0], 0.85
    method = st.selectbox("Reduced-feature model selector", METHODS, key="model-method")
    threshold = st.select_slider("Correlation threshold", [0.80, 0.85, 0.90, 0.95], value=0.85, key="model-threshold")
    with st.spinner("Training and evaluating seasonal models..."):
        records, predictions, errors = get_evaluations(frame, tuple(features), method, threshold)
    results = comparison_table(records)
    if results.empty:
        st.warning("No season had sufficient observations for model evaluation.")
        return results, predictions, errors, method, threshold
    st.dataframe(results.round(4), hide_index=True, use_container_width=True)
    st.download_button("Download evaluation table", results.to_csv(index=False).encode(), "patna_seasonal_model_results.csv", "text/csv")
    season = st.selectbox("Prediction view", list(predictions), key="prediction-season")
    details = predictions[season]
    plot = pd.DataFrame({"Timestamp": details["timestamp"], "Observed PM2.5": details["actual"], "All Features": details["All Features"], "Reduced Features": details["Reduced Features"]})
    st.plotly_chart(px.line(plot, x="Timestamp", y=plot.columns[1:], title=f"Observed and predicted PM2.5 · {season}"), use_container_width=True)
    st.caption(f"Train: {details['train_rows']:,} · Test: {details['test_rows']:,} · Selected: {', '.join(details['selected_features'])}")
    for season_name, error in errors.items():
        st.info(f"{season_name} was skipped: {error}")
    return results, predictions, errors, method, threshold


def render_comparison(results: pd.DataFrame) -> None:
    _heading("Model comparison", "Compare measured holdout performance for All Features and Reduced Features.")
    if results.empty:
        st.info("No evaluated results are available yet.")
        return
    seasons = [season for season in SEASONS if season in results["Season"].values]
    selected = st.selectbox("Comparison scope", ["Overall (mean by season)", *seasons], key="comparison-season")
    if selected == "Overall (mean by season)":
        view = results.groupby("Model", as_index=False)[["Feature Count", "MAE", "RMSE", "R²", "Training Time (s)"]].mean()
        view.insert(0, "Season", "Mean across evaluated seasons")
    else:
        view = results.loc[results["Season"] == selected].copy()
    st.dataframe(view.round(4), hide_index=True, use_container_width=True)
    metrics = ["MAE", "RMSE", "R²", "Feature Count", "Training Time (s)"]
    for tab, metric in zip(st.tabs(["MAE", "RMSE", "R²", "Feature count", "Training time"]), metrics):
        with tab:
            chart = px.bar(view, x="Model", y=metric, color="Model", text_auto=".3f", title=f"{metric} · {selected}")
            st.plotly_chart(chart, use_container_width=True)
    st.caption("Overall values are unweighted means of the season-level test metrics.")


def render_insights(results: pd.DataFrame, predictions: dict[str, dict[str, object]], method: str) -> None:
    _heading("Final insights", "A data-derived summary of selected predictors and out-of-sample results.")
    if results.empty:
        st.info("No calculated model results are available yet.")
        return
    criteria = {"Lowest MAE": ("MAE", True), "Lowest RMSE": ("RMSE", True), "Highest R²": ("R²", False)}
    criterion = st.selectbox("Best model criterion", list(criteria), key="insight-criterion")
    metric, ascending = criteria[criterion]
    for season in SEASONS:
        season_results = results.loc[results["Season"] == season]
        if season_results.empty or season not in predictions:
            st.warning(f"{season}: no result available; there may not be enough observations.")
            continue
        details = predictions[season]
        best = season_results.sort_values(metric, ascending=ascending).iloc[0]
        important = details["importance"].sort_values("Score", ascending=False)["Feature"].head(5).tolist()
        selected = details["selected_features"]
        count = int(season_results.loc[season_results["Model"] == "All Features", "Feature Count"].iloc[0])
        with st.expander(season, expanded=True):
            st.write(f"**Most important features:** {', '.join(important)}")
            st.write(f"**Selected by {method}:** {', '.join(selected)}")
            st.write(f"**Feature reduction:** {count} → {len(selected)} ({count - len(selected)} removed)")
            st.write(f"**Best model:** {best['Model']} · MAE {best['MAE']:.3f} · RMSE {best['RMSE']:.3f} · R² {best['R²']:.3f}")
