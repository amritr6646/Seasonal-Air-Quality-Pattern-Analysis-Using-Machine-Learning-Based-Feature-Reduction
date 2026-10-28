from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation import comparison_table
from src.feature_selection import METHODS
from src.pages import (
    get_evaluations,
    render_comparison,
    render_correlation,
    render_dashboard,
    render_feature_reduction,
    render_insights,
    render_machine_learning,
    render_overview,
    render_seasonal,
)
from src.preprocessing import DEFAULT_DATA_PATH, available_pollutants, load_dataset, numeric_features


PAGES = [
    "Dashboard",
    "Dataset Overview",
    "Seasonal Analysis",
    "Correlation Analysis",
    "Feature Reduction",
    "Machine Learning",
    "Model Comparison",
    "Final Insights",
]

APP_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#172a26; --muted:#64736d; --paper:#f4f6f1; --line:#dce3db; --forest:#176b53; }
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:radial-gradient(ellipse at 92% 0%,#e8efe2 0,transparent 34%),var(--paper)}
[data-testid="stSidebar"]{background:#eaf0e8;border-right:1px solid var(--line)}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:0!important;color:var(--ink)}
h1{font-size:2rem!important;font-weight:800!important} h2{font-size:1.35rem!important;font-weight:750!important} h3{font-size:1rem!important;font-weight:700!important}
.eyebrow{font:500 .72rem 'DM Mono',monospace;color:var(--forest);text-transform:uppercase;letter-spacing:.04em}
.hero{border-bottom:1px solid var(--line);padding:1.1rem 0 1.35rem;margin-bottom:1.5rem}
.hero-title{font:800 2.15rem/1.12 'Manrope',sans-serif;max-width:900px;margin:.4rem 0 .45rem}
.hero-sub{color:var(--muted);font-size:.96rem}
.metric{background:rgba(255,255,255,.72);border:1px solid var(--line);border-top:3px solid var(--forest);padding:1rem 1.1rem;min-height:105px}
.metric-label{color:var(--muted);font:500 .72rem 'DM Mono',monospace;text-transform:uppercase}
.metric-value{font:800 1.65rem 'Manrope',sans-serif;margin-top:.35rem;color:var(--ink)}
.metric-note{color:var(--muted);font-size:.78rem;margin-top:.1rem}
.workflow{display:flex;flex-wrap:wrap;align-items:stretch;gap:.45rem;margin:.7rem 0 1.4rem}
.workflow-step{background:#fff;border:1px solid var(--line);border-radius:4px;padding:.65rem .78rem;font-size:.82rem;font-weight:600}
.workflow-arrow{align-self:center;color:var(--forest);font-weight:700}
div[data-testid="stMetric"]{background:rgba(255,255,255,.7);border:1px solid var(--line);padding:.8rem;border-radius:3px}
.stButton button,.stDownloadButton button{border-radius:3px}
</style>
"""


@st.cache_data(show_spinner="Reading the selected Patna dataset...")
def get_dataset(path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    return load_dataset(path)


def _calculate_results(frame: pd.DataFrame, features: list[str]):
    method = st.selectbox("Feature selection method", METHODS, key="shared-model-method")
    threshold = st.select_slider("Correlation threshold", [0.80, 0.85, 0.90, 0.95], value=0.85, key="shared-model-threshold")
    with st.spinner("Calculating real holdout results..."):
        records, predictions, _ = get_evaluations(frame, tuple(features), method, threshold)
    results = comparison_table(records)
    st.session_state.update(model_results=results, model_predictions=predictions, model_method=method)
    return results, predictions, method


def main() -> None:
    st.set_page_config(
        page_title="Patna Air Quality | Seasonal Intelligence",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_STYLE, unsafe_allow_html=True)

    try:
        frame, metadata = get_dataset(str(DEFAULT_DATA_PATH))
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to load the selected dataset: {error}")
        st.stop()

    pollutants = available_pollutants(frame)
    features = numeric_features(frame)
    st.sidebar.markdown("<div class='eyebrow'>FIELD NOTES / PATNA</div>", unsafe_allow_html=True)
    st.sidebar.title("Seasonal Air Quality")
    st.sidebar.caption("B.Tech Major Project · Bihar, India")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(f"Source · {Path(str(metadata['source_path'])).name}")
    st.sidebar.caption(f"{len(frame):,} hourly records · {len(features)} numeric predictors")

    if page == "Dashboard":
        render_dashboard(frame, metadata, pollutants, features)
    elif page == "Dataset Overview":
        render_overview(frame, metadata)
    elif page == "Seasonal Analysis":
        render_seasonal(frame, pollutants, features)
    elif page == "Correlation Analysis":
        render_correlation(frame, features, pollutants)
    elif page == "Feature Reduction":
        render_feature_reduction(frame, features)
    elif page == "Machine Learning":
        results, predictions, errors, method, _ = render_machine_learning(frame, features)
        st.session_state.update(
            model_results=results,
            model_predictions=predictions,
            model_errors=errors,
            model_method=method,
        )
    elif page == "Model Comparison":
        results = st.session_state.get("model_results", pd.DataFrame())
        if results.empty:
            results, _, _ = _calculate_results(frame, features)
        render_comparison(results)
    else:
        results = st.session_state.get("model_results", pd.DataFrame())
        predictions = st.session_state.get("model_predictions", {})
        method = st.session_state.get("model_method", METHODS[0])
        if results.empty or not predictions:
            results, predictions, method = _calculate_results(frame, features)
        render_insights(results, predictions, method)


if __name__ == "__main__":
    main()