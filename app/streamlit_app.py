"""Real Estate Market Intelligence & Price Analysis Terminal.

A refined analytics product and predictive valuation interface
derived from King County, Washington residential property transactions.

Run from the project root (REAL_ESTATE/):
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Configure project root path for module resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import app_utils, config, predict, train

# Page Configuration - Clean title, no emojis
st.set_page_config(
    page_title="Market Intelligence | Real Estate Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refined Editorial Design System
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --oxblood: #6D2438;
        --oxblood-dark: #501827;
        --bg-main: #F5F0E8;
        --bg-card: #FFFDF8;
        --text-main: #302A2B;
        --text-muted: #746A67;
        --border-color: #DED5CC;
        --accent-sand: #D9C7B9;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F5F0E8 !important;
        color: #302A2B !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Hide Streamlit default decoration header */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Streamlit's fixed top bar: transparent so it doesn't paint a gap over the header */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] span {
        color: #FFFDF8 !important;
    }

    /* Main content area: no top/bottom gap so header & footer sit flush */
    [data-testid="stMainBlockContainer"], .main .block-container {
        max-width: 100% !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    /* ---------- Site Header (full-width band) ---------- */
    .brand-banner {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, #3D101C 0%, #501827 45%, #6D2438 100%);
        margin: 0 -2.5rem 1.75rem -2.5rem;
        padding: 2.4rem 2.5rem 1.9rem 2.5rem;
        border-bottom: 3px solid #D9C7B9;
    }

    .brand-banner::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -4%;
        width: 520px;
        height: 520px;
        background: radial-gradient(circle, rgba(217, 199, 185, 0.20) 0%, rgba(217, 199, 185, 0) 65%);
        pointer-events: none;
    }

    .brand-banner .brand-eyebrow {
        color: #E3CFC0 !important;
    }

    .brand-banner .brand-title {
        color: #FFFDF8 !important;
    }

    .brand-banner .brand-subtitle {
        color: rgba(255, 253, 248, 0.78) !important;
    }

    .brand-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
        position: relative;
        z-index: 1;
    }

    .brand-chip {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        color: #FFFDF8;
        background: rgba(255, 253, 248, 0.10);
        border: 1px solid rgba(255, 253, 248, 0.22);
        border-radius: 999px;
        padding: 4px 12px;
    }

    .brand-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #746A67;
        margin-bottom: 6px;
    }

    .brand-title {
        font-family: 'Newsreader', Georgia, Cambria, serif;
        font-size: 2.35rem;
        font-weight: 500;
        color: #302A2B;
        letter-spacing: -0.015em;
        margin: 0 0 4px 0;
        line-height: 1.15;
    }

    .brand-subtitle {
        font-size: 0.92rem;
        color: #746A67;
        margin: 0;
        line-height: 1.4;
    }

    /* ---------- Left Sidebar Menu ---------- */
    [data-testid="stSidebar"] {
        background-color: #FFFDF8 !important;
        border-right: 1px solid #DED5CC !important;
    }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] span {
        color: #746A67 !important;
    }

    .sidebar-brand {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #6D2438;
        line-height: 1.2;
        margin-bottom: 2px;
    }

    .sidebar-tagline {
        font-size: 0.72rem;
        color: #746A67;
        letter-spacing: 0.04em;
        padding-bottom: 14px;
        margin-bottom: 14px;
        border-bottom: 1px solid #EFE8DE;
    }

    .sidebar-section {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9C8F88;
        margin: 4px 0 6px 2px;
    }

    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has([role="radiogroup"]),
    [data-testid="stSidebar"] [data-testid="stRadio"],
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        width: 100% !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
        width: 100% !important;
    }

    /* Hide the radio mark: any div in the label that doesn't contain the text */
    [data-testid="stSidebar"] div[role="radiogroup"] label div:not(:has(p)) {
        display: none !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100% !important;
        margin: 0 !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        border-left: 3px solid transparent !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #5E5452 !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #F5F0E8 !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #F3E6E3 !important;
        border-left-color: #6D2438 !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #6D2438 !important;
        font-weight: 600 !important;
    }

    .sidebar-note {
        margin-top: 22px;
        padding: 12px 14px;
        background: #F5F0E8;
        border: 1px solid #DED5CC;
        border-radius: 8px;
        font-size: 0.74rem;
        color: #746A67;
        line-height: 1.5;
    }

    /* Cards */
    .editorial-card {
        background-color: #FFFDF8;
        border: 1px solid #DED5CC;
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }

    .editorial-card-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.22rem;
        font-weight: 600;
        color: #302A2B;
        margin: 0 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #EFE8DE;
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 22px;
    }

    @media (max-width: 800px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    .kpi-card {
        background-color: #FFFDF8;
        border: 1px solid #DED5CC;
        border-top: 3px solid #6D2438;
        border-radius: 6px;
        padding: 14px 18px;
    }

    .kpi-label {
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #746A67;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.85rem;
        font-weight: 600;
        color: #302A2B;
        line-height: 1.1;
        margin-bottom: 2px;
    }

    .kpi-subtext {
        font-size: 0.76rem;
        color: #746A67;
    }

    /* Valuation hero */
    .valuation-hero {
        background-color: #FFFDF8;
        border: 1px solid #DED5CC;
        border-left: 5px solid #6D2438;
        border-radius: 8px;
        padding: 24px 28px;
        margin-top: 18px;
        margin-bottom: 22px;
    }

    .valuation-figure {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.85rem;
        font-weight: 600;
        color: #6D2438;
        line-height: 1.05;
        margin: 4px 0 8px 0;
    }

    /* Streamlit Primary Button */
    button[kind="primary"], div.stButton > button {
        background-color: #6D2438 !important;
        color: #FFFDF8 !important;
        border: 1px solid #501827 !important;
        border-radius: 6px !important;
        padding: 0.55rem 1.6rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.15s ease-in-out !important;
        width: 100% !important;
    }

    button[kind="primary"]:hover, div.stButton > button:hover {
        background-color: #501827 !important;
        border-color: #3D101C !important;
        color: #FFFDF8 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.86rem !important;
        color: #746A67 !important;
        background-color: transparent !important;
        border: none !important;
        padding: 8px 16px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #6D2438 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #6D2438 !important;
    }

    /* Form input labels */
    label[data-testid="stWidgetLabel"] p {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #302A2B !important;
        letter-spacing: 0.01em !important;
    }

    /* Inputs, Selectboxes */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #FFFDF8 !important;
        border: 1px solid #DED5CC !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #6D2438 !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #6D2438 !important;
    }

    /* Disclaimer callout */
    .editorial-callout {
        background-color: #FAF5EE;
        border: 1px solid #DED5CC;
        border-left: 3px solid #746A67;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #746A67;
        line-height: 1.45;
        margin-bottom: 18px;
    }

    /* Error callout */
    .editorial-error {
        background-color: #FAF5EE;
        border: 1px solid #DED5CC;
        border-left: 3px solid #6D2438;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #6D2438;
        line-height: 1.45;
        margin-bottom: 14px;
        font-weight: 500;
    }

    /* Report Table */
    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
        background-color: #FFFDF8;
        border: 1px solid #DED5CC;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    .benchmark-table th {
        background-color: #EFE8DE;
        color: #302A2B;
        font-weight: 600;
        text-align: left;
        padding: 10px 14px;
        border-bottom: 1px solid #DED5CC;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .benchmark-table td {
        padding: 11px 14px;
        border-bottom: 1px solid #EFE8DE;
        color: #302A2B;
    }

    .benchmark-table tr.winner-row {
        background-color: #FAF2EE;
        font-weight: 600;
        border-left: 3px solid #6D2438;
    }

    .benchmark-table td.winner-cell {
        color: #6D2438;
    }

    /* ---------- Site Footer (distinct block) ---------- */
    .terminal-footer {
        margin: 56px -2.5rem 0 -2.5rem;
        background-color: #2A2223;
        border-top: 3px solid #6D2438;
        padding: 2.2rem 2.5rem 1.2rem 2.5rem;
        color: #CFC3BA;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .footer-grid {
        display: grid;
        grid-template-columns: 1.6fr 1fr 1fr;
        gap: 28px;
    }

    @media (max-width: 800px) {
        .footer-grid {
            grid-template-columns: 1fr;
        }
    }

    .footer-brand {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.25rem;
        font-weight: 500;
        color: #FFFDF8;
        margin-bottom: 6px;
    }

    .footer-heading {
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #D9C7B9;
        margin-bottom: 8px;
    }

    .terminal-footer a {
        color: #E3CFC0 !important;
        text-decoration: none;
        border-bottom: 1px solid rgba(227, 207, 192, 0.35);
    }

    .terminal-footer a:hover {
        color: #FFFDF8 !important;
        border-bottom-color: #FFFDF8;
    }

    .footer-bottom {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 8px;
        margin-top: 22px;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 253, 248, 0.10);
        font-size: 0.74rem;
        color: #9C8F88;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_clean_data():
    """Load cleaned dataset for descriptive summaries."""
    if not config.CLEAN_DATA_PATH.exists():
        return None
    return pd.read_csv(config.CLEAN_DATA_PATH, parse_dates=["sale_date"])


@st.cache_data
def load_metrics_report():
    """Load machine-readable model metrics report."""
    if not config.METRICS_PATH.exists():
        return None
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@st.cache_resource
def get_trained_model():
    """Load trained prediction pipeline artifact."""
    if not config.SAVED_MODEL_PATH.exists():
        return None
    try:
        return predict.load_model(config.SAVED_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model artifact: {e}")
        return None


# Top Brand Header
st.markdown(
    """
    <div class="brand-banner">
        <div class="brand-eyebrow">MONARCH REAL ESTATE RESEARCH &middot; KING COUNTY TERMINAL</div>
        <div class="brand-title">Market Intelligence &amp; Price Analysis</div>
        <div class="brand-subtitle">Empirical Housing Valuation, Neighborhood Disparity Analytics, and Machine Learning Benchmarks</div>
        <div class="brand-meta">
            <span class="brand-chip">King County, WA</span>
            <span class="brand-chip">21,613 Transactions &middot; 2014&ndash;2015</span>
            <span class="brand-chip">HistGradientBoosting &middot; R&sup2; 0.8967</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Left Sidebar Navigation Menu
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">Monarch Research</div>
        <div class="sidebar-tagline">King County Real Estate Terminal</div>
        <div class="sidebar-section">Menu</div>
        """,
        unsafe_allow_html=True,
    )
    selected_section = st.radio(
        "Navigation",
        ["Price Estimator", "Market Insights", "Model Performance", "Project Details"],
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="sidebar-note">
            <strong>Data window:</strong> May 2014 &ndash; May 2015<br>
            <strong>Model:</strong> HistGradientBoosting<br>
            <strong>Test MAE:</strong> &plusmn;$63,704
        </div>
        """,
        unsafe_allow_html=True,
    )

df = load_clean_data()
metrics_report = load_metrics_report()
model = get_trained_model()


# ==============================================================================
# SECTION 1: PRICE ESTIMATOR
# ==============================================================================
if selected_section == "Price Estimator":
    st.markdown('<div class="brand-eyebrow">PROPERTY VALUATION</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style="font-family: 'Newsreader', Georgia, serif; font-size: 1.85rem; font-weight: 500; margin-top: 0; margin-bottom: 6px;">
            Residential Price Estimator
        </h2>
        <p style="color: #746A67; font-size: 0.92rem; margin-bottom: 14px;">
            Input structural specifications, physical quality, and location coordinates to compute an estimated market valuation.
        </p>
        <div class="editorial-callout">
            <strong>Historical Market Scope:</strong> Predictions are generated by the production <code>HistGradientBoosting</code> pipeline fitted on King County, Washington sales between May 2014 and May 2015. This interface serves as an analytical demonstration and does not constitute a real-time appraisal or non-US valuation tool.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.markdown(
            f"""
            <div class="editorial-error">
                Model artifact not located at <code>{config.SAVED_MODEL_PATH}</code>. Please execute <code>python -m src.train</code> to generate and serialize the production pipeline.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Precompute ZIP code coordinate centroids
        zip_coords = (
            df.groupby("zipcode")[["lat", "long"]].mean().to_dict(orient="index")
            if df is not None
            else {}
        )
        zip_options = sorted(list(zip_coords.keys())) if zip_coords else [98103]

        with st.form("valuation_form"):
            col_form1, col_form2 = st.columns(2)

            with col_form1:
                st.markdown('<div class="editorial-card-title">Property Dimensions</div>', unsafe_allow_html=True)
                sqft_living = st.number_input(
                    "Interior Living Area (sq. ft.)",
                    min_value=300,
                    max_value=12000,
                    value=2000,
                    step=50,
                    help="Finished above-ground and finished basement living area.",
                )
                sqft_lot = st.number_input(
                    "Lot Parcel Area (sq. ft.)",
                    min_value=500,
                    max_value=500000,
                    value=7500,
                    step=250,
                    help="Total land lot area in square feet.",
                )
                sqft_basement = st.number_input(
                    "Basement Area (sq. ft.)",
                    min_value=0,
                    max_value=5000,
                    value=0,
                    step=50,
                    help="Basement square footage (0 if no basement).",
                )
                sqft_living15 = st.number_input(
                    "Nearest 15 Neighbors Avg Living Area (sq. ft.)",
                    min_value=500,
                    max_value=8000,
                    value=1900,
                    step=50,
                    help="Average living space of the 15 nearest neighboring properties.",
                )
                sqft_lot15 = st.number_input(
                    "Nearest 15 Neighbors Avg Lot Area (sq. ft.)",
                    min_value=500,
                    max_value=200000,
                    value=7500,
                    step=250,
                    help="Average lot size of the 15 nearest neighboring properties.",
                )

                st.markdown('<div class="editorial-card-title" style="margin-top: 18px;">Location and Timing</div>', unsafe_allow_html=True)
                selected_zip = st.selectbox(
                    "ZIP Code",
                    options=zip_options,
                    index=zip_options.index(98103) if 98103 in zip_options else 0,
                    help="King County postal code.",
                )
                default_lat = zip_coords[selected_zip]["lat"] if selected_zip in zip_coords else 47.57
                default_long = zip_coords[selected_zip]["long"] if selected_zip in zip_coords else -122.23

                col_geo1, col_geo2 = st.columns(2)
                with col_geo1:
                    lat = st.number_input("Latitude", value=float(default_lat), format="%.4f")
                with col_geo2:
                    long = st.number_input("Longitude", value=float(default_long), format="%.4f")

                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    sale_month = st.selectbox("Sale Month", options=list(range(1, 13)), index=5)
                with col_time2:
                    sale_year = st.selectbox("Sale Year", options=[2014, 2015], index=0)

            with col_form2:
                st.markdown('<div class="editorial-card-title">Property Characteristics</div>', unsafe_allow_html=True)
                col_bed_bath1, col_bed_bath2 = st.columns(2)
                with col_bed_bath1:
                    bedrooms = st.number_input("Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
                with col_bed_bath2:
                    bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=8.0, value=2.25, step=0.25)

                floors = st.number_input("Floors", min_value=1.0, max_value=3.5, value=1.5, step=0.5)

                grade = st.slider(
                    "Construction Grade (1–13)",
                    min_value=1,
                    max_value=13,
                    value=7,
                    help="King County construction quality rating: 1-3 substandard, 7 average, 10-13 luxury.",
                )
                condition = st.slider(
                    "Condition Rating (1–5)",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="Physical maintenance state: 1 poor, 3 average, 5 pristine.",
                )

                col_view1, col_view2 = st.columns(2)
                with col_view1:
                    waterfront = st.selectbox(
                        "Waterfront Status",
                        options=[0, 1],
                        format_func=lambda x: "Waterfront property" if x == 1 else "Non-waterfront",
                    )
                with col_view2:
                    view = st.slider("View Quality (0–4)", min_value=0, max_value=4, value=0)

                col_age1, col_age2 = st.columns(2)
                with col_age1:
                    house_age = st.number_input("House Age at Sale (Years)", min_value=0, max_value=120, value=35, step=1)
                with col_age2:
                    is_renovated = st.selectbox(
                        "Renovation History",
                        options=[0, 1],
                        format_func=lambda x: "Recorded renovation" if x == 1 else "Never renovated",
                    )

                years_since_update = st.number_input(
                    "Years Since Last Structural Update",
                    min_value=0,
                    max_value=120,
                    value=15 if is_renovated else house_age,
                    step=1,
                    help="Years since construction or last recorded renovation (cannot exceed house age).",
                )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            submit_valuation = st.form_submit_button("Generate Property Valuation", type="primary")

        if submit_valuation:
            has_basement = 1 if sqft_basement > 0 else 0

            property_spec = {
                "sqft_living": sqft_living,
                "sqft_basement": sqft_basement,
                "sqft_lot": sqft_lot,
                "sqft_living15": sqft_living15,
                "sqft_lot15": sqft_lot15,
                "bathrooms": bathrooms,
                "bedrooms": bedrooms,
                "floors": floors,
                "waterfront": waterfront,
                "view": view,
                "condition": condition,
                "grade": grade,
                "house_age": house_age,
                "years_since_update": years_since_update,
                "has_basement": has_basement,
                "is_renovated": is_renovated,
                "lat": lat,
                "long": long,
                "sale_year": sale_year,
                "sale_month": sale_month,
                "zipcode": selected_zip,
            }

            is_valid, validation_errors = app_utils.validate_property_input(property_spec)

            if not is_valid:
                for err in validation_errors:
                    st.markdown(f'<div class="editorial-error">{err}</div>', unsafe_allow_html=True)
            else:
                try:
                    estimated_val = predict.predict_price(model, property_spec)[0]
                    test_mae = (
                        metrics_report["model_performance"]["HistGradientBoosting"]["mae_dollar"]
                        if metrics_report
                        else 63704
                    )
                    implied_rate = estimated_val / sqft_living

                    st.markdown(
                        f"""
                        <div class="valuation-hero">
                            <div class="brand-eyebrow" style="color: #6D2438;">ESTIMATED MARKET VALUATION</div>
                            <div class="valuation-figure">${estimated_val:,.0f}</div>
                            <div style="font-size: 0.88rem; color: #746A67; margin-bottom: 12px;">
                                Typical held-out test absolute error reference: <strong>&plusmn;${test_mae:,.0f}</strong> (Test MAE: $63,704).
                            </div>
                            <div style="font-size: 0.78rem; color: #746A67; line-height: 1.45; border-top: 1px solid #DED5CC; padding-top: 10px;">
                                <strong>Analytical Note:</strong> The error reference represents the average empirical absolute error of the HistGradientBoosting pipeline on the independent test sample (n = 4,324). This figure is not a guaranteed confidence interval or contractual price. Implied unit valuation: <strong>${implied_rate:,.2f} per sq. ft.</strong> &middot; ZIP Code: <strong>{selected_zip}</strong> &middot; Grade: <strong>{grade}/13</strong>.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as ex:
                    st.markdown(f'<div class="editorial-error">Valuation failed: {ex}</div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: MARKET INSIGHTS
# ==============================================================================
elif selected_section == "Market Insights":
    st.markdown('<div class="brand-eyebrow">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style="font-family: 'Newsreader', Georgia, serif; font-size: 1.85rem; font-weight: 500; margin-top: 0; margin-bottom: 6px;">
            King County Housing Market Insights
        </h2>
        <p style="color: #746A67; font-size: 0.92rem; margin-bottom: 18px;">
            Descriptive statistical analysis of 21,613 verified residential transactions recorded between May 2014 and May 2015.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if df is not None:
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Transactions</div>
                    <div class="kpi-value">{len(df):,}</div>
                    <div class="kpi-subtext">Verified residential sales</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Median Sale Price</div>
                    <div class="kpi-value">${df['price'].median():,.0f}</div>
                    <div class="kpi-subtext">Robust central tendency</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Mean Sale Price</div>
                    <div class="kpi-value">${df['price'].mean():,.0f}</div>
                    <div class="kpi-subtext">Elevated by luxury right-tail</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Geographic Coverage</div>
                    <div class="kpi-value">{df['zipcode'].nunique()}</div>
                    <div class="kpi-subtext">Distinct King County ZIPs</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tabs = st.tabs(
        [
            "Price Distribution",
            "Quality and Condition",
            "Geographic Variation",
            "Monthly Trends",
            "Correlation Analysis",
        ]
    )

    with tabs[0]:
        st.markdown('<div class="editorial-card-title">Sale Price Skewness and Log Normalization</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color: #746A67; font-size: 0.88rem; line-height: 1.5;">
                Raw transaction prices demonstrate pronounced positive skewness (<strong>4.02</strong>), with 50% of homes trading between $321,950 and $645,000, alongside an extended right-tail reaching $7,700,000. Applying a log-transformation yields near-symmetry (<strong>skewness: 0.43</strong>), providing empirical justification for log-target regression modeling.
            </p>
            """,
            unsafe_allow_html=True,
        )
        fig1 = config.FIGURES_DIR / "01_price_distribution.png"
        if fig1.exists():
            st.image(str(fig1), use_container_width=True)

    with tabs[1]:
        st.markdown('<div class="editorial-card-title">Construction Grade vs. Physical Condition</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color: #746A67; font-size: 0.88rem; line-height: 1.5;">
                Construction grade (1–13 scale rating design and materials) acts as the primary structural price driver, with median valuation progressing monotonically from $375,000 at Grade 7 (n = 8,981) to $914,327 at Grade 10 (n = 1,134). Conversely, maintenance condition (1–5 scale) exhibits a non-monotonic association with price (Condition 3 median: $450,000; Condition 4: $440,000), reflecting confounding interactions with property age.
            </p>
            """,
            unsafe_allow_html=True,
        )
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            fig2 = config.FIGURES_DIR / "02_price_by_grade.png"
            if fig2.exists():
                st.image(str(fig2), use_container_width=True)
        with c_col2:
            fig3 = config.FIGURES_DIR / "03_price_by_condition.png"
            if fig3.exists():
                st.image(str(fig3), use_container_width=True)

    with tabs[2]:
        st.markdown('<div class="editorial-card-title">Neighborhood Pricing Tiers and Micro-Market Disparities</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color: #746A67; font-size: 0.88rem; line-height: 1.5;">
                Location represents the strongest non-structural price determinant. Median home prices span an eightfold range across King County postal codes, ranging from $235,000 in South King County (98002, 98168) to $1,892,500 in Medina (98039, n = 50). Unit prices per square foot exhibit a fourfold spread from ~$145 to ~$565/sq. ft. across neighborhoods.
            </p>
            """,
            unsafe_allow_html=True,
        )
        z_col1, z_col2 = st.columns(2)
        with z_col1:
            fig4 = config.FIGURES_DIR / "04_price_top10_zipcodes.png"
            if fig4.exists():
                st.image(str(fig4), use_container_width=True)
        with z_col2:
            fig5 = config.FIGURES_DIR / "05_price_per_sqft_by_zipcode.png"
            if fig5.exists():
                st.image(str(fig5), use_container_width=True)

    with tabs[3]:
        st.markdown('<div class="editorial-card-title">Transaction Volumes and Monthly Pricing Stability</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color: #746A67; font-size: 0.88rem; line-height: 1.5;">
                Monthly transaction volume exhibits strong seasonal peaks during late spring and summer (>2,100 sales per month in June, July, and April), before dipping to 978 sales in January. Concurrently, median sales prices stayed within a stable $425,545–$476,500 band throughout the 13-month historical window.
            </p>
            """,
            unsafe_allow_html=True,
        )
        fig9 = config.FIGURES_DIR / "09_monthly_sales_and_median_price.png"
        if fig9.exists():
            st.image(str(fig9), use_container_width=True)

    with tabs[4]:
        st.markdown('<div class="editorial-card-title">Rank Correlation Hierarchy</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color: #746A67; font-size: 0.88rem; line-height: 1.5;">
                Spearman rank correlations quantify monotonic associations without linear parametric assumptions: construction grade (0.66), living area (0.64), neighbor living area (0.57), and bathrooms (0.50) correlate highest with price. House age demonstrates a near-zero linear correlation (-0.10) attributable to a pronounced U-shaped pattern where both brand-new and pre-war historic homes command premiums.
            </p>
            """,
            unsafe_allow_html=True,
        )
        fig10 = config.FIGURES_DIR / "10_correlation_heatmap_spearman.png"
        if fig10.exists():
            st.image(str(fig10), use_container_width=True)


# ==============================================================================
# SECTION 3: MODEL PERFORMANCE
# ==============================================================================
elif selected_section == "Model Performance":
    st.markdown('<div class="brand-eyebrow">BENCHMARK EVALUATION REPORT</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style="font-family: 'Newsreader', Georgia, serif; font-size: 1.85rem; font-weight: 500; margin-top: 0; margin-bottom: 6px;">
            Model Performance &amp; Evaluation Benchmarks
        </h2>
        <p style="color: #746A67; font-size: 0.92rem; margin-bottom: 16px;">
            Candidate algorithms evaluated on an identical property-aware test partition (n = 4,324, zero overlapping property IDs).
        </p>
        <div class="editorial-callout">
            <strong>Selected Production Pipeline:</strong> <code>HistGradientBoostingRegressor</code> achieved the highest generalization score across all evaluation criteria (lowest MAE: <strong>$63,704</strong>, lowest RMSE: <strong>$120,140</strong>, highest R&sup2;: <strong>0.8967</strong>) while maintaining a lightweight on-disk footprint (~560 KB).
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics_report is not None:
        perf = metrics_report.get("model_performance", {})

        # Render refined financial-terminal style table
        table_rows = []
        for m_name, stats in perf.items():
            is_winner = (m_name == "HistGradientBoosting")
            row_class = ' class="winner-row"' if is_winner else ""
            prefix = "<strong>" if is_winner else ""
            suffix = " (Selected)</strong>" if is_winner else ""

            row_str = (
                f'<tr{row_class}>'
                f'<td>{prefix}{m_name}{suffix}</td>'
                f'<td style="text-align: right;">${stats["mae_dollar"]:,.0f}</td>'
                f'<td style="text-align: right;">${stats["rmse_dollar"]:,.0f}</td>'
                f'<td style="text-align: right;">{stats["r2"]:.4f}</td>'
                f'<td style="text-align: right;">{stats["mae_log"]:.4f}</td>'
                f'<td style="text-align: right;">{stats["rmse_log"]:.4f}</td>'
                f'<td style="text-align: right;">{stats["fit_time_seconds"]:.2f}s</td>'
                f'</tr>'
            )
            table_rows.append(row_str)

        rows_joined = "".join(table_rows)
        table_html = (
            f'<table class="benchmark-table">'
            f'<thead><tr>'
            f'<th>Model Architecture</th>'
            f'<th style="text-align: right;">Test MAE ($)</th>'
            f'<th style="text-align: right;">Test RMSE ($)</th>'
            f'<th style="text-align: right;">Test R&sup2; Score</th>'
            f'<th style="text-align: right;">Test MAE (Log)</th>'
            f'<th style="text-align: right;">Test RMSE (Log)</th>'
            f'<th style="text-align: right;">Fit Time</th>'
            f'</tr></thead>'
            f'<tbody>{rows_joined}</tbody>'
            f'</table>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

        # Metric Definitions & Interpretation
        st.markdown(
            """
            <div class="editorial-card" style="padding: 16px 20px; margin-bottom: 24px;">
                <div class="editorial-card-title" style="font-size: 1.05rem; margin-bottom: 8px;">Evaluation Metric Definitions</div>
                <div style="font-size: 0.82rem; color: #746A67; line-height: 1.5;">
                    &bull; <strong>Mean Absolute Error (MAE):</strong> Average dollar error magnitude (|Actual &minus; Predicted|). Less sensitive to extreme outlier home values.<br>
                    &bull; <strong>Root Mean Squared Error (RMSE):</strong> Quadratic metric penalizing large pricing mistakes heavily. Expressed in real USD ($).<br>
                    &bull; <strong>R&sup2; Score (Coefficient of Determination):</strong> Proportion of variance in real property prices explained by the model on unseen test data (1.0 = perfect explanation; baseline median achieves -0.06).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown('<div class="editorial-card-title">Algorithm Comparison</div>', unsafe_allow_html=True)
            fig11 = config.FIGURES_DIR / "11_model_performance_comparison.png"
            if fig11.exists():
                st.image(str(fig11), use_container_width=True)
        with b_col2:
            st.markdown('<div class="editorial-card-title">Residual Diagnostics</div>', unsafe_allow_html=True)
            fig12 = config.FIGURES_DIR / "12_model_residuals_diagnostic.png"
            if fig12.exists():
                st.image(str(fig12), use_container_width=True)


# ==============================================================================
# SECTION 4: PROJECT DETAILS
# ==============================================================================
elif selected_section == "Project Details":
    st.markdown('<div class="brand-eyebrow">GOVERNANCE &amp; METHODOLOGY</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style="font-family: 'Newsreader', Georgia, serif; font-size: 1.85rem; font-weight: 500; margin-top: 0; margin-bottom: 6px;">
            Project Architecture &amp; Methodology
        </h2>
        <p style="color: #746A67; font-size: 0.92rem; margin-bottom: 18px;">
            Technical implementation standards, data leakage safeguards, and experimental governance.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col_det1, col_det2 = st.columns(2)

    with col_det1:
        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-card-title">1. Dataset &amp; Independent Attribution</div>
                <div style="font-size: 0.84rem; color: #302A2B; line-height: 1.55;">
                    The dataset comprises 21,613 property sales across King County, Washington (May 2014 &ndash; May 2015). The raw file originates from the open-source repository <a href="https://github.com/akashrajhub/Residential-Real-Estate-Price-Prediction" target="_blank" style="color: #6D2438;">akashrajhub/Residential-Real-Estate-Price-Prediction</a>, mapping directly to the public King County House Sales dataset on Kaggle. All data cleaning, leakage prevention, modeling pipelines, unit tests, and user interfaces were written independently.
                </div>
            </div>

            <div class="editorial-card">
                <div class="editorial-card-title">2. Data Cleaning &amp; Feature Engineering</div>
                <div style="font-size: 0.84rem; color: #302A2B; line-height: 1.55;">
                    &bull; Resolved a single outlier property recording 33 bedrooms in 1,620 sq. ft. by setting bedrooms to <code>NaN</code> (imputed in pipeline) rather than dropping the record.<br>
                    &bull; Preserved 176 repeat property sales occurring on different dates as valid market observations.<br>
                    &bull; Dropped <code>sqft_above</code> due to perfect multicollinearity (<code>sqft_living &minus; sqft_basement</code>).<br>
                    &bull; Engineered pre-sale indicators: <code>house_age</code>, <code>is_renovated</code>, <code>years_since_update</code>, and calendar indicators.
                </div>
            </div>

            <div class="editorial-card">
                <div class="editorial-card-title">3. Technology Stack</div>
                <div style="font-size: 0.84rem; color: #302A2B; line-height: 1.55;">
                    &bull; <strong>Core:</strong> Python 3.10+, Pandas, NumPy<br>
                    &bull; <strong>Machine Learning:</strong> scikit-learn (HistGradientBoosting, RandomForest, Ridge, Pipelines, ColumnTransformer), Joblib<br>
                    &bull; <strong>Visualizations:</strong> Matplotlib, Seaborn<br>
                    &bull; <strong>Web Application:</strong> Streamlit<br>
                    &bull; <strong>Quality Assurance:</strong> pytest (31 automated unit and integration tests), SQLite
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_det2:
        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-card-title">4. Zero-Leakage Grouped Partitioning</div>
                <div style="font-size: 0.84rem; color: #302A2B; line-height: 1.55;">
                    To prevent data leakage from repeat transactions of the same physical residence, dataset partitioning was conducted using <code>GroupShuffleSplit</code> grouped on property <code>id</code>:<br>
                    &bull; <strong>Training Set:</strong> 17,289 rows (17,148 unique properties)<br>
                    &bull; <strong>Held-out Test Set:</strong> 4,324 rows (4,288 unique properties)<br>
                    &bull; <strong>Property ID Overlap:</strong> Exactly 0 shared IDs.<br>
                    All imputation medians, scaling parameters, and one-hot encoders are fitted exclusively on the training partition.
                </div>
            </div>

            <div class="editorial-card">
                <div class="editorial-card-title">5. Production Pipeline Architecture</div>
                <div style="font-size: 0.84rem; color: #302A2B; line-height: 1.55;">
                    Models are trained using a <code>TransformedTargetRegressor</code> with <code>log1p</code> transformation to normalize the right-skewed target distribution. Predictions are converted back to real dollars using <code>expm1</code>. The selected <code>HistGradientBoosting</code> regressor handles non-linear geospatial interactions across latitude, longitude, and ZIP codes while packaging into a compact ~560 KB artifact.
                </div>
            </div>

            <div class="editorial-card">
                <div class="editorial-card-title">6. Scope &amp; Limitations</div>
                <div style="font-size: 0.84rem; color: #302A2B; line-height: 1.55;">
                    &bull; <strong>Temporal Limits:</strong> Reflects May 2014 &ndash; May 2015 price levels. Does not reflect inflation or recent macroeconomic rate changes.<br>
                    &bull; <strong>Geographic Bounds:</strong> Applicable only to King County, Washington.<br>
                    &bull; <strong>Model Uncertainty:</strong> Predictions represent statistical expected values with an empirical held-out error reference (&plusmn;$63.7K MAE).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Terminal Footer
st.markdown(
    """
    <div class="terminal-footer">
        <div class="footer-grid">
            <div>
                <div class="footer-brand">Monarch Real Estate Research</div>
                <div>Empirical valuation and market analytics for King County, Washington residential property. For analytical demonstration only &mdash; not a formal appraisal.</div>
            </div>
            <div>
                <div class="footer-heading">Data</div>
                <div>King County House Sales<br>May 2014 &ndash; May 2015<br>21,613 verified transactions</div>
            </div>
            <div>
                <div class="footer-heading">Stack</div>
                <div>scikit-learn &middot; Pandas<br>Matplotlib &middot; Seaborn<br>Streamlit</div>
            </div>
        </div>
        <div class="footer-bottom">
            <span>&copy; Monarch Real Estate Research Terminal</span>
            <span>Source: <a href="https://github.com/akashrajhub/Residential-Real-Estate-Price-Prediction" target="_blank">akashrajhub / Residential-Real-Estate-Price-Prediction</a></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
