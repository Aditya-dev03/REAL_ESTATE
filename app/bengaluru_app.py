"""Bengaluru Real Estate Market Intelligence & Price Valuation Terminal.

An interactive analytics product and predictive valuation interface
built on the Bengaluru House Price dataset using interactive Plotly charts
and the trained Random Forest (Log) valuation pipeline.

Run from the project root (REAL_ESTATE/):
    streamlit run app/bengaluru_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Configure project root path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import bengaluru_dashboard_data as bdd
from src import bengaluru_predict, config

# Page Configuration
st.set_page_config(
    page_title="Bengaluru Real Estate Intelligence | Market Analytics & Valuation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refined Editorial Design System (Maroon & Warm Off-White)
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

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #EFE9DF !important;
        border-right: 1px solid #DED5CC !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #302A2B !important;
        font-family: 'Newsreader', Georgia, serif !important;
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
        right: -8%;
        width: 420px;
        height: 420px;
        background: radial-gradient(circle, rgba(217, 199, 185, 0.22) 0%, rgba(217, 199, 185, 0) 65%);
        pointer-events: none;
    }
    .brand-eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #E3CFC0 !important;
        margin-bottom: 4px;
    }
    .brand-title {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 2.35rem !important;
        font-weight: 500 !important;
        color: #FFFDF8 !important;
        letter-spacing: -0.015em;
        margin: 0 0 4px 0 !important;
        padding: 0 !important;
        line-height: 1.15 !important;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: rgba(255, 253, 248, 0.78) !important;
        margin: 0 !important;
        line-height: 1.4;
    }
    .brand-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 1rem;
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

    /* ---------- Site Footer (distinct block) ---------- */
    .site-footer {
        margin: 3.5rem -2.5rem 0 -2.5rem;
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
        gap: 1.75rem;
    }
    @media (max-width: 800px) {
        .footer-grid { grid-template-columns: 1fr; }
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
    .footer-bottom {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 8px;
        margin-top: 1.4rem;
        padding-top: 0.9rem;
        border-top: 1px solid rgba(255, 253, 248, 0.10);
        font-size: 0.74rem;
        color: #9C8F88;
        letter-spacing: 0.02em;
    }

    /* Editorial KPI Metric Card */
    .kpi-card {
        background: #FFFDF8;
        border: 1px solid #DED5CC;
        border-radius: 4px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(48, 42, 43, 0.03);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #746A67;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.85rem;
        font-weight: 600;
        color: #6D2438;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #746A67;
    }

    /* Notice and Disclaimer Callout */
    .editorial-notice {
        background: #FFFDF8;
        border-left: 3px solid #6D2438;
        border-top: 1px solid #DED5CC;
        border-right: 1px solid #DED5CC;
        border-bottom: 1px solid #DED5CC;
        padding: 0.85rem 1.1rem;
        margin: 1rem 0;
        font-size: 0.84rem;
        color: #746A67;
        line-height: 1.45;
    }

    /* Valuation Result Card */
    .valuation-card {
        background: #FFFDF8;
        border: 1px solid #DED5CC;
        border-top: 3px solid #6D2438;
        border-radius: 4px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
    }
    .valuation-eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #746A67;
        margin-bottom: 0.4rem;
    }
    .valuation-price {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.4rem;
        font-weight: 600;
        color: #6D2438;
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }
    .valuation-breakdown {
        font-size: 0.9rem;
        color: #302A2B;
        margin-bottom: 0.75rem;
    }

    /* Primary Button Override */
    div.stButton > button:first-child {
        background-color: #6D2438 !important;
        color: #FFFDF8 !important;
        border: 1px solid #501827 !important;
        border-radius: 3px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #501827 !important;
        border-color: #302A2B !important;
        color: #FFFDF8 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #DED5CC;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 0.2rem;
        background-color: transparent !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #746A67 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #6D2438 !important;
        border-bottom: 2px solid #6D2438 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_bengaluru_data() -> pd.DataFrame:
    """Load and cache clean Bengaluru dataset."""
    return bdd.load_bengaluru_clean_data()


@st.cache_data
def get_cached_metrics() -> Dict[str, Any]:
    """Load and cache serialized model benchmark metrics."""
    if not config.BENGALURU_METRICS_PATH.exists():
        return {}
    with open(config.BENGALURU_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_cached_test_predictions():
    """Compute and cache predictions on the held-out test split."""
    return bdd.get_held_out_test_predictions()


def main():
    # Load dataset
    try:
        raw_df = get_cached_bengaluru_data()
    except Exception as e:
        st.error(f"Error loading clean dataset: {e}")
        return

    # Header Banner
    st.markdown(
        """
        <div class="brand-banner">
            <div class="brand-eyebrow">Bengaluru Real Estate Research & Valuation Terminal</div>
            <h1 class="brand-title">Market Intelligence & Predictive Valuation</h1>
            <p class="brand-subtitle">
                Comprehensive analytics, exploratory distributions, and ML-driven asking price estimates
                across 12,640 verified residential listings in Bengaluru, India.
            </p>
            <div class="brand-meta">
                <span class="brand-chip">Bengaluru, India</span>
                <span class="brand-chip">12,640 Listings</span>
                <span class="brand-chip">Random Forest (Log) Valuation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Interactive Shared Filters
    with st.sidebar:
        st.markdown("<h3 style='margin-bottom: 0.2rem;'>Market Filters</h3>", unsafe_allow_html=True)
        st.caption("Filters update the Market Overview, distributions, and comparisons dynamically.")

        # 1. Locality Filter
        locality_counts = raw_df["location"].value_counts()
        frequent_localities = locality_counts[locality_counts >= 10].index.tolist()
        locality_options = ["All Localities"] + sorted(frequent_localities)
        selected_localities = st.multiselect(
            "Locality",
            options=locality_options,
            default=["All Localities"],
            help="Select one or more prominent localities. Localities with ≥10 listings shown.",
        )

        # 2. BHK Range
        min_bhk_val = int(raw_df["bhk"].dropna().min())
        max_bhk_val = int(min(raw_df["bhk"].dropna().max(), 8))
        bhk_slider = st.slider(
            "Bedrooms (BHK)",
            min_value=min_bhk_val,
            max_value=max_bhk_val,
            value=(1, 5),
            step=1,
            help="Filter by number of bedrooms.",
        )

        # 3. Price Range Slider (in ₹ Lakhs)
        p_min = float(raw_df["price"].min())
        p_max = float(min(raw_df["price"].max(), 800.0))
        price_slider = st.slider(
            "Asking Price Range (₹ Lakhs)",
            min_value=float(int(p_min)),
            max_value=float(int(p_max)),
            value=(10.0, 400.0),
            step=5.0,
            format="₹%dL",
            help="Filter by seller asking price in Lakhs of INR (1 Lakh = ₹100,000).",
        )

        # 4. Property Size Range (sq. ft.)
        sqft_min = 400.0
        sqft_max = 6000.0
        sqft_slider = st.slider(
            "Property Size (sq. ft.)",
            min_value=sqft_min,
            max_value=sqft_max,
            value=(500.0, 3500.0),
            step=50.0,
            format="%d sqft",
            help="Filter by total clean built-up or plot area.",
        )

        # 5. Area Type Filter
        area_type_options = ["All Area Types"] + sorted(raw_df["area_type"].dropna().unique().tolist())
        selected_area_types = st.multiselect(
            "Area Classification",
            options=area_type_options,
            default=["All Area Types"],
        )

        # 6. Availability Filter
        avail_choice = st.radio(
            "Possession Status",
            options=["All", "Ready to Move", "Under Construction"],
            index=0,
            horizontal=True,
        )

        # Apply Filters
        filtered_df = bdd.filter_bengaluru_data(
            raw_df,
            localities=selected_localities,
            bhk_range=bhk_slider,
            price_range_lakhs=price_slider,
            sqft_range=sqft_slider,
            area_types=selected_area_types,
            ready_to_move=avail_choice,
        )

        # Filter Stats Badge
        total_n = len(raw_df)
        filt_n = len(filtered_df)
        pct = (filt_n / total_n * 100) if total_n > 0 else 0
        st.markdown(
            f"""
            <div style="background: #FFFDF8; border: 1px solid #DED5CC; padding: 0.65rem 0.85rem; border-radius: 4px; margin-top: 1rem;">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #746A67;">Filtered Listings</div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #6D2438;">{filt_n:,} <span style="font-size: 0.8rem; color: #746A67; font-weight: 400;">of {total_n:,} ({pct:.1f}%)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Reset Instructions
        st.caption("To reset filters, refresh the page or clear selections.")

    # Top-Level Tabs
    tab_overview, tab_exploration, tab_valuation, tab_performance, tab_methodology = st.tabs([
        "Market Overview",
        "Interactive Exploration",
        "Valuation Engine",
        "Model Performance",
        "Methodology & Governance",
    ])

    # ==========================================
    # TAB 1: MARKET OVERVIEW
    # ==========================================
    with tab_overview:
        st.markdown("### Market Summary & Key Indicators")

        # Global vs Filtered KPIs
        kpis = bdd.compute_market_kpis(filtered_df)
        global_kpis = bdd.compute_market_kpis(raw_df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Active Listings</div>
                    <div class="kpi-value">{kpis['total_listings']:,}</div>
                    <div class="kpi-subtext">Dataset Total: {global_kpis['total_listings']:,} properties</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Median Asking Price</div>
                    <div class="kpi-value">₹{kpis['median_price_lakhs']:.1f}L</div>
                    <div class="kpi-subtext">Overall Median: ₹{global_kpis['median_price_lakhs']:.1f}L</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            avg_crores = kpis["mean_price_lakhs"] / 100.0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Average Asking Price</div>
                    <div class="kpi-value">₹{kpis['mean_price_lakhs']:.1f}L</div>
                    <div class="kpi-subtext">₹{avg_crores:.2f} Crore (skewed by luxury villas)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Median Price / Sq. Ft.</div>
                    <div class="kpi-value">₹{kpis['median_price_per_sqft']:,.0f}</div>
                    <div class="kpi-subtext">Overall Median: ₹{global_kpis['median_price_per_sqft']:,.0f}/sq.ft</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Market Context Callout
        st.markdown(
            """
            <div class="editorial-notice">
                <strong>Portal Asking Price Notice:</strong> All metrics and distributions in this terminal
                reflect <em>seller-quoted online listing asking prices</em> collected from real estate web portals,
                <strong>not registered municipal transfer deed transaction values</strong> (sub-registrar guidance values).
                Quoted portal prices typically incorporate an unstandardized 5%–15% negotiation buffer, floor-rise surcharges,
                and parking fees.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if filtered_df.empty:
            st.warning("No listings match your selected filter criteria. Please broaden your filters in the sidebar.")
        else:
            # Chart 1: Price Distribution (full width)
            fig_price = bdd.create_price_distribution_chart(filtered_df, n_bins=35, log_scale=False)
            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown("<hr style='border:0; border-top:1px solid #DED5CC; margin: 0.5rem 0;'>", unsafe_allow_html=True)

            # Chart 2: Locality Bar Chart (full width)
            fig_loc = bdd.create_locality_bar_chart(filtered_df, min_listings=10, top_n=12, metric="median")
            st.plotly_chart(fig_loc, use_container_width=True)

    # ==========================================
    # TAB 2: INTERACTIVE MARKET EXPLORATION
    # ==========================================
    with tab_exploration:
        st.markdown("### Interactive Market Diagnostics")
        st.caption("Examine multi-dimensional pricing dynamics across area classifications, unit sizes, and localities.")

        if filtered_df.empty:
            st.warning("No listings match your current filters. Please adjust the sidebar filters.")
        else:
            # Chart 1: Size vs Price Scatter (full width)
            fig_scatter = bdd.create_size_vs_price_chart(filtered_df, max_sqft=sqft_slider[1])
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("<hr style='border:0; border-top:1px solid #DED5CC; margin: 0.5rem 0;'>", unsafe_allow_html=True)

            # Chart 2: BHK Box Plot (full width)
            fig_box = bdd.create_bhk_boxplot(filtered_df, max_bhk=int(bhk_slider[1]))
            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("<hr style='border:0; border-top:1px solid #DED5CC; margin: 0.5rem 0;'>", unsafe_allow_html=True)

            # Chart 3: Locality Bar Chart with controls (full width)
            loc_metric = st.radio(
                "Locality Ranking Metric",
                options=["Median Asking Price", "Listing Volume"],
                horizontal=True,
                key="loc_rank_metric",
            )
            min_thresh = st.slider("Min. Listings Filter", min_value=5, max_value=50, value=15, step=5)
            fig_loc_deep = bdd.create_locality_bar_chart(
                filtered_df,
                min_listings=min_thresh,
                top_n=15,
                metric="median" if loc_metric == "Median Asking Price" else "volume",
            )
            st.plotly_chart(fig_loc_deep, use_container_width=True)

            st.markdown("<hr style='border:0; border-top:1px solid #DED5CC; margin: 0.5rem 0;'>", unsafe_allow_html=True)

            # Chart 4: Area Type Chart (full width)
            fig_area = bdd.create_area_type_chart(filtered_df)
            st.plotly_chart(fig_area, use_container_width=True)

    # ==========================================
    # TAB 3: PRICE PREDICTION (VALUATION ENGINE)
    # ==========================================
    with tab_valuation:
        st.markdown("### Predictive Property Valuation Engine")
        st.caption(
            "Generates fair-market asking price estimates using the production "
            "<strong>Random Forest (Log)</strong> pipeline trained with zero-leakage safeguards."
        )

        # Form Inputs
        with st.form("bengaluru_valuation_form"):
            st.markdown("##### Property Specifications")
            f_col1, f_col2, f_col3 = st.columns(3)

            with f_col1:
                # Top locations + option to enter custom
                top_locations = sorted(raw_df["location"].value_counts().head(80).index.tolist())
                input_locality = st.selectbox(
                    "Locality / Micro-Market",
                    options=top_locations + ["Other / Unlisted"],
                    index=top_locations.index("Whitefield") if "Whitefield" in top_locations else 0,
                    help="Select locality. Unseen or rare locations map gracefully to infrequent category representations.",
                )
                input_area_type = st.selectbox(
                    "Area Classification",
                    options=["Super built-up  Area", "Built-up  Area", "Plot  Area", "Carpet  Area"],
                    index=0,
                )

            with f_col2:
                input_sqft = st.number_input(
                    "Total Area (sq. ft.)",
                    min_value=250.0,
                    max_value=15000.0,
                    value=1450.0,
                    step=50.0,
                    help="Gross or carpet area in square feet.",
                )
                input_bhk = st.selectbox(
                    "Bedrooms (BHK)",
                    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    index=2,  # Default 3 BHK
                )

            with f_col3:
                input_bath = st.selectbox(
                    "Bathrooms",
                    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    index=1,  # Default 2 Bath
                )
                input_balcony = st.selectbox(
                    "Balcony Count",
                    options=[0.0, 1.0, 2.0, 3.0],
                    index=2,  # Default 2 Balconies
                )

            st.markdown("##### Development & Community Attributes")
            f_col4, f_col5 = st.columns(2)
            with f_col4:
                input_ready = st.radio(
                    "Possession Status",
                    options=["Ready to Move", "Under Construction"],
                    index=0,
                    horizontal=True,
                )
            with f_col5:
                input_society = st.radio(
                    "Gated Society / Branded Complex",
                    options=["Yes, Named Society / Complex", "No, Standalone / Independent Plot"],
                    index=0,
                    horizontal=True,
                )

            submit_val = st.form_submit_button("Estimate Market Asking Price")

        if submit_val:
            # Build input dictionary
            prop_payload = {
                "total_sqft_clean": float(input_sqft),
                "bhk": float(input_bhk),
                "bath": float(input_bath),
                "balcony": float(input_balcony),
                "area_type": input_area_type,
                "location": "Other" if input_locality == "Other / Unlisted" else input_locality,
                "is_ready_to_move": 1 if input_ready == "Ready to Move" else 0,
                "is_in_named_society": 1 if "Yes" in input_society else 0,
            }

            # Run validation
            val_errors = bengaluru_predict.validate_bengaluru_input(prop_payload)
            if val_errors:
                for err in val_errors:
                    st.error(f"Validation Notice: {err}")
            else:
                try:
                    price_lakhs, inr_str = bengaluru_predict.predict_bengaluru_price(prop_payload)
                    price_per_sqft_est = (price_lakhs * 100000.0) / float(input_sqft)

                    # Compute contextual locality benchmarks
                    loc_name = prop_payload["location"]
                    loc_match = raw_df[raw_df["location"] == loc_name]
                    loc_med = loc_match["price"].median() if len(loc_match) >= 5 else None

                    bhk_match = raw_df[raw_df["bhk"] == float(input_bhk)]
                    bhk_med = bhk_match["price"].median() if len(bhk_match) >= 10 else None

                    st.markdown(
                        f"""
                        <div class="valuation-card">
                            <div class="valuation-eyebrow">Estimated Portal Asking Price</div>
                            <div class="valuation-price">{inr_str}</div>
                            <div class="valuation-breakdown">
                                <strong>Estimated Rate:</strong> ₹{price_per_sqft_est:,.0f} / sq. ft. &nbsp;|&nbsp;
                                <strong>Configuration:</strong> {int(input_bhk)} BHK, {int(input_bath)} Bath, {int(input_sqft):,} sq. ft. &nbsp;|&nbsp;
                                <strong>Locality:</strong> {input_locality}
                            </div>
                            <hr style="border: 0; border-top: 1px solid #DED5CC; margin: 0.85rem 0;" />
                            <div style="font-size: 0.82rem; color: #746A67; line-height: 1.45;">
                                <strong>Benchmark Comparisons:</strong><br/>
                                • City-wide median for {int(input_bhk)} BHK units: <strong>₹{bhk_med:.1f} Lakhs</strong><br/>
                                {f'• Locality median for {input_locality}: <strong>₹{loc_med:.1f} Lakhs</strong>' if loc_med else '• Locality median: Limited listing volume in selected micro-market.'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Model Uncertainty and Scope Notice
                    st.markdown(
                        """
                        <div class="editorial-notice">
                            <strong>Valuation Bounds & Statistical Margin:</strong><br/>
                            • <strong>Median Absolute Error (MedAE):</strong> ₹12.95 Lakhs. On the independent held-out test set,
                            50% of model estimates were within ₹12.95 Lakhs of the quoted asking price.<br/>
                            • <strong>Mean Absolute Error (MAE):</strong> ₹32.60 Lakhs (inclusive of heavy-tail luxury dispersion).<br/>
                            • <strong>Advisory Disclaimer:</strong> This valuation is a statistical estimate based on historical listing portal data.
                            It does not constitute a formal banking mortgage appraisal, RERA compliance certification, or legal title verification.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as ex:
                    st.error(f"Inference engine failure: {ex}")

    # ==========================================
    # TAB 4: MODEL PERFORMANCE & DIAGNOSTICS
    # ==========================================
    with tab_performance:
        st.markdown("### Model Benchmarking & Statistical Diagnostics")
        st.caption(
            "Evaluation metrics computed on the locked, held-out test partition (20% split, $n = 2,528$) "
            "with zero data leakage."
        )

        metrics = get_cached_metrics()
        perf = metrics.get("model_performance", {})
        sel = metrics.get("selected_model", {})

        # Display Selected Model Overview
        sel_test = sel.get("test_metrics", {})
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Test MAE (₹ Lakhs)</div>
                    <div class="kpi-value">₹{sel_test.get('test_mae_lakhs', 32.60):.2f}L</div>
                    <div class="kpi-subtext">{sel_test.get('test_mae_inr', '₹3,259,818')} avg. dollar error</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Median Absolute Error</div>
                    <div class="kpi-value">₹{sel_test.get('test_medae_lakhs', 12.95):.2f}L</div>
                    <div class="kpi-subtext">50% of test errors within ₹12.95L</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Test RMSE</div>
                    <div class="kpi-value">₹{sel_test.get('test_rmse_lakhs', 86.05):.2f}L</div>
                    <div class="kpi-subtext">Penalizes luxury tail errors</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Test R² Score</div>
                    <div class="kpi-value">{sel_test.get('test_r2', 0.6390):.4f}</div>
                    <div class="kpi-subtext">Variance explained by attributes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plain language explanation of metrics
        st.markdown(
            """
            <div class="editorial-notice">
                <strong>Plain-Language Interpretations:</strong><br/>
                • <strong>R² (0.6390)</strong> measures the proportion of total asking price variance explained by model features.
                <em>It does not mean "63.9% prediction accuracy"</em>, because variance in real estate pricing is heavily driven by unobserved factors (e.g. view premiums, builder brand, interior woodwork).<br/>
                • <strong>MAE (₹32.60L)</strong> is the sample mean of absolute errors. It is <em>not a guaranteed upper bound</em> on error, as high-end luxury penthouses (>₹4 Crore) exhibit larger absolute dollar spreads.<br/>
                • <strong>MedAE (₹12.95L)</strong> confirms that for typical budget and mid-market apartments, the valuation accuracy is substantially tighter than the overall mean.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Interactive Diagnostic Plots
        try:
            y_test, y_pred, _ = get_cached_test_predictions()

            # Chart 1: Predicted vs Actual (full width)
            fig_pva = bdd.create_predicted_vs_actual_chart(y_test, y_pred)
            st.plotly_chart(fig_pva, use_container_width=True)

            st.markdown("<hr style='border:0; border-top:1px solid #DED5CC; margin: 0.5rem 0;'>", unsafe_allow_html=True)

            # Chart 2: Residuals (full width)
            fig_res = bdd.create_residuals_chart(y_test, y_pred)
            st.plotly_chart(fig_res, use_container_width=True)
        except Exception as ex:
            st.warning(f"Could not render test split diagnostics: {ex}")

        # Multi-Model Benchmark Chart
        if metrics:
            st.markdown("#### Candidate Architecture Benchmark")
            fig_bench = bdd.create_model_comparison_chart(metrics)
            st.plotly_chart(fig_bench, use_container_width=True)

    # ==========================================
    # TAB 5: METHODOLOGY & DATA GOVERNANCE
    # ==========================================
    with tab_methodology:
        st.markdown("### Data Governance, Ethics & Implementation Architecture")

        st.markdown(
            """
            #### 1. Dataset Origin & Attribution
            - **Source**: Public Bengaluru House Price dataset collected by Amitabha Joy on Kaggle.
            - **Scope**: 13,320 initial raw records cleaned down to 12,640 verified residential listings across the Bengaluru metropolitan region.
            - **Nature of Data**: Online real estate portal listings reflecting seller asking prices.

            #### 2. Strict Target Leakage Safeguards
            - **Excluded Features**: `price_per_sqft` is strictly excluded from all model inputs. In real estate data, price per square foot is a direct mathematical derivative of the target ($y / \text{sqft}$); using it as an input creates circular target leakage.
            - **Training-Fold Preprocessing**: All locality frequency aggregations (`OneHotEncoder(min_frequency=10)`) and scalers were fitted exclusively on training folds. Unseen localities or rare micro-markets map safely to an infrequent category representation.

            #### 3. Empirical Heuristics & Exclusions
            - The cleaning pipeline identified 151 records with area-per-BHK ratios below 150 sq. ft. (such as multi-bedroom houses listed with under 300 total sq. ft.).
            - These records represent data entry errors (e.g. plot dimensions recorded in square yards or cents without proper unit flags). The raw data is preserved untouched in `data/raw/bengaluru_house_data.csv`.

            #### 4. Limitations & Real Estate Market Nuance
            1. **No Municipal Title Guarantees**: Does not capture whether a property possesses BBMP 'A' Khata, 'B' Khata, or BDA layout title clearances.
            2. **Untracked Qualitative Features**: Floor rise premiums (e.g. 20th floor vs. ground floor), clubhouse luxury amenities, vastu orientation, and builder tier (e.g. Sobha vs. unbranded) are uncaptured in web portal listings.
            """
        )

    # Site Footer
    st.markdown(
        """
        <div class="site-footer">
            <div class="footer-grid">
                <div>
                    <div class="footer-brand">Bengaluru Real Estate Intelligence</div>
                    <div>Market analytics and ML-driven asking price estimates for Bengaluru residential listings. Statistical estimates only &mdash; not a formal appraisal or RERA certification.</div>
                </div>
                <div>
                    <div class="footer-heading">Data</div>
                    <div>Bengaluru House Price dataset (Kaggle)<br>12,640 cleaned listings<br>Portal asking prices</div>
                </div>
                <div>
                    <div class="footer-heading">Stack</div>
                    <div>scikit-learn &middot; Pandas<br>Plotly<br>Streamlit</div>
                </div>
            </div>
            <div class="footer-bottom">
                <span>&copy; Bengaluru Real Estate Research &amp; Valuation Terminal</span>
                <span>Model: Random Forest (Log) &middot; Test MedAE &plusmn;&#8377;12.95L</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
