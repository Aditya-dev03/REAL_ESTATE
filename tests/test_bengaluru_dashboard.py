"""Unit and integration tests for Bengaluru Dashboard data processing and Plotly charts.

Tests cover:
- Clean dataset loading and typing
- Market KPI computations (including empty dataset handling)
- Multi-dimensional filter behavior and edge cases
- Interactive Plotly chart builders (valid traces, empty data resilience)
- Held-out test split prediction retrieval
- End-to-end valuation prediction from dashboard input payloads
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import bengaluru_dashboard_data as bdd
from src import bengaluru_predict, config


@pytest.fixture
def clean_data():
    """Load cached cleaned Bengaluru data."""
    return bdd.load_bengaluru_clean_data()


def test_load_bengaluru_clean_data(clean_data):
    """Verify cleaned data loads with expected columns and non-empty rows."""
    assert isinstance(clean_data, pd.DataFrame)
    assert len(clean_data) > 10000
    assert "price" in clean_data.columns
    assert "total_sqft_clean" in clean_data.columns
    assert "bhk" in clean_data.columns
    assert "location" in clean_data.columns
    assert "area_type" in clean_data.columns
    assert "price_per_sqft" in clean_data.columns


def test_compute_market_kpis(clean_data):
    """Verify market KPIs are positive and within expected domain ranges."""
    kpis = bdd.compute_market_kpis(clean_data)

    assert kpis["total_listings"] == len(clean_data)
    assert 40.0 < kpis["median_price_lakhs"] < 120.0
    assert 60.0 < kpis["mean_price_lakhs"] < 200.0
    assert 3000.0 < kpis["median_price_per_sqft"] < 10000.0
    assert 800.0 < kpis["median_sqft"] < 2000.0
    assert 1.0 <= kpis["median_bhk"] <= 4.0


def test_compute_market_kpis_empty_df():
    """Verify compute_market_kpis handles empty data gracefully without crashing."""
    empty_df = pd.DataFrame(columns=["price", "total_sqft_clean", "bhk", "price_per_sqft"])
    kpis = bdd.compute_market_kpis(empty_df)

    assert kpis["total_listings"] == 0
    assert kpis["median_price_lakhs"] == 0.0
    assert kpis["median_price_per_sqft"] == 0.0


def test_filter_bengaluru_data_bhk_and_price(clean_data):
    """Verify BHK and price range filters return subsets within requested bounds."""
    filtered = bdd.filter_bengaluru_data(
        clean_data,
        bhk_range=(2.0, 3.0),
        price_range_lakhs=(50.0, 100.0),
    )

    assert len(filtered) > 0
    assert len(filtered) < len(clean_data)
    assert filtered["bhk"].min() >= 2.0
    assert filtered["bhk"].max() <= 3.0
    assert filtered["price"].min() >= 50.0
    assert filtered["price"].max() <= 100.0


def test_filter_bengaluru_data_locality(clean_data):
    """Verify locality filter restricts data to specified micro-markets."""
    target_locs = ["Whitefield", "Sarjapur  Road"]
    filtered = bdd.filter_bengaluru_data(
        clean_data,
        localities=target_locs,
    )

    assert len(filtered) > 0
    assert set(filtered["location"].unique()).issubset(set(target_locs))


def test_filter_bengaluru_data_area_type_and_possession(clean_data):
    """Verify area type and possession status filtering."""
    filtered = bdd.filter_bengaluru_data(
        clean_data,
        area_types=["Super built-up  Area"],
        ready_to_move="Ready to Move",
    )

    assert len(filtered) > 0
    assert (filtered["area_type"] == "Super built-up  Area").all()
    assert (filtered["is_ready_to_move"] == 1).all()


def test_filter_bengaluru_data_empty_result(clean_data):
    """Verify filtering on impossible bounds returns empty dataframe without crashing."""
    filtered = bdd.filter_bengaluru_data(
        clean_data,
        price_range_lakhs=(99999.0, 999999.0),
    )
    assert len(filtered) == 0


def test_plotly_price_distribution_chart(clean_data):
    """Verify price distribution chart returns valid Plotly Figure."""
    fig = bdd.create_price_distribution_chart(clean_data, n_bins=30, log_scale=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.data[0].type == "histogram"

    # Test empty data resilience
    empty_fig = bdd.create_price_distribution_chart(pd.DataFrame())
    assert isinstance(empty_fig, go.Figure)


def test_plotly_size_vs_price_chart(clean_data):
    """Verify size vs price scatter plot returns valid Plotly Figure."""
    fig = bdd.create_size_vs_price_chart(clean_data, max_sqft=4000.0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1

    # Test empty data resilience
    empty_fig = bdd.create_size_vs_price_chart(pd.DataFrame())
    assert isinstance(empty_fig, go.Figure)


def test_plotly_bhk_boxplot(clean_data):
    """Verify BHK box plot returns valid Plotly Figure."""
    fig = bdd.create_bhk_boxplot(clean_data, max_bhk=4)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1

    empty_fig = bdd.create_bhk_boxplot(pd.DataFrame())
    assert isinstance(empty_fig, go.Figure)


def test_plotly_locality_bar_chart(clean_data):
    """Verify locality comparison bar chart returns valid Plotly Figure."""
    fig = bdd.create_locality_bar_chart(clean_data, min_listings=10, top_n=10, metric="median")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.data[0].type == "bar"

    empty_fig = bdd.create_locality_bar_chart(pd.DataFrame())
    assert isinstance(empty_fig, go.Figure)


def test_plotly_area_type_chart(clean_data):
    """Verify area type bar chart returns valid Plotly Figure."""
    fig = bdd.create_area_type_chart(clean_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Median and Mean bars

    empty_fig = bdd.create_area_type_chart(pd.DataFrame())
    assert isinstance(empty_fig, go.Figure)


def test_held_out_test_predictions():
    """Verify test predictions match expected holdout sample size."""
    y_test, y_pred, X_test = bdd.get_held_out_test_predictions()
    assert len(y_test) == 2528
    assert len(y_pred) == 2528
    assert len(X_test) == 2528
    assert np.all(np.isfinite(y_pred))


def test_plotly_diagnostic_charts():
    """Verify Predicted vs Actual and Residuals diagnostic charts."""
    y_test, y_pred, _ = bdd.get_held_out_test_predictions()

    fig_pva = bdd.create_predicted_vs_actual_chart(y_test, y_pred)
    assert isinstance(fig_pva, go.Figure)
    assert len(fig_pva.data) == 2  # scatter + line

    fig_res = bdd.create_residuals_chart(y_test, y_pred)
    assert isinstance(fig_res, go.Figure)
    assert len(fig_res.data) >= 1


def test_plotly_model_comparison_chart():
    """Verify multi-model comparison chart using stored metrics JSON."""
    assert config.BENGALURU_METRICS_PATH.exists()
    with open(config.BENGALURU_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    fig = bdd.create_model_comparison_chart(metrics)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # CV MAE, Test MAE, Test MedAE


def test_dashboard_prediction_flow():
    """Verify end-to-end valuation using simulated dashboard form payload."""
    payload = {
        "total_sqft_clean": 1800.0,
        "bhk": 3.0,
        "bath": 3.0,
        "balcony": 2.0,
        "area_type": "Super built-up  Area",
        "location": "Whitefield",
        "is_ready_to_move": 1,
        "is_in_named_society": 1,
    }

    price_lakhs, formatted = bengaluru_predict.predict_bengaluru_price(payload)
    assert isinstance(price_lakhs, float)
    assert price_lakhs > 20.0
    assert "₹" in formatted
