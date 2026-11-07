"""Data processing, KPI calculations, and Plotly chart generation for Bengaluru Dashboard.

Separates analytical computations and chart generation from Streamlit UI rendering,
enabling thorough unit and integration testing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import config, bengaluru_predict, bengaluru_train

# Luxury Maroon & Warm Off-White Design Palette
COLOR_OXBLOOD = "#6D2438"
COLOR_OXBLOOD_DARK = "#501827"
COLOR_BG_CARD = "#FFFDF8"
COLOR_TEXT_MAIN = "#302A2B"
COLOR_TEXT_MUTED = "#746A67"
COLOR_BORDER = "#DED5CC"
COLOR_GRID = "#E8DFD8"
COLOR_ACCENT_AMBER = "#8B5A2B"
COLOR_ACCENT_ROSE = "#A3485E"
COLOR_ACCENT_SLATE = "#5C677D"


def load_bengaluru_clean_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load cleaned Bengaluru housing dataset for dashboard analytics."""
    path = data_path or config.BENGALURU_CLEAN_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Clean Bengaluru dataset not found at {path}")
    df = pd.read_csv(path)
    # Ensure numeric columns are properly typed
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["total_sqft_clean"] = pd.to_numeric(df["total_sqft_clean"], errors="coerce")
    df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
    df["balcony"] = pd.to_numeric(df["balcony"], errors="coerce").fillna(1.0)
    df["price_per_sqft"] = pd.to_numeric(df["price_per_sqft"], errors="coerce")
    return df


def compute_market_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary KPI metrics for a given property dataframe."""
    if df.empty:
        return {
            "total_listings": 0,
            "median_price_lakhs": 0.0,
            "mean_price_lakhs": 0.0,
            "median_price_per_sqft": 0.0,
            "median_sqft": 0.0,
            "median_bhk": 0.0,
        }

    return {
        "total_listings": int(len(df)),
        "median_price_lakhs": round(float(df["price"].median()), 2),
        "mean_price_lakhs": round(float(df["price"].mean()), 2),
        "median_price_per_sqft": round(float(df["price_per_sqft"].median()), 0),
        "median_sqft": round(float(df["total_sqft_clean"].median()), 0),
        "median_bhk": round(float(df["bhk"].median()), 1),
    }


def filter_bengaluru_data(
    df: pd.DataFrame,
    localities: Optional[List[str]] = None,
    bhk_range: Optional[Tuple[float, float]] = None,
    price_range_lakhs: Optional[Tuple[float, float]] = None,
    sqft_range: Optional[Tuple[float, float]] = None,
    area_types: Optional[List[str]] = None,
    ready_to_move: Optional[str] = "All",
) -> pd.DataFrame:
    """Filter property dataset based on user-selected criteria."""
    filtered = df.copy()

    # Locality filter
    if localities and "All Localities" not in localities:
        filtered = filtered[filtered["location"].isin(localities)]

    # BHK range filter
    if bhk_range:
        min_bhk, max_bhk = bhk_range
        filtered = filtered[(filtered["bhk"] >= min_bhk) & (filtered["bhk"] <= max_bhk)]

    # Price range filter
    if price_range_lakhs:
        min_p, max_p = price_range_lakhs
        filtered = filtered[(filtered["price"] >= min_p) & (filtered["price"] <= max_p)]

    # Property size range filter
    if sqft_range:
        min_sqft, max_sqft = sqft_range
        filtered = filtered[
            (filtered["total_sqft_clean"] >= min_sqft) & (filtered["total_sqft_clean"] <= max_sqft)
        ]

    # Area type filter
    if area_types and "All Area Types" not in area_types:
        filtered = filtered[filtered["area_type"].isin(area_types)]

    # Ready to move filter
    if ready_to_move == "Ready to Move":
        filtered = filtered[filtered["is_ready_to_move"] == 1]
    elif ready_to_move == "Under Construction":
        filtered = filtered[filtered["is_ready_to_move"] == 0]

    return filtered


def get_base_plotly_layout(title: str, height: int = 420) -> Dict[str, Any]:
    """Return standard minimalist editorial Plotly layout dictionary."""
    return {
        "title": {
            "text": title,
            "font": {"family": "Newsreader, Georgia, serif", "size": 18, "color": COLOR_TEXT_MAIN},
            "x": 0.02,
            "y": 0.95,
        },
        "plot_bgcolor": COLOR_BG_CARD,
        "paper_bgcolor": COLOR_BG_CARD,
        "font": {"family": "Inter, sans-serif", "color": COLOR_TEXT_MAIN, "size": 12},
        "margin": {"l": 55, "r": 35, "t": 60, "b": 50},
        "height": height,
        "hoverlabel": {
            "bgcolor": COLOR_OXBLOOD_DARK,
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#FFFDF8"},
            "bordercolor": COLOR_OXBLOOD,
        },
        "xaxis": {
            "showgrid": True,
            "gridcolor": COLOR_GRID,
            "gridwidth": 1,
            "linecolor": COLOR_BORDER,
            "tickfont": {"size": 11, "color": COLOR_TEXT_MUTED},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": COLOR_GRID,
            "gridwidth": 1,
            "linecolor": COLOR_BORDER,
            "tickfont": {"size": 11, "color": COLOR_TEXT_MUTED},
        },
    }


def create_price_distribution_chart(
    df: pd.DataFrame, n_bins: int = 40, log_scale: bool = False
) -> go.Figure:
    """Create interactive Plotly histogram for asking price distribution."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**get_base_plotly_layout("No listings match selected filters"))
        return fig

    # Clip extreme display outlier for readable visualization if not in log scale
    disp_df = df.copy()
    if not log_scale:
        disp_df = disp_df[disp_df["price"] <= 600]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=disp_df["price"],
            nbinsx=n_bins,
            marker_color=COLOR_OXBLOOD,
            opacity=0.85,
            marker_line={"color": COLOR_BG_CARD, "width": 1},
            hovertemplate="Price Range: ₹%{x:.1f} Lakhs<br>Listings: %{y}<extra></extra>",
            name="Listings",
        )
    )

    median_val = df["price"].median()
    mean_val = df["price"].mean()

    # Median reference line
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color=COLOR_ACCENT_AMBER,
        line_width=2,
        annotation_text=f"Median: ₹{median_val:.1f}L",
        annotation_position="top right",
        annotation_font={"size": 11, "color": COLOR_ACCENT_AMBER},
    )

    # Mean reference line
    fig.add_vline(
        x=mean_val,
        line_dash="dot",
        line_color=COLOR_OXBLOOD_DARK,
        line_width=2,
        annotation_text=f"Mean: ₹{mean_val:.1f}L",
        annotation_position="top left",
        annotation_font={"size": 11, "color": COLOR_OXBLOOD_DARK},
    )

    layout = get_base_plotly_layout("Asking Price Distribution (in ₹ Lakhs)")
    layout["xaxis"]["title"] = "Asking Price (₹ Lakhs)"
    layout["yaxis"]["title"] = "Number of Listings"
    if log_scale:
        layout["xaxis"]["type"] = "log"
        layout["xaxis"]["title"] = "Asking Price (₹ Lakhs, Log Scale)"

    fig.update_layout(**layout)
    return fig


def create_size_vs_price_chart(df: pd.DataFrame, max_sqft: float = 6000.0) -> go.Figure:
    """Create interactive scatter plot of property area vs. asking price."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**get_base_plotly_layout("No listings match selected filters"))
        return fig

    sample_df = df[df["total_sqft_clean"] <= max_sqft].copy()
    if len(sample_df) > 2500:
        sample_df = sample_df.sample(2500, random_state=config.RANDOM_STATE)

    sample_df["bhk_label"] = sample_df["bhk"].fillna(0).astype(int).astype(str) + " BHK"

    fig = px.scatter(
        sample_df,
        x="total_sqft_clean",
        y="price",
        color="area_type",
        color_discrete_sequence=[COLOR_OXBLOOD, COLOR_ACCENT_AMBER, COLOR_ACCENT_ROSE, COLOR_ACCENT_SLATE],
        hover_data={
            "location": True,
            "bhk_label": True,
            "bath": True,
            "total_sqft_clean": ":.0f",
            "price": ":.1f",
            "area_type": False,
        },
        labels={
            "total_sqft_clean": "Total Area (sq. ft.)",
            "price": "Asking Price (₹ Lakhs)",
            "area_type": "Area Type",
            "bhk_label": "Configuration",
            "location": "Locality",
            "bath": "Baths",
        },
    )

    layout = get_base_plotly_layout("Property Area vs. Asking Price")
    layout["legend"] = {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "font": {"size": 11},
    }
    fig.update_layout(**layout)
    fig.update_traces(marker={"size": 6, "opacity": 0.65})
    return fig


def create_bhk_boxplot(df: pd.DataFrame, max_bhk: int = 5) -> go.Figure:
    """Create interactive box plot of asking prices by bedroom tier."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**get_base_plotly_layout("No listings match selected filters"))
        return fig

    bhk_df = df[(df["bhk"] >= 1) & (df["bhk"] <= max_bhk)].copy()
    bhk_df["bhk_label"] = bhk_df["bhk"].astype(int).astype(str) + " BHK"
    order = [f"{i} BHK" for i in range(1, max_bhk + 1)]

    fig = px.box(
        bhk_df,
        x="bhk_label",
        y="price",
        category_orders={"bhk_label": order},
        color_discrete_sequence=[COLOR_OXBLOOD],
        points=False,  # Hide points for cleaner editorial view
    )

    layout = get_base_plotly_layout("Asking Price Dispersion Across Bedroom Tiers (1 to 5 BHK)")
    layout["xaxis"]["title"] = "Bedroom Configuration"
    layout["yaxis"]["title"] = "Asking Price (₹ Lakhs)"
    layout["yaxis"]["range"] = [0, 450]  # Focus view on core distribution
    fig.update_layout(**layout)
    return fig


def create_locality_bar_chart(
    df: pd.DataFrame, min_listings: int = 15, top_n: int = 15, metric: str = "median"
) -> go.Figure:
    """Create locality comparison chart with listing frequency threshold."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**get_base_plotly_layout("No listings match selected filters"))
        return fig

    counts = df["location"].value_counts()
    valid_locs = counts[counts >= min_listings].index
    loc_df = df[df["location"].isin(valid_locs)]

    if loc_df.empty:
        fig = go.Figure()
        fig.update_layout(**get_base_plotly_layout(f"No localities with ≥{min_listings} listings in filtered set"))
        return fig

    if metric == "median":
        agg = loc_df.groupby("location")["price"].median().sort_values(ascending=True).tail(top_n)
        title = f"Top {top_n} Localities by Median Asking Price (Min. {min_listings} listings)"
        x_title = "Median Asking Price (₹ Lakhs)"
        x_vals = agg.values
        y_vals = agg.index
        hover_tmpl = "<b>%{y}</b><br>Median Price: ₹%{x:.1f} Lakhs<extra></extra>"
    else:
        agg = counts.loc[valid_locs].sort_values(ascending=True).tail(top_n)
        title = f"Top {top_n} Localities by Listing Volume (Min. {min_listings} listings)"
        x_title = "Number of Listings"
        x_vals = agg.values
        y_vals = agg.index
        hover_tmpl = "<b>%{y}</b><br>Listings: %{x:,}<extra></extra>"

    fig = go.Figure(
        go.Bar(
            x=x_vals,
            y=y_vals,
            orientation="h",
            marker_color=COLOR_OXBLOOD,
            marker_line={"color": COLOR_OXBLOOD_DARK, "width": 1},
            hovertemplate=hover_tmpl,
        )
    )

    layout = get_base_plotly_layout(title, height=460)
    layout["xaxis"]["title"] = x_title
    layout["yaxis"]["title"] = ""
    fig.update_layout(**layout)
    return fig


def create_area_type_chart(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart comparing area types by median price and listing counts."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**get_base_plotly_layout("No listings match selected filters"))
        return fig

    agg = df.groupby("area_type").agg(
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        count=("price", "count"),
    ).reset_index().sort_values("median_price", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=agg["area_type"],
            y=agg["median_price"],
            name="Median Price (₹L)",
            marker_color=COLOR_OXBLOOD,
            hovertemplate="<b>%{x}</b><br>Median Price: ₹%{y:.1f} Lakhs<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=agg["area_type"],
            y=agg["mean_price"],
            name="Mean Price (₹L)",
            marker_color=COLOR_ACCENT_AMBER,
            hovertemplate="<b>%{x}</b><br>Mean Price: ₹%{y:.1f} Lakhs<extra></extra>",
        )
    )

    layout = get_base_plotly_layout("Asking Price by Property Area Type")
    layout["xaxis"]["title"] = "Area Type"
    layout["yaxis"]["title"] = "Asking Price (₹ Lakhs)"
    layout["barmode"] = "group"
    layout["legend"] = {"orientation": "h", "y": 1.05, "x": 1, "xanchor": "right"}
    fig.update_layout(**layout)
    return fig


def get_held_out_test_predictions() -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """Retrieve held-out test split and compute predictions using saved pipeline."""
    X, y = bengaluru_train.load_bengaluru_training_data()
    _, X_test, _, y_test = bengaluru_train.train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    model = bengaluru_predict.get_bengaluru_model()
    y_pred = model.predict(X_test)
    return y_test, y_pred, X_test


def create_predicted_vs_actual_chart(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Create interactive Predicted vs Actual asking price scatter plot."""
    plot_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
        "residual": y_test.values - y_pred,
    })
    # Filter view to ₹500L for clarity
    plot_df = plot_df[(plot_df["actual"] <= 500) & (plot_df["predicted"] <= 500)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["actual"],
            y=plot_df["predicted"],
            mode="markers",
            marker={
                "color": COLOR_OXBLOOD,
                "size": 6,
                "opacity": 0.5,
                "line": {"color": COLOR_OXBLOOD_DARK, "width": 0.5},
            },
            name="Test Listings",
            hovertemplate="Actual: ₹%{x:.1f}L<br>Predicted: ₹%{y:.1f}L<extra></extra>",
        )
    )

    # Perfect prediction line y = ŷ
    fig.add_trace(
        go.Scatter(
            x=[0, 500],
            y=[0, 500],
            mode="lines",
            line={"color": COLOR_TEXT_MAIN, "dash": "dash", "width": 1.5},
            name="Perfect Fit (y = ŷ)",
            hoverinfo="none",
        )
    )

    layout = get_base_plotly_layout("Predicted vs. Actual Asking Price on Held-Out Test Set (n = 2,528)")
    layout["xaxis"]["title"] = "Actual Portal Asking Price (₹ Lakhs)"
    layout["yaxis"]["title"] = "Predicted Valuation (₹ Lakhs)"
    layout["xaxis"]["range"] = [0, 500]
    layout["yaxis"]["range"] = [0, 500]
    layout["legend"] = {"orientation": "h", "y": 1.05, "x": 1, "xanchor": "right"}
    fig.update_layout(**layout)
    return fig


def create_residuals_chart(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Create interactive residual error distribution histogram."""
    residuals = y_test.values - y_pred
    res_df = pd.DataFrame({"residual": residuals})
    res_clipped = res_df[res_df["residual"].between(-100, 100)]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=res_clipped["residual"],
            nbinsx=45,
            marker_color=COLOR_OXBLOOD,
            opacity=0.85,
            marker_line={"color": COLOR_BG_CARD, "width": 1},
            hovertemplate="Residual: ₹%{x:.1f}L<br>Count: %{y}<extra></extra>",
            name="Residuals",
        )
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=COLOR_TEXT_MAIN,
        line_width=2,
        annotation_text="Zero Residual (Actual = Predicted)",
        annotation_position="top right",
    )

    layout = get_base_plotly_layout("Residual Error Distribution (Actual − Predicted in ₹ Lakhs)")
    layout["xaxis"]["title"] = "Residual Error (₹ Lakhs, Clipped ±₹100L)"
    layout["yaxis"]["title"] = "Frequency"
    fig.update_layout(**layout)
    return fig


def create_model_comparison_chart(metrics_data: Dict[str, Any]) -> go.Figure:
    """Create interactive bar chart comparing all evaluated models."""
    perf = metrics_data.get("model_performance", {})
    if not perf:
        fig = go.Figure()
        fig.update_layout(title="No model metrics found", **get_base_plotly_layout("Model Benchmark"))
        return fig

    models = list(perf.keys())
    cv_maes = [perf[m].get("cv_mae_lakhs", 0) for m in models]
    test_maes = [perf[m].get("test_mae_lakhs", 0) for m in models]
    test_medaes = [perf[m].get("test_medae_lakhs", 0) for m in models]

    # Filter out models with extreme errors for readable bar scale
    clean_indices = [i for i, m in enumerate(models) if cv_maes[i] < 70]
    filt_models = [models[i] for i in clean_indices]
    filt_cv = [cv_maes[i] for i in clean_indices]
    filt_test = [test_maes[i] for i in clean_indices]
    filt_medae = [test_medaes[i] for i in clean_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=filt_models,
            y=filt_cv,
            name="5-Fold CV MAE (₹L)",
            marker_color=COLOR_ACCENT_AMBER,
            hovertemplate="<b>%{x}</b><br>CV MAE: ₹%{y:.2f}L<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=filt_models,
            y=filt_test,
            name="Held-Out Test MAE (₹L)",
            marker_color=COLOR_OXBLOOD,
            hovertemplate="<b>%{x}</b><br>Test MAE: ₹%{y:.2f}L<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=filt_models,
            y=filt_medae,
            name="Test MedAE (₹L)",
            marker_color=COLOR_ACCENT_ROSE,
            hovertemplate="<b>%{x}</b><br>Test MedAE: ₹%{y:.2f}L<extra></extra>",
        )
    )

    layout = get_base_plotly_layout("Model Benchmark Comparison Across Architectures", height=460)
    layout["xaxis"]["title"] = "Candidate Model Architecture"
    layout["yaxis"]["title"] = "Error (₹ Lakhs)"
    layout["barmode"] = "group"
    layout["legend"] = {"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"}
    layout["xaxis"]["tickangle"] = -25
    fig.update_layout(**layout)
    return fig
