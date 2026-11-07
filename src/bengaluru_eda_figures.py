"""Script to generate publication-grade analytical visualizations for Bengaluru Housing EDA.

Adheres strictly to the project design system:
- Primary Oxblood / Maroon: #6D2438
- Dark Maroon: #501827
- Off-white card surface: #FFFDF8
- Background canvas: #F5F0E8
- Charcoal primary text: #302A2B
- Secondary text: #746A67
- Muted sand grid/dividers: #DED5CC
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

# Ensure destination directory exists
OUTPUT_DIR = config.PROJECT_ROOT / "reports" / "figures" / "bengaluru"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set editorial styling
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#DED5CC"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#E8DFD8"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7


def load_data():
    df_clean = pd.read_csv(config.BENGALURU_CLEAN_DATA_PATH)
    from src.bengaluru_data_prep import clean_bengaluru_data
    df_all, _ = clean_bengaluru_data(filter_severe_anomalies=False)
    excluded = df_all[~df_all.index.isin(df_clean.index)]
    return df_clean, excluded


def plot_price_distribution(df):
    """01: Raw vs Log-transformed Price Distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#FFFDF8")
    fig.patch.set_facecolor("#FFFDF8")

    # Raw Price (Lakhs)
    ax1 = axes[0]
    ax1.set_facecolor("#FFFDF8")
    # Clip view to 99th percentile for legible visualization
    p99 = df["price"].quantile(0.99)
    sns.histplot(
        df[df["price"] <= p99]["price"],
        bins=50,
        color="#6D2438",
        edgecolor="#FFFDF8",
        kde=True,
        ax=ax1,
        line_kws={"color": "#501827", "linewidth": 2},
    )
    median_val = df["price"].median()
    mean_val = df["price"].mean()
    ax1.axvline(median_val, color="#302A2B", linestyle="--", linewidth=1.8, label=f"Median: ₹{median_val:.1f} Lakhs")
    ax1.axvline(mean_val, color="#8B4513", linestyle=":", linewidth=1.8, label=f"Mean: ₹{mean_val:.1f} Lakhs")
    ax1.set_title("Raw Price Distribution (n = 12,640, clipped at 99th %ile)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax1.set_xlabel("Asking Price (₹ Lakhs) [1 Lakh = ₹100,000]", fontsize=10, color="#302A2B")
    ax1.set_ylabel("Listing Count", fontsize=10, color="#302A2B")
    ax1.legend(frameon=True, facecolor="#FFFDF8", edgecolor="#DED5CC", fontsize=9)
    ax1.grid(True)

    # Log1p Price
    ax2 = axes[1]
    ax2.set_facecolor("#FFFDF8")
    log_price = np.log1p(df["price"])
    sns.histplot(
        log_price,
        bins=40,
        color="#501827",
        edgecolor="#FFFDF8",
        kde=True,
        ax=ax2,
        line_kws={"color": "#6D2438", "linewidth": 2},
    )
    ax2.axvline(log_price.median(), color="#302A2B", linestyle="--", linewidth=1.8, label=f"Median log1p: {log_price.median():.2f}")
    ax2.set_title("Log-Transformed Price [log1p(price)] (Skewness: 0.86 vs Raw: 7.95)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax2.set_xlabel("log1p(Asking Price in ₹ Lakhs)", fontsize=10, color="#302A2B")
    ax2.set_ylabel("Listing Count", fontsize=10, color="#302A2B")
    ax2.legend(frameon=True, facecolor="#FFFDF8", edgecolor="#DED5CC", fontsize=9)
    ax2.grid(True)

    plt.tight_layout()
    out = OUTPUT_DIR / "01_price_distribution_and_log.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")


def plot_price_by_bhk(df):
    """02: Asking Price by BHK Configuration."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#FFFDF8")
    ax.set_facecolor("#FFFDF8")

    # Filter to representative 1 to 6 BHK units
    subset = df[(df["bhk"] >= 1) & (df["bhk"] <= 6)].copy()
    subset["bhk_label"] = subset["bhk"].astype(int).astype(str) + " BHK"

    order = [f"{i} BHK" for i in range(1, 7)]
    sns.boxplot(
        data=subset,
        x="bhk_label",
        y="price",
        order=order,
        color="#D9C7B9",
        showmeans=True,
        meanprops={"marker": "D", "markeredgecolor": "#501827", "markerfacecolor": "#501827", "markersize": 5},
        medianprops={"color": "#6D2438", "linewidth": 2.2},
        boxprops={"edgecolor": "#746A67", "linewidth": 1.2},
        whiskerprops={"color": "#746A67"},
        capprops={"color": "#746A67"},
        flierprops={"marker": ".", "color": "#DED5CC", "alpha": 0.4, "markersize": 4},
        ax=ax,
    )
    # Clip upper axis to ₹600 Lakhs (₹6 Cr) for clear boxplot visibility
    ax.set_ylim(0, 600)
    ax.set_title("Asking Price Stratification Across Bedroom Configurations (1 to 6 BHK)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax.set_xlabel("Bedroom Tier (BHK)", fontsize=10, color="#302A2B")
    ax.set_ylabel("Asking Price (₹ Lakhs, capped at ₹600L for view)", fontsize=10, color="#302A2B")

    # Add sample counts and median labels
    counts = subset["bhk_label"].value_counts()
    medians = subset.groupby("bhk_label")["price"].median()
    for idx, bhk_str in enumerate(order):
        c = counts.get(bhk_str, 0)
        m = medians.get(bhk_str, 0)
        ax.text(idx, 40, f"n={c:,}\nMed: ₹{m:.0f}L", ha="center", va="bottom", fontsize=8.5, color="#302A2B", fontweight="medium", bbox=dict(facecolor="#FFFDF8", edgecolor="#DED5CC", boxstyle="round,pad=0.3"))

    ax.grid(True)
    plt.tight_layout()
    out = OUTPUT_DIR / "02_price_by_bhk.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")


def plot_price_vs_sqft(df):
    """03: Asking Price vs Living Area (Total Sq. Ft.)."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#FFFDF8")
    ax.set_facecolor("#FFFDF8")

    # Filter to residential range <= 8000 sqft and price <= 1000 Lakhs for readable visual
    subset = df[(df["total_sqft_clean"] <= 8000) & (df["price"] <= 1000)].copy()

    sns.scatterplot(
        data=subset,
        x="total_sqft_clean",
        y="price",
        hue="area_type",
        palette={"Super built-up  Area": "#6D2438", "Built-up  Area": "#501827", "Plot  Area": "#8B5A2B", "Carpet  Area": "#746A67"},
        alpha=0.55,
        s=25,
        ax=ax,
    )

    # Linear trendline
    m, b = np.polyfit(subset["total_sqft_clean"], subset["price"], 1)
    x_seq = np.linspace(300, 8000, 100)
    ax.plot(x_seq, m * x_seq + b, color="#302A2B", linestyle="--", linewidth=1.5, label=f"Trendline (slope: ₹{m*100000:.0f}/sqft)")

    ax.set_title("Asking Price vs. Total Area (sq. ft.) Stratified by Area Type (n = 12,488)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax.set_xlabel("Total Area (Square Feet)", fontsize=10, color="#302A2B")
    ax.set_ylabel("Asking Price (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax.legend(title="Area Type", frameon=True, facecolor="#FFFDF8", edgecolor="#DED5CC", fontsize=8.5)
    ax.grid(True)

    plt.tight_layout()
    out = OUTPUT_DIR / "03_price_vs_total_sqft.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")


def plot_price_by_area_type(df):
    """04: Price Comparison by Area Type."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#FFFDF8")
    ax.set_facecolor("#FFFDF8")

    area_stats = df.groupby("area_type")["price"].agg(["count", "median", "mean"]).reset_index()
    area_stats = area_stats.sort_values("median", ascending=False)

    bars = ax.bar(
        area_stats["area_type"],
        area_stats["median"],
        color=["#6D2438", "#8B5A2B", "#501827", "#746A67"],
        edgecolor="#302A2B",
        linewidth=0.8,
        width=0.55,
    )

    for bar, (_, row) in zip(bars, area_stats.iterrows()):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 3,
            f"Median: ₹{row['median']:.1f}L\nMean: ₹{row['mean']:.1f}L\n(n={int(row['count']):,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#302A2B",
        )

    ax.set_ylim(0, 195)
    ax.set_title("Median Asking Price by Area Type (Plot Area Commands High Land Premium)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax.set_xlabel("Development Area Type", fontsize=10, color="#302A2B")
    ax.set_ylabel("Median Asking Price (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax.grid(axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / "04_price_by_area_type.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")


def plot_top15_localities(df):
    """05: Top 15 Localities by Median Asking Price (min 30 listings)."""
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#FFFDF8")
    ax.set_facecolor("#FFFDF8")

    loc_counts = df["location"].value_counts()
    valid_locs = loc_counts[loc_counts >= 30].index
    loc_stats = (
        df[df["location"].isin(valid_locs)]
        .groupby("location")["price"]
        .agg(["count", "median", "mean"])
        .sort_values("median", ascending=True)
        .tail(15)
    )

    y_pos = np.arange(len(loc_stats))
    bars = ax.barh(y_pos, loc_stats["median"], color="#6D2438", edgecolor="#501827", height=0.6)

    for i, (_, row) in enumerate(loc_stats.iterrows()):
        ax.text(
            row["median"] + 3,
            i,
            f"₹{row['median']:.0f}L (n={int(row['count'])})",
            va="center",
            fontsize=8.5,
            color="#302A2B",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(loc_stats.index, fontsize=9.5, color="#302A2B")
    ax.set_xlim(0, 310)
    ax.set_title("Top 15 Bengaluru Localities by Median Asking Price (Threshold: n ≥ 30 Listings)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax.set_xlabel("Median Asking Price (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax.grid(axis="x")

    plt.tight_layout()
    out = OUTPUT_DIR / "05_top15_localities_median_price.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")


def plot_correlation_heatmap(df):
    """06: Spearman Rank Correlation Matrix of Numerical Features."""
    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor="#FFFDF8")
    ax.set_facecolor("#FFFDF8")

    cols = [
        "price",
        "price_per_sqft",
        "total_sqft_clean",
        "bath",
        "bhk",
        "balcony",
        "is_ready_to_move",
        "is_in_named_society",
    ]
    labels = [
        "Price (₹L)",
        "Price/Sq.Ft.",
        "Total Sq.Ft.",
        "Bathrooms",
        "BHK",
        "Balconies",
        "Ready to Move",
        "Named Society",
    ]

    corr = df[cols].corr(method="spearman")

    cmap = sns.diverging_palette(350, 10, s=80, l=45, as_cmap=True)
    sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-0.3,
        vmax=1.0,
        square=True,
        linewidths=0.8,
        linecolor="#FFFDF8",
        cbar_kws={"shrink": 0.75, "label": "Spearman Rank Correlation (ρ)"},
        ax=ax,
    )

    ax.set_title("Spearman Rank Correlation Hierarchy Among Numerical Features", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    plt.tight_layout()
    out = OUTPUT_DIR / "06_correlation_heatmap_numerical.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")


def plot_anomaly_exclusion_impact(df_clean, excluded):
    """07: Visualizing the Data Quality Impact of the 151 Excluded Anomaly Records."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#FFFDF8")
    fig.patch.set_facecolor("#FFFDF8")

    # Panel 1: Sqft per BHK comparison
    ax1 = axes[0]
    ax1.set_facecolor("#FFFDF8")
    sns.kdeplot(
        df_clean["sqft_per_bhk"].clip(upper=1500),
        color="#6D2438",
        fill=True,
        alpha=0.35,
        linewidth=2,
        ax=ax1,
        label=f"Retained Clean Listings (n={len(df_clean):,})",
    )
    sns.kdeplot(
        excluded["sqft_per_bhk"].clip(upper=1500),
        color="#8B4513",
        fill=True,
        alpha=0.55,
        linewidth=2,
        ax=ax1,
        label=f"Excluded Severe Anomalies (n={len(excluded):,})",
    )
    ax1.axvline(150, color="#302A2B", linestyle="--", linewidth=1.5, label="Exclusion Cutoff (<150 sqft/BHK)")
    ax1.set_title("Floor Area per Bedroom: Retained vs. Excluded", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax1.set_xlabel("Living Area per Bedroom (Sq. Ft. / BHK)", fontsize=10, color="#302A2B")
    ax1.set_ylabel("Density", fontsize=10, color="#302A2B")
    ax1.legend(frameon=True, facecolor="#FFFDF8", edgecolor="#DED5CC", fontsize=8.5)
    ax1.grid(True)

    # Panel 2: Price per sqft distortion
    ax2 = axes[1]
    ax2.set_facecolor("#FFFDF8")
    data_comparison = [
        df_clean["price_per_sqft"],
        excluded["price_per_sqft"],
    ]
    bp = ax2.boxplot(
        data_comparison,
        tick_labels=["Clean Listings\n(Median: ₹5,445)", "Excluded Anomalies\n(Median: ₹14,333)"],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markeredgecolor": "#302A2B", "markerfacecolor": "#302A2B"},
        medianprops={"color": "#6D2438", "linewidth": 2.2},
    )
    bp["boxes"][0].set_facecolor("#D9C7B9")
    bp["boxes"][1].set_facecolor("#E8C4C4")

    ax2.set_yscale("log")
    ax2.set_title("Price per Sq. Ft. Distribution (Log Scale Demonstrating Typo Distortion)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax2.set_ylabel("Price per Sq. Ft. (₹ / sq. ft. on Log Scale)", fontsize=10, color="#302A2B")
    ax2.grid(True, which="both", axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / "07_anomaly_exclusion_impact.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")


def main():
    print("Generating Bengaluru EDA figures in reports/figures/bengaluru/...")
    df_clean, excluded = load_data()

    plot_price_distribution(df_clean)
    plot_price_by_bhk(df_clean)
    plot_price_vs_sqft(df_clean)
    plot_price_by_area_type(df_clean)
    plot_top15_localities(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_anomaly_exclusion_impact(df_clean, excluded)
    print("All 7 figures generated successfully!")


if __name__ == "__main__":
    main()
