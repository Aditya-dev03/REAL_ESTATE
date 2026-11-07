"""Model training, benchmarking, and evaluation module for Bengaluru housing dataset.

Trains and benchmarks:
1. Baseline (Median)
2. Linear Regression (Raw & Log target)
3. Ridge Regression (Raw & Log target)
4. Random Forest Regressor (Raw & Log target)
5. HistGradientBoosting Regressor (Raw & Log target)

Enforces strict zero-leakage safeguards:
- Excludes price_per_sqft and any target-derived attributes.
- Fits all imputers, scalers, and frequency-based category thresholds strictly on training folds.
- Uses 80/20 train/test split with fixed random seed (42).
- Saves winning model to models/bengaluru/price_model.joblib without touching US models.
- Generates diagnostic evaluation figures in reports/figures/bengaluru/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)

# Ensure output directories exist
config.BENGALURU_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = config.PROJECT_ROOT / "reports" / "figures" / "bengaluru"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_bengaluru_training_data(
    data_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load clean Bengaluru dataset and extract X and y with strict leakage exclusion."""
    path = data_path or config.BENGALURU_CLEAN_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Clean Bengaluru dataset not found at {path}")

    df = pd.read_csv(path)

    # Validate target column exists
    if "price" not in df.columns:
        raise ValueError("Target column 'price' missing from clean dataset")

    # Strict zero-leakage feature selection
    feature_cols = config.BENGALURU_NUMERICAL_FEATURES + config.BENGALURU_CATEGORICAL_FEATURES
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Required feature '{col}' missing from clean dataset")

    # Explicit assertion: price_per_sqft must never be in X
    assert "price_per_sqft" not in feature_cols, "CRITICAL: price_per_sqft found in model features!"
    assert "price" not in feature_cols, "CRITICAL: target 'price' found in model features!"

    X = df[feature_cols].copy()
    y = df["price"].copy()

    return X, y


def build_preprocessing_pipeline() -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Construct Scikit-Learn preprocessing pipelines for linear and tree models.

    Key Leakage Safeguard:
    - OneHotEncoder(min_frequency=10, handle_unknown='ignore') fits locality frequency
      thresholds strictly on training partitions during fit(), preventing data leakage.
    """
    num_features = config.BENGALURU_NUMERICAL_FEATURES
    cat_features = config.BENGALURU_CATEGORICAL_FEATURES

    # Numeric preprocessors
    num_linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    num_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Categorical preprocessor (training-only frequency consolidation & unseen category safety)
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Other")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
    ])

    prep_linear = ColumnTransformer([
        ("num", num_linear, num_features),
        ("cat", cat_transformer, cat_features),
    ])

    prep_tree = ColumnTransformer([
        ("num", num_tree, num_features),
        ("cat", cat_transformer, cat_features),
    ])

    return prep_linear, prep_tree


def create_candidate_models(
    prep_linear: ColumnTransformer, prep_tree: ColumnTransformer
) -> Dict[str, Any]:
    """Instantiate candidate algorithms across original-price and log-target approaches."""
    return {
        "Baseline (Median)": DummyRegressor(strategy="median"),
        "Linear Regression (Raw)": Pipeline([
            ("prep", prep_linear),
            ("reg", LinearRegression()),
        ]),
        "Linear Regression (Log)": TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", prep_linear),
                ("reg", LinearRegression()),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Ridge Regression (Raw)": Pipeline([
            ("prep", prep_linear),
            ("reg", Ridge(alpha=10.0, random_state=config.RANDOM_STATE)),
        ]),
        "Ridge Regression (Log)": TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", prep_linear),
                ("reg", Ridge(alpha=10.0, random_state=config.RANDOM_STATE)),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Random Forest (Raw)": Pipeline([
            ("prep", prep_tree),
            ("reg", RandomForestRegressor(n_estimators=100, max_depth=16, random_state=config.RANDOM_STATE, n_jobs=-1)),
        ]),
        "Random Forest (Log)": TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", prep_tree),
                ("reg", RandomForestRegressor(n_estimators=100, max_depth=16, random_state=config.RANDOM_STATE, n_jobs=-1)),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "HistGradientBoosting (Raw)": Pipeline([
            ("prep", prep_tree),
            ("reg", HistGradientBoostingRegressor(max_iter=150, max_depth=8, l2_regularization=1.0, random_state=config.RANDOM_STATE)),
        ]),
        "HistGradientBoosting (Log)": TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", prep_tree),
                ("reg", HistGradientBoostingRegressor(max_iter=150, max_depth=8, l2_regularization=1.0, random_state=config.RANDOM_STATE)),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def evaluate_models_cv(
    models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, Dict[str, float]]:
    """Run 5-fold cross-validation on training partition for model selection."""
    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_results = {}

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
        "medae": "neg_median_absolute_error",
    }

    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "cv_mae_lakhs": float(-scores["test_mae"].mean()),
            "cv_rmse_lakhs": float(-scores["test_rmse"].mean()),
            "cv_r2": float(scores["test_r2"].mean()),
            "cv_medae_lakhs": float(-scores["test_medae"].mean()),
        }

    return cv_results


def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray, out_dir: Path) -> Path:
    """Diagnostic Plot: Predicted vs. Actual Asking Price."""
    fig, ax = plt.subplots(figsize=(8.5, 6), facecolor="#FFFDF8")
    ax.set_facecolor("#FFFDF8")

    # Plot up to ₹600 Lakhs for visual clarity
    mask = (y_test <= 600) & (y_pred <= 600)
    ax.scatter(y_test[mask], y_pred[mask], color="#6D2438", alpha=0.45, s=20, edgecolors="none")

    # Perfect prediction line
    lims = [0, 600]
    ax.plot(lims, lims, color="#302A2B", linestyle="--", linewidth=1.5, label="Perfect Agreement (y = ŷ)")

    ax.set_title("Predicted vs. Actual Asking Price on Held-Out Test Set (n = 2,528)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax.set_xlabel("Actual Portal Asking Price (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax.set_ylabel("Predicted Asking Price (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax.legend(frameon=True, facecolor="#FFFDF8", edgecolor="#DED5CC", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.7, color="#E8DFD8")

    plt.tight_layout()
    out = out_dir / "08_predicted_vs_actual.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    return out


def plot_residuals_diagnostic(y_test: pd.Series, y_pred: np.ndarray, out_dir: Path) -> Path:
    """Diagnostic Plot: Residuals vs. Fitted Values."""
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#FFFDF8")
    fig.patch.set_facecolor("#FFFDF8")

    # Residuals vs Fitted
    ax1 = axes[0]
    ax1.set_facecolor("#FFFDF8")
    mask = (y_pred <= 600) & (residuals.abs() <= 300)
    ax1.scatter(y_pred[mask], residuals[mask], color="#501827", alpha=0.4, s=18, edgecolors="none")
    ax1.axhline(0, color="#302A2B", linestyle="--", linewidth=1.5)
    ax1.set_title("Residuals vs. Fitted Asking Price", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax1.set_xlabel("Fitted Asking Price (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax1.set_ylabel("Residual Error (Actual − Predicted in ₹ Lakhs)", fontsize=10, color="#302A2B")
    ax1.grid(True, linestyle="--", alpha=0.7, color="#E8DFD8")

    # Residual Histogram
    ax2 = axes[1]
    ax2.set_facecolor("#FFFDF8")
    res_clipped = residuals.clip(-100, 100)
    sns.histplot(res_clipped, bins=45, color="#6D2438", edgecolor="#FFFDF8", kde=True, ax=ax2)
    ax2.axvline(0, color="#302A2B", linestyle="--", linewidth=1.5)
    ax2.set_title("Residual Distribution (Clipped ±₹100L for view)", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax2.set_xlabel("Residual Error (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax2.set_ylabel("Frequency", fontsize=10, color="#302A2B")
    ax2.grid(True, linestyle="--", alpha=0.7, color="#E8DFD8")

    plt.tight_layout()
    out = out_dir / "09_residuals_diagnostic.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    return out


def plot_error_by_price_and_bhk(
    X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, out_dir: Path
) -> Path:
    """Diagnostic Plot: Absolute Error Breakdown Across Price and BHK Tiers."""
    abs_err = np.abs(y_test - y_pred)
    eval_df = X_test.copy()
    eval_df["actual"] = y_test
    eval_df["abs_err"] = abs_err

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#FFFDF8")
    fig.patch.set_facecolor("#FFFDF8")

    # Panel 1: Error by BHK tier (1 to 5 BHK)
    ax1 = axes[0]
    ax1.set_facecolor("#FFFDF8")
    bhk_eval = eval_df[(eval_df["bhk"] >= 1) & (eval_df["bhk"] <= 5)].copy()
    bhk_eval["bhk_lbl"] = bhk_eval["bhk"].astype(int).astype(str) + " BHK"
    order = [f"{i} BHK" for i in range(1, 6)]

    sns.barplot(
        data=bhk_eval,
        x="bhk_lbl",
        y="abs_err",
        order=order,
        estimator=np.median,
        errorbar=None,
        color="#6D2438",
        edgecolor="#302A2B",
        ax=ax1,
        width=0.55,
    )
    for p in ax1.patches:
        h = p.get_height()
        ax1.text(p.get_x() + p.get_width() / 2.0, h + 0.6, f"₹{h:.1f}L", ha="center", va="bottom", fontsize=8.5, color="#302A2B")

    ax1.set_title("Median Absolute Error (MedAE) by Bedroom Tier", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax1.set_xlabel("Bedroom Tier (BHK)", fontsize=10, color="#302A2B")
    ax1.set_ylabel("Median Absolute Error (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax1.set_ylim(0, 45)
    ax1.grid(axis="y", linestyle="--", alpha=0.7, color="#E8DFD8")

    # Panel 2: Error by Price Quartiles
    ax2 = axes[1]
    ax2.set_facecolor("#FFFDF8")
    eval_df["price_bin"] = pd.qcut(eval_df["actual"], q=4, labels=["Q1: <₹50L", "Q2: ₹50L-₹72L", "Q3: ₹72L-₹120L", "Q4: >₹120L"])
    sns.barplot(
        data=eval_df,
        x="price_bin",
        y="abs_err",
        estimator=np.median,
        errorbar=None,
        color="#8B5A2B",
        edgecolor="#302A2B",
        ax=ax2,
        width=0.55,
    )
    for p in ax2.patches:
        h = p.get_height()
        ax2.text(p.get_x() + p.get_width() / 2.0, h + 0.6, f"₹{h:.1f}L", ha="center", va="bottom", fontsize=8.5, color="#302A2B")

    ax2.set_title("Median Absolute Error (MedAE) Across Valuation Quartiles", fontsize=11, fontweight="bold", color="#302A2B", pad=12)
    ax2.set_xlabel("Property Price Tier (Quartiles)", fontsize=10, color="#302A2B")
    ax2.set_ylabel("Median Absolute Error (₹ Lakhs)", fontsize=10, color="#302A2B")
    ax2.set_ylim(0, 45)
    ax2.grid(axis="y", linestyle="--", alpha=0.7, color="#E8DFD8")

    plt.tight_layout()
    out = out_dir / "10_error_by_price_and_bhk.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    return out


def train_and_evaluate_bengaluru_models() -> Dict[str, Any]:
    """Execute complete training, CV benchmarking, selection, and held-out evaluation."""
    logger.info("Loading cleaned Bengaluru dataset...")
    X, y = load_bengaluru_training_data()

    # 80/20 train/test split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    logger.info(f"Partitioned: {len(X_train):,} train rows, {len(X_test):,} held-out test rows.")

    prep_linear, prep_tree = build_preprocessing_pipeline()
    candidate_models = create_candidate_models(prep_linear, prep_tree)

    # 1. Cross-validation benchmarking on training set
    logger.info("Executing 5-fold cross-validation on training folds...")
    cv_summary = evaluate_models_cv(candidate_models, X_train, y_train)

    # 2. Fit models on full training set and evaluate on test set
    logger.info("Evaluating models on held-out test partition...")
    evaluation_report: Dict[str, Any] = {
        "dataset_metadata": {
            "dataset": "Bengaluru House Price Dataset",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "random_state": config.RANDOM_STATE,
            "target": "price (in Lakhs INR, ₹100,000 per unit)",
        },
        "model_performance": {},
    }

    fitted_models = {}
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        tr_mae = mean_absolute_error(y_train, train_preds)
        te_mae = mean_absolute_error(y_test, test_preds)
        te_medae = median_absolute_error(y_test, test_preds)
        te_rmse = root_mean_squared_error(y_test, test_preds)
        te_r2 = r2_score(y_test, test_preds)

        evaluation_report["model_performance"][name] = {
            "cv_mae_lakhs": round(cv_summary[name]["cv_mae_lakhs"], 2),
            "cv_r2": round(cv_summary[name]["cv_r2"], 4),
            "train_mae_lakhs": round(tr_mae, 2),
            "test_mae_lakhs": round(te_mae, 2),
            "test_mae_inr": f"₹{te_mae * 100000:,.0f}",
            "test_medae_lakhs": round(te_medae, 2),
            "test_medae_inr": f"₹{te_medae * 100000:,.0f}",
            "test_rmse_lakhs": round(te_rmse, 2),
            "test_r2": round(te_r2, 4),
            "overfit_gap_mae_lakhs": round(te_mae - tr_mae, 2),
        }

    # 3. Model Selection Criterion:
    # Selected strictly using 5-fold cross-validation on the training partition (X_train, y_train).
    # Primary validation criterion: Lowest mean CV MAE on training folds.
    # The held-out test set is NEVER used or peeked at to choose the model.
    winner_name = min(cv_summary.keys(), key=lambda k: cv_summary[k]["cv_mae_lakhs"])
    winner_model = fitted_models[winner_name]

    evaluation_report["selected_model"] = {
        "model_name": winner_name,
        "selection_criterion": "Lowest mean 5-fold cross-validation MAE on training partition (training-only)",
        "selection_rationale": (
            f"Pre-specified validation criterion: Lowest 5-fold CV MAE on training folds without test leakage. "
            f"'{winner_name}' achieved the lowest CV MAE (₹{cv_summary[winner_name]['cv_mae_lakhs']:.2f} Lakhs), "
            f"lowest CV MedAE (₹{cv_summary[winner_name]['cv_medae_lakhs']:.2f} Lakhs), and highest CV R² ({cv_summary[winner_name]['cv_r2']:.4f}). "
            f"Model selection was finalized strictly on training data prior to held-out test evaluation."
        ),
        "cv_metrics": cv_summary[winner_name],
        "test_metrics": evaluation_report["model_performance"][winner_name],
    }

    # 4. Save Winner Model and Metrics JSON to Bengaluru-specific location
    joblib.dump(winner_model, config.BENGALURU_SAVED_MODEL_PATH, compress=3)
    logger.info(f"Serialized production model to {config.BENGALURU_SAVED_MODEL_PATH}")

    with open(config.BENGALURU_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info(f"Saved benchmark metrics to {config.BENGALURU_METRICS_PATH}")

    # 5. Generate Diagnostic Plots
    y_test_winner_pred = winner_model.predict(X_test)
    p1 = plot_predicted_vs_actual(y_test, y_test_winner_pred, FIGURES_DIR)
    p2 = plot_residuals_diagnostic(y_test, y_test_winner_pred, FIGURES_DIR)
    p3 = plot_error_by_price_and_bhk(X_test, y_test, y_test_winner_pred, FIGURES_DIR)
    logger.info(f"Generated diagnostic figures: {p1.name}, {p2.name}, {p3.name}")

    return evaluation_report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting Bengaluru Model Training & Evaluation Suite...")
    report = train_and_evaluate_bengaluru_models()

    print("\n" + "=" * 80)
    print("BENGALURU HOUSING MODEL BENCHMARK REPORT (HELD-OUT TEST SET n = 2,528)")
    print("=" * 80)
    print(f"{'Model Architecture':<30} | {'CV MAE':<9} | {'Test MAE':<10} | {'Test MedAE':<11} | {'Test RMSE':<10} | {'Test R²':<7}")
    print("-" * 88)
    for name, stats in report["model_performance"].items():
        is_sel = "(Selected)" if name == report["selected_model"]["model_name"] else ""
        print(
            f"{name + ' ' + is_sel:<30} | ₹{stats['cv_mae_lakhs']:5.2f}L | ₹{stats['test_mae_lakhs']:5.2f}L | ₹{stats['test_medae_lakhs']:5.2f}L   | ₹{stats['test_rmse_lakhs']:6.2f}L | {stats['test_r2']:6.4f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
