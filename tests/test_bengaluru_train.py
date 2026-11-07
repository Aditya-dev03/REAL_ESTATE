"""Unit and integration tests for Bengaluru model training, inference, and leakage prevention.

Tests cover:
- Reproducible 80/20 train/test split and partition isolation
- Zero-leakage policy (exclusion of price_per_sqft and target columns)
- Learned preprocessing (OHE min_frequency fitted on train, handling unseen locations)
- Saved pipeline loading and inference
- Single-property dictionary validation and prediction
- Currency formatting for Indian Rupee (Lakhs / Crores)
- Benchmark metrics JSON schema and values
- Protection of King County model assets
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src import config, bengaluru_predict, bengaluru_train


@pytest.fixture
def clean_bengaluru_data():
    """Load the processed clean Bengaluru dataset."""
    assert config.BENGALURU_CLEAN_DATA_PATH.exists()
    return pd.read_csv(config.BENGALURU_CLEAN_DATA_PATH)


def test_bengaluru_feature_list_excludes_leakage():
    """Verify price_per_sqft, price, and raw uncleaned columns are excluded from features."""
    feature_cols = config.BENGALURU_NUMERICAL_FEATURES + config.BENGALURU_CATEGORICAL_FEATURES

    # Zero-leakage assertions
    assert "price_per_sqft" not in feature_cols, "CRITICAL: price_per_sqft in features!"
    assert "price" not in feature_cols, "CRITICAL: target 'price' in features!"

    for excluded in config.BENGALURU_EXCLUDED_COLUMNS:
        assert excluded not in feature_cols

    # Ensure required features exist in config
    assert "total_sqft_clean" in config.BENGALURU_NUMERICAL_FEATURES
    assert "bhk" in config.BENGALURU_NUMERICAL_FEATURES
    assert "location" in config.BENGALURU_CATEGORICAL_FEATURES
    assert "area_type" in config.BENGALURU_CATEGORICAL_FEATURES


def test_bengaluru_train_test_split_ratio_and_reproducibility(clean_bengaluru_data):
    """Verify reproducible 80/20 split with no overlap and exact sizes."""
    X, y = bengaluru_train.load_bengaluru_training_data()

    X_train1, X_test1, y_train1, y_test1 = bengaluru_train.train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    X_train2, X_test2, y_train2, y_test2 = bengaluru_train.train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Reproducibility with seed 42
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)

    # 80/20 split check
    n_total = len(clean_bengaluru_data)
    assert len(X_train1) == int(np.round(0.8 * n_total))
    assert len(X_test1) == n_total - len(X_train1)
    assert len(X_train1) + len(X_test1) == n_total

    # Index disjointness
    assert set(X_train1.index).isdisjoint(set(X_test1.index))


def test_learned_preprocessing_handles_unseen_locations():
    """Verify OHE handles unseen categories safely and rare thresholding is learned on train."""
    prep_linear, prep_tree = bengaluru_train.build_preprocessing_pipeline()

    # Train data with frequent and infrequent locations
    train_df = pd.DataFrame({
        "total_sqft_clean": [1200.0] * 20,
        "bhk": [2.0] * 20,
        "bath": [2.0] * 20,
        "balcony": [1.0] * 20,
        "is_ready_to_move": [1] * 20,
        "is_in_named_society": [1] * 20,
        "area_type": ["Super built-up  Area"] * 20,
        "location": ["Whitefield"] * 15 + ["RareLocality"] * 5,
    })

    # Fit transformer
    prep_tree.fit(train_df)

    # Unseen location in test set should not crash
    test_df = pd.DataFrame({
        "total_sqft_clean": [1500.0],
        "bhk": [3.0],
        "bath": [2.0],
        "balcony": [1.0],
        "is_ready_to_move": [1],
        "is_in_named_society": [0],
        "area_type": ["Built-up  Area"],
        "location": ["CompletelyNewLocality999"],
    })

    transformed = prep_tree.transform(test_df)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_bengaluru_saved_model_artifact_exists_and_loads():
    """Verify saved Bengaluru model exists and implements predict."""
    assert config.BENGALURU_SAVED_MODEL_PATH.exists()
    model = bengaluru_predict.get_bengaluru_model()
    assert hasattr(model, "predict")


def test_bengaluru_batch_predictions_finite_and_positive(clean_bengaluru_data):
    """Verify predictions on a sample batch from clean dataset are finite, positive, and realistic."""
    model = bengaluru_predict.get_bengaluru_model()
    feature_cols = config.BENGALURU_NUMERICAL_FEATURES + config.BENGALURU_CATEGORICAL_FEATURES
    sample_batch = clean_bengaluru_data[feature_cols].head(20)

    preds = model.predict(sample_batch)

    assert len(preds) == 20
    assert np.all(np.isfinite(preds))
    # Bengaluru homes in dataset are > ₹5 Lakhs and < ₹50,000 Lakhs
    assert np.all(preds > 5.0)
    assert np.all(preds < 10000.0)


def test_bengaluru_single_property_prediction():
    """Verify predict_bengaluru_price returns valid (price_lakhs, formatted_str)."""
    prop = {
        "total_sqft_clean": 1450.0,
        "bhk": 3.0,
        "bath": 2.0,
        "balcony": 2.0,
        "area_type": "Super built-up  Area",
        "location": "Whitefield",
        "is_ready_to_move": 1,
        "is_in_named_society": 1,
    }

    price_lakhs, formatted = bengaluru_predict.predict_bengaluru_price(prop)

    assert isinstance(price_lakhs, float)
    assert price_lakhs > 10.0  # realistic lower bound
    assert "₹" in formatted
    assert ("Lakhs" in formatted or "Crore" in formatted)


def test_validate_bengaluru_input_rejects_anomalies():
    """Verify input validation catches missing fields, tiny areas, and irrational ratios."""
    # Missing required field
    missing_sqft = {"bhk": 2, "bath": 2, "area_type": "Plot  Area", "location": "Indiranagar"}
    errs = bengaluru_predict.validate_bengaluru_input(missing_sqft)
    assert any("total_sqft_clean" in e for e in errs)

    # Unrealistically small area
    tiny_area = {
        "total_sqft_clean": 40.0,
        "bhk": 2,
        "bath": 1,
        "area_type": "Plot  Area",
        "location": "Indiranagar",
    }
    errs = bengaluru_predict.validate_bengaluru_input(tiny_area)
    assert any("<100 sq. ft." in e for e in errs)

    # Severe area-per-BHK heuristic failure (<100 sqft/BHK)
    crowded = {
        "total_sqft_clean": 300.0,
        "bhk": 5,
        "bath": 2,
        "area_type": "Plot  Area",
        "location": "Indiranagar",
    }
    errs = bengaluru_predict.validate_bengaluru_input(crowded)
    assert any("sq. ft./BHK" in e for e in errs)

    # Invalid bedroom count
    absurd_bhk = {
        "total_sqft_clean": 2000.0,
        "bhk": 30,
        "bath": 2,
        "area_type": "Super built-up  Area",
        "location": "Whitefield",
    }
    errs = bengaluru_predict.validate_bengaluru_input(absurd_bhk)
    assert any("Bedroom count" in e for e in errs)


def test_format_inr_price():
    """Verify INR formatting across Lakhs and Crores."""
    assert bengaluru_predict.format_inr_price(45.5) == "₹45.50 Lakhs"
    assert bengaluru_predict.format_inr_price(85.0) == "₹85.00 Lakhs"
    assert bengaluru_predict.format_inr_price(150.0) == "₹1.50 Crore (₹150.0 Lakhs)"
    assert bengaluru_predict.format_inr_price(325.75) == "₹3.26 Crore (₹325.8 Lakhs)"


def test_bengaluru_model_metrics_json():
    """Verify that models/bengaluru/model_metrics.json exists, contains valid metrics and metadata."""
    assert config.BENGALURU_METRICS_PATH.exists()

    with open(config.BENGALURU_METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "model_performance" in data
    assert "selected_model" in data
    assert data["selected_model"]["model_name"] == "Random Forest (Log)"

    # Verify all expected candidate models are recorded
    expected_models = [
        "Baseline (Median)",
        "Linear Regression (Raw)",
        "Linear Regression (Log)",
        "Ridge Regression (Raw)",
        "Ridge Regression (Log)",
        "Random Forest (Raw)",
        "Random Forest (Log)",
        "HistGradientBoosting (Raw)",
        "HistGradientBoosting (Log)",
    ]
    for model_name in expected_models:
        assert model_name in data["model_performance"]
        perf = data["model_performance"][model_name]
        assert perf["test_mae_lakhs"] > 0
        assert perf["test_rmse_lakhs"] > 0
        assert "test_r2" in perf
        assert "test_medae_lakhs" in perf
        assert "test_mae_inr" in perf

    # Test set evaluation should exist for selected model
    assert "test_metrics" in data["selected_model"]
    test_eval = data["selected_model"]["test_metrics"]
    assert test_eval["test_mae_lakhs"] < 35.0
    assert test_eval["test_r2"] > 0.55


def test_king_county_assets_preserved():
    """Ensure King County models, data, and metrics were NOT overwritten."""
    assert config.SAVED_MODEL_PATH.exists()
    assert config.METRICS_PATH.exists()
    assert config.CLEAN_DATA_PATH.exists()

    with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
        us_metrics = json.load(f)

    # US metrics should still track King County model results ($ dollar units)
    assert "Random Forest" in us_metrics["model_performance"]
    assert "mae_dollar" in us_metrics["model_performance"]["Random Forest"]
