"""Unit and integration tests for model training, inference, and leakage prevention.

Run from project root (REAL_ESTATE/):
    python -m pytest tests/test_train.py -v
"""
import joblib
import numpy as np
import pandas as pd
import pytest

from src import config, predict, train


@pytest.fixture
def clean_sample_data():
    """Load a sample of the cleaned data for testing."""
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    return df


def test_grouped_train_test_split_no_leakage(clean_sample_data):
    """Ensure property IDs in train and test sets are strictly disjoint."""
    X_train, y_train, X_test, y_test, split_info = train.split_data(clean_sample_data)

    assert split_info["id_overlap_count"] == 0
    assert len(X_train) + len(X_test) == len(clean_sample_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Re-verify independently from original IDs
    gss_train_ids = set(clean_sample_data.iloc[X_train.index]["id"])
    gss_test_ids = set(clean_sample_data.iloc[X_test.index]["id"])
    assert gss_train_ids.isdisjoint(gss_test_ids)


def test_feature_list_excludes_leakage():
    """Verify target and identifier columns are explicitly excluded from features."""
    for col in train.EXCLUDED_COLUMNS:
        assert col not in train.FEATURE_COLUMNS

    assert "price" not in train.FEATURE_COLUMNS
    assert "price_per_sqft" not in train.FEATURE_COLUMNS
    assert "id" not in train.FEATURE_COLUMNS
    assert "sale_date" not in train.FEATURE_COLUMNS


def test_saved_model_artifact_exists_and_loads():
    """Verify saved price_model.joblib exists and can be loaded."""
    assert config.SAVED_MODEL_PATH.exists()
    model = predict.load_model(config.SAVED_MODEL_PATH)
    assert hasattr(model, "predict")


def test_saved_model_prediction_finite_and_positive(clean_sample_data):
    """Verify predictions on test sample are finite, positive, and sensible."""
    model = predict.load_model(config.SAVED_MODEL_PATH)
    sample_batch = clean_sample_data[train.FEATURE_COLUMNS].head(10)

    preds = predict.predict_price(model, sample_batch)

    assert len(preds) == 10
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 50000)  # Residential King County homes exceed $50K


def test_single_property_dict_prediction():
    """Verify model can predict for a single raw property dictionary."""
    model = predict.load_model(config.SAVED_MODEL_PATH)

    single_prop = {
        "sqft_living": 2000,
        "sqft_basement": 400,
        "sqft_lot": 6000,
        "sqft_living15": 1900,
        "sqft_lot15": 6000,
        "bathrooms": 2.5,
        "bedrooms": 3.0,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 8,
        "house_age": 30,
        "years_since_update": 10,
        "has_basement": 1,
        "is_renovated": 1,
        "lat": 47.55,
        "long": -122.25,
        "sale_year": 2014,
        "sale_month": 7,
        "zipcode": 98103,
    }

    pred = predict.predict_price(model, single_prop)
    assert len(pred) == 1
    assert 100000 < pred[0] < 5000000


def test_model_metrics_json_validity():
    """Verify that model_metrics.json exists, contains all 5 models, and has valid values."""
    assert config.METRICS_PATH.exists()
    import json

    with open(config.METRICS_PATH) as f:
        data = json.load(f)

    models_expected = [
        "Baseline (Median)",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "HistGradientBoosting",
    ]
    for m in models_expected:
        assert m in data["model_performance"]
        perf = data["model_performance"][m]
        assert perf["mae_dollar"] > 0
        assert perf["rmse_dollar"] > 0
        assert "r2" in perf
