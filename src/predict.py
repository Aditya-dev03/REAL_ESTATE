"""Inference and price prediction helper.

Provides convenience functions to load the trained model pipeline
and predict prices on single property specifications or batch DataFrames.
"""
import joblib
import numpy as np
import pandas as pd

from src import config, train


def load_model(model_path=config.SAVED_MODEL_PATH):
    """Load the serialized price prediction pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def predict_price(model, property_data):
    """Predict real estate sale price for one or more properties.

    Parameters
    ----------
    model : TransformedTargetRegressor or Pipeline
        Fitted model pipeline.
    property_data : pd.DataFrame or dict
        Features matching train.FEATURE_COLUMNS.

    Returns
    -------
    np.ndarray
        Predicted dollar price(s).
    """
    if isinstance(property_data, dict):
        df = pd.DataFrame([property_data])
    elif isinstance(property_data, pd.DataFrame):
        df = property_data.copy()
    else:
        raise TypeError("property_data must be a dict or a pandas DataFrame.")

    # Ensure required feature columns exist
    missing_cols = [c for c in train.FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    preds = model.predict(df[train.FEATURE_COLUMNS])
    # Dollar prices should be positive
    preds = np.clip(preds, a_min=1000.0, a_max=None)
    return preds
