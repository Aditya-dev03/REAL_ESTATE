"""Inference and valuation prediction module for Bengaluru housing properties.

Loads the serialized production pipeline from models/bengaluru/price_model.joblib
and provides validation and price prediction functionality.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Cached model instance
_MODEL_CACHE: Optional[Any] = None


def get_bengaluru_model(model_path: Optional[Path] = None) -> Any:
    """Load and cache the trained Bengaluru production pipeline."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    path = model_path or config.BENGALURU_SAVED_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Production model artifact not found at {path}. "
            "Please run 'python -m src.bengaluru_train' first."
        )

    _MODEL_CACHE = joblib.load(path)
    logger.info(f"Loaded Bengaluru model from {path}")
    return _MODEL_CACHE


def validate_bengaluru_input(data: Dict[str, Any]) -> List[str]:
    """Validate property specification dictionary before prediction.

    Returns:
        List of human-readable error messages (empty list if valid).
    """
    errors: List[str] = []

    # Required field existence
    required = ["total_sqft_clean", "bhk", "bath", "area_type", "location"]
    for field in required:
        if field not in data or data[field] is None:
            errors.append(f"Missing required property field: '{field}'")

    if errors:
        return errors

    # Numerical range checks
    try:
        sqft = float(data["total_sqft_clean"])
        bhk = float(data["bhk"])
        bath = float(data["bath"])
    except (ValueError, TypeError):
        errors.append("Invalid data type for numerical dimensions (total_sqft, bhk, or bath)")
        return errors

    if sqft < 100:
        errors.append(f"Total area ({sqft:.0f} sq. ft.) is unrealistically small (<100 sq. ft.)")
    if sqft > 100000:
        errors.append(f"Total area ({sqft:,.0f} sq. ft.) exceeds typical residential scale (>100,000 sq. ft.)")

    if bhk < 1 or bhk > 20:
        errors.append(f"Bedroom count ({bhk:.0f}) must be between 1 and 20")

    if bath < 1 or bath > 20:
        errors.append(f"Bathroom count ({bath:.0f}) must be between 1 and 20")

    # Ratio sanity checks
    if bhk > 0 and sqft / bhk < 100:
        errors.append(
            f"Area per bedroom ({sqft/bhk:.1f} sq. ft./BHK) is unrealistically low (<100 sq. ft./BHK)"
        )

    return errors


def format_inr_price(price_lakhs: float) -> str:
    """Format a price in ₹ Lakhs into human-readable Indian currency string.

    Examples:
    - 45.5 -> '₹45.50 Lakhs'
    - 150.0 -> '₹1.50 Crore (₹150.0 Lakhs)'
    """
    if price_lakhs >= 100.0:
        crores = price_lakhs / 100.0
        return f"₹{crores:.2f} Crore (₹{price_lakhs:.1f} Lakhs)"
    return f"₹{price_lakhs:.2f} Lakhs"


def predict_bengaluru_price(
    property_dict: Dict[str, Any], model_path: Optional[Path] = None
) -> Tuple[float, str]:
    """Predict estimated market asking price for a property specification dictionary.

    Args:
        property_dict: Dictionary containing property specifications.
        model_path: Optional override for model artifact location.

    Returns:
        Tuple of (price_lakhs: float, formatted_inr_string: str).
    """
    errors = validate_bengaluru_input(property_dict)
    if errors:
        raise ValueError(f"Property validation failed: {'; '.join(errors)}")

    model = get_bengaluru_model(model_path)

    # Prepare single-row DataFrame with expected schema
    payload = {
        "total_sqft_clean": float(property_dict["total_sqft_clean"]),
        "bhk": float(property_dict["bhk"]),
        "bath": float(property_dict["bath"]),
        "balcony": float(property_dict.get("balcony", 1.0)),
        "is_ready_to_move": int(property_dict.get("is_ready_to_move", 1)),
        "is_in_named_society": int(property_dict.get("is_in_named_society", 0)),
        "area_type": str(property_dict.get("area_type", "Super built-up  Area")).strip(),
        "location": str(property_dict.get("location", "Other")).strip(),
    }

    df_single = pd.DataFrame([payload])
    raw_pred = model.predict(df_single)[0]
    price_lakhs = round(float(raw_pred), 2)
    formatted = format_inr_price(price_lakhs)

    return price_lakhs, formatted
