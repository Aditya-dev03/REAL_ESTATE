"""Validation helper for Streamlit property inputs."""
from typing import Any, Dict, List, Tuple


def validate_property_input(inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate user input fields for property price prediction.

    Parameters
    ----------
    inputs : dict
        Dictionary of input parameters.

    Returns
    -------
    (is_valid, error_messages) : Tuple[bool, List[str]]
    """
    errors = []

    sqft_living = inputs.get("sqft_living", 0)
    sqft_lot = inputs.get("sqft_lot", 0)
    sqft_basement = inputs.get("sqft_basement", 0)
    bedrooms = inputs.get("bedrooms", 0)
    bathrooms = inputs.get("bathrooms", 0)
    house_age = inputs.get("house_age", 0)
    years_since_update = inputs.get("years_since_update", 0)

    if sqft_living <= 0:
        errors.append("Living area (sqft_living) must be greater than 0.")

    if sqft_lot <= 0:
        errors.append("Lot area (sqft_lot) must be greater than 0.")

    if sqft_basement < 0:
        errors.append("Basement area cannot be negative.")

    if sqft_basement > sqft_living:
        errors.append("Basement area cannot exceed total interior living area.")

    if bedrooms <= 0:
        errors.append("Number of bedrooms must be at least 1.")

    if bathrooms <= 0:
        errors.append("Number of bathrooms must be greater than 0.")

    if bedrooms > 0 and (sqft_living / bedrooms) < 80:
        errors.append(
            f"Living area per bedroom ({sqft_living / bedrooms:.1f} sq.ft) is unrealistically small."
        )

    if house_age < 0:
        errors.append("House age cannot be negative.")

    if years_since_update < 0:
        errors.append("Years since last update cannot be negative.")

    if years_since_update > house_age:
        errors.append(
            "Years since last update cannot exceed total house age."
        )

    return len(errors) == 0, errors
