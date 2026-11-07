"""Tests for Streamlit application input validation."""
from src import app_utils


def test_validate_property_input_valid():
    """Verify that a standard valid property input dictionary passes validation."""
    valid_input = {
        "sqft_living": 2000,
        "sqft_basement": 500,
        "sqft_lot": 6000,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "house_age": 25,
        "years_since_update": 10,
    }
    is_valid, errors = app_utils.validate_property_input(valid_input)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_property_input_negative_living_area():
    """Verify living area <= 0 is rejected."""
    bad_input = {
        "sqft_living": 0,
        "sqft_basement": 0,
        "sqft_lot": 5000,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "house_age": 10,
        "years_since_update": 10,
    }
    is_valid, errors = app_utils.validate_property_input(bad_input)
    assert is_valid is False
    assert any("Living area" in e for e in errors)


def test_validate_property_input_basement_exceeds_living():
    """Verify basement > living area is rejected."""
    bad_input = {
        "sqft_living": 1500,
        "sqft_basement": 2000,
        "sqft_lot": 5000,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "house_age": 10,
        "years_since_update": 10,
    }
    is_valid, errors = app_utils.validate_property_input(bad_input)
    assert is_valid is False
    assert any("Basement area cannot exceed" in e for e in errors)


def test_validate_property_input_unrealistic_sqft_per_bedroom():
    """Verify unrealistic sqft per bedroom is caught."""
    bad_input = {
        "sqft_living": 500,
        "sqft_basement": 0,
        "sqft_lot": 5000,
        "bedrooms": 10,
        "bathrooms": 1.0,
        "house_age": 10,
        "years_since_update": 10,
    }
    is_valid, errors = app_utils.validate_property_input(bad_input)
    assert is_valid is False
    assert any("unrealistically small" in e for e in errors)


def test_validate_property_input_update_exceeds_house_age():
    """Verify years since update cannot exceed house age."""
    bad_input = {
        "sqft_living": 2000,
        "sqft_basement": 0,
        "sqft_lot": 5000,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "house_age": 20,
        "years_since_update": 35,
    }
    is_valid, errors = app_utils.validate_property_input(bad_input)
    assert is_valid is False
    assert any("Years since last update cannot exceed" in e for e in errors)
