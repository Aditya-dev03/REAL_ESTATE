"""Unit tests for Bengaluru data preparation and feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.bengaluru_data_prep import (
    clean_bengaluru_data,
    deduplicate_bengaluru_data,
    engineer_bengaluru_features,
    parse_bhk,
    parse_total_sqft,
    save_clean_bengaluru_dataset,
    validate_raw_bengaluru_schema,
)


# ==============================================================================
# 1. Schema Validation Tests
# ==============================================================================
def test_validate_raw_bengaluru_schema_accepts_good_data():
    df = pd.DataFrame(
        columns=[
            "area_type",
            "availability",
            "location",
            "size",
            "society",
            "total_sqft",
            "bath",
            "balcony",
            "price",
        ]
    )
    # Should execute without error
    validate_raw_bengaluru_schema(df)


def test_validate_raw_bengaluru_schema_rejects_missing_column():
    df = pd.DataFrame(columns=["area_type", "location", "price"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_bengaluru_schema(df)


# ==============================================================================
# 2. total_sqft Parsing Tests (Direct, Ranges, Unit Conversions, Edge Cases)
# ==============================================================================
@pytest.mark.parametrize(
    "raw_input,expected_sqft",
    [
        ("1200", 1200.0),
        ("1560.5", 1560.5),
        (2000, 2000.0),
        ("2,500", 2500.0),
    ],
)
def test_parse_total_sqft_numeric(raw_input, expected_sqft):
    assert parse_total_sqft(raw_input) == expected_sqft


@pytest.mark.parametrize(
    "raw_input,expected_sqft",
    [
        ("2100 - 2850", 2475.0),
        ("1000 - 2000", 1500.0),
        (" 1200 - 1400 ", 1300.0),
        ("2249.81 - 4112.19", 3181.0),
    ],
)
def test_parse_total_sqft_hyphenated_range(raw_input, expected_sqft):
    assert parse_total_sqft(raw_input) == expected_sqft


@pytest.mark.parametrize(
    "raw_input,expected_approx",
    [
        ("34.46Sq. Meter", 370.92),
        ("100Sq. Yards", 900.0),
        ("1Acres", 43560.0),
        ("2Guntha", 2178.0),
        ("3Cents", 1306.8),
        ("1Grounds", 2400.0),
        ("1Perch", 272.25),
    ],
)
def test_parse_total_sqft_unit_conversions(raw_input, expected_approx):
    result = parse_total_sqft(raw_input)
    assert result is not None
    assert pytest.approx(result, rel=1e-2) == expected_approx


@pytest.mark.parametrize(
    "bad_input",
    [
        None,
        np.nan,
        "",
        "   ",
        "invalid text",
        "100 - 200 - 300",
        "100 to 200",
        "-500",
        "0",
        "500bananas",
    ],
)
def test_parse_total_sqft_unrecognized_returns_none(bad_input):
    assert parse_total_sqft(bad_input) is None


# ==============================================================================
# 3. size / bhk Parsing Tests
# ==============================================================================
@pytest.mark.parametrize(
    "raw_size,expected_bhk",
    [
        ("2 BHK", 2),
        ("3 BHK", 3),
        ("4 Bedroom", 4),
        ("1 Bedroom", 1),
        ("10 Bedroom", 10),
        ("1 RK", 1),  # 1 Room-Kitchen studio configuration maps to 1 BHK equivalent
        ("6 BHK", 6),
    ],
)
def test_parse_bhk_valid(raw_size, expected_bhk):
    assert parse_bhk(raw_size) == expected_bhk


@pytest.mark.parametrize("bad_size", [None, np.nan, "", "Unknown", "Studio Flat"])
def test_parse_bhk_missing_or_invalid(bad_size):
    assert parse_bhk(bad_size) is None


# ==============================================================================
# 4. Deduplication Tests
# ==============================================================================
def test_deduplicate_bengaluru_data():
    sample_data = pd.DataFrame(
        [
            {"area_type": "Super built-up  Area", "location": "Whitefield", "price": 45.0, "total_sqft": "1000"},
            {"area_type": "Super built-up  Area", "location": "Whitefield", "price": 45.0, "total_sqft": "1000"},  # Duplicate
            {"area_type": "Plot  Area", "location": "Yelahanka", "price": 80.0, "total_sqft": "1500"},
        ]
    )
    df_dedup, stats = deduplicate_bengaluru_data(sample_data)
    assert len(df_dedup) == 2
    assert stats["initial_rows"] == 3
    assert stats["removed_duplicates"] == 1
    assert stats["final_dedup_rows"] == 2


# ==============================================================================
# 5. Missing Values & Society Governance Tests
# ==============================================================================
def test_missing_values_treatment():
    sample_data = pd.DataFrame(
        [
            {
                "area_type": "Super built-up  Area",
                "availability": "Ready To Move",
                "location": None,  # Missing location
                "size": "2 BHK",
                "society": "Prestige",  # Named society
                "total_sqft": "1200",
                "bath": 2.0,
                "balcony": 1.0,
                "price": 60.0,
            },
            {
                "area_type": "Built-up  Area",
                "availability": "18-Dec",
                "location": "Indira Nagar",
                "size": None,  # Missing size
                "society": None,  # Missing society
                "total_sqft": "800",
                "bath": np.nan,
                "balcony": np.nan,
                "price": 50.0,
            },
        ]
    )
    engineered = engineer_bengaluru_features(sample_data)

    # Missing location should be filled with 'Other'
    assert engineered.iloc[0]["location"] == "Other"

    # Named society indicator
    assert engineered.iloc[0]["is_in_named_society"] == 1
    assert engineered.iloc[1]["is_in_named_society"] == 0

    # Society names must NOT be fabricated
    assert pd.isna(engineered.iloc[1]["society"])

    # Missing size parsed as NaN BHK
    assert pd.isna(engineered.iloc[1]["bhk"])

    # Ready to move indicator
    assert engineered.iloc[0]["is_ready_to_move"] == 1
    assert engineered.iloc[1]["is_ready_to_move"] == 0


# ==============================================================================
# 6. Flag Generation and Extreme Values Tests
# ==============================================================================
def test_flags_generation():
    sample_data = pd.DataFrame(
        [
            # Typo area (< 100 sqft)
            {"area_type": "Plot Area", "availability": "Ready To Move", "location": "A", "size": "4 Bedroom", "society": None, "total_sqft": "1", "bath": 4.0, "balcony": 1.0, "price": 100.0},
            # Low sqft per bhk (< 250 sqft)
            {"area_type": "Plot Area", "availability": "Ready To Move", "location": "B", "size": "8 Bedroom", "society": None, "total_sqft": "600", "bath": 8.0, "balcony": 1.0, "price": 75.0},
            # Excessive bathrooms (bath > bhk + 2)
            {"area_type": "Super built-up Area", "availability": "Ready To Move", "location": "C", "size": "6 BHK", "society": None, "total_sqft": "10000", "bath": 9.0, "balcony": 3.0, "price": 900.0},
            # Normal property
            {"area_type": "Super built-up Area", "availability": "Ready To Move", "location": "D", "size": "3 BHK", "society": "Sobha", "total_sqft": "1600", "bath": 3.0, "balcony": 2.0, "price": 120.0},
        ]
    )
    eng = engineer_bengaluru_features(sample_data)

    # Typo flag
    assert eng.iloc[0]["flag_area_typo"] == True
    assert eng.iloc[3]["flag_area_typo"] == False

    # Low sqft per bhk flag
    assert eng.iloc[1]["flag_sqft_per_bhk_low"] == True
    assert eng.iloc[3]["flag_sqft_per_bhk_low"] == False

    # Excessive bathrooms flag
    assert eng.iloc[2]["flag_excessive_bathrooms"] == True
    assert eng.iloc[3]["flag_excessive_bathrooms"] == False


# ==============================================================================
# 7. Safety & Pipeline Execution Tests
# ==============================================================================
def test_save_clean_bengaluru_dataset_refuses_raw_overwrite(tmp_path):
    df_dummy = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(PermissionError, match="Safety violation"):
        save_clean_bengaluru_dataset(df_dummy, {}, output_path=config.BENGALURU_RAW_DATA_PATH)


def test_clean_bengaluru_data_produces_valid_output():
    df_clean, report = clean_bengaluru_data()
    assert len(df_clean) > 12000
    assert report["initial_raw_rows"] == 13320
    assert report["deduplication"]["removed_duplicates"] == 529
    assert "total_sqft_clean" in df_clean.columns
    assert "bhk" in df_clean.columns
    assert "price_per_sqft" in df_clean.columns
