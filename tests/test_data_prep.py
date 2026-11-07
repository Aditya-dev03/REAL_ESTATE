"""Tests for src/data_prep.py.

Run from the project root (REAL_ESTATE/):

    python -m pytest -v

Most tests use a tiny hand-made DataFrame, so we know exactly what the
correct answer should be. The last few tests check the real cleaned file.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, data_prep


# ---------------------------------------------------------------------------
# Helper: build a small fake raw dataset with the same columns as the real one
# ---------------------------------------------------------------------------
def make_row(**overrides):
    """One normal-looking house. Pass keyword arguments to change any column."""
    row = {
        "Unnamed: 0": 0, "id": 1, "date": "20141013T000000", "price": 300000.0,
        "bedrooms": 3.0, "bathrooms": 2.0, "sqft_living": 1500, "sqft_lot": 5000,
        "floors": 1.0, "waterfront": 0, "view": 0, "condition": 3, "grade": 7,
        "sqft_above": 1000, "sqft_basement": 500, "yr_built": 1990,
        "yr_renovated": 0, "zipcode": 98000, "lat": 47.5, "long": -122.2,
        "sqft_living15": 1400, "sqft_lot15": 5000,
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_df():
    return pd.DataFrame([
        make_row(id=1),
        # same house sold again later -> repeat sale, not a duplicate
        make_row(id=1, date="20150301T000000", price=320000.0),
        # implausible: 33 bedrooms in 1,620 sq ft
        make_row(id=2, bedrooms=33.0, sqft_living=1620, sqft_above=1620, sqft_basement=0),
        # missing bathrooms
        make_row(id=3, bathrooms=np.nan),
        # pre-construction sale: built 2015, sold 2014
        make_row(id=4, yr_built=2015),
        # renovated in 2015, but sold in 2014 -> renovation is in the future
        make_row(id=5, yr_renovated=2015),
        # renovated in 2005, sold in 2014
        make_row(id=6, yr_renovated=2005),
    ])


# ---------------------------------------------------------------------------
# Loading and schema
# ---------------------------------------------------------------------------
def test_drop_index_column(raw_df):
    df = data_prep.drop_index_column(raw_df)
    assert "Unnamed: 0" not in df.columns
    assert len(df) == len(raw_df)


def test_validate_raw_schema_accepts_good_data(raw_df):
    data_prep.validate_raw_schema(data_prep.drop_index_column(raw_df))


def test_validate_raw_schema_rejects_missing_column(raw_df):
    df = data_prep.drop_index_column(raw_df).drop(columns=["price"])
    with pytest.raises(ValueError, match="missing columns"):
        data_prep.validate_raw_schema(df)


def test_validate_raw_schema_rejects_wrong_type(raw_df):
    df = data_prep.drop_index_column(raw_df)
    df["price"] = "expensive"
    with pytest.raises(ValueError, match="should be numeric"):
        data_prep.validate_raw_schema(df)


def test_parse_sale_date(raw_df):
    df = data_prep.parse_sale_date(raw_df)
    assert df["sale_date"].iloc[0] == pd.Timestamp("2014-10-13")


def test_parse_sale_date_fails_loudly_on_bad_date(raw_df):
    raw_df.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="could not be parsed"):
        data_prep.parse_sale_date(raw_df)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def test_repeat_sales_are_not_duplicates(raw_df):
    result = data_prep.investigate_repeat_sales(data_prep.drop_index_column(raw_df))
    assert result["ids_sold_more_than_once"] == 1
    assert result["same_id_and_same_date"] == 0
    assert result["conclusion"] == "repeat sales - kept"


def test_true_duplicate_is_detected(raw_df):
    df = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    result = data_prep.investigate_repeat_sales(data_prep.drop_index_column(df))
    assert result["same_id_and_same_date"] == 1


def test_missing_values_summary(raw_df):
    assert data_prep.summarise_missing_values(raw_df) == {"bathrooms": 1}


# ---------------------------------------------------------------------------
# Cleaning decisions
# ---------------------------------------------------------------------------
def test_invalid_bedrooms_become_missing_but_row_is_kept(raw_df):
    df, changed_ids = data_prep.set_invalid_bedrooms_to_missing(raw_df)
    assert changed_ids == [2]
    assert len(df) == len(raw_df)
    assert pd.isna(df.loc[df["id"] == 2, "bedrooms"]).all()
    # the input is not modified in place
    assert raw_df.loc[raw_df["id"] == 2, "bedrooms"].iloc[0] == 33


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def features_for(raw_df, house_id):
    df = data_prep.add_features(data_prep.parse_sale_date(raw_df))
    return df[df["id"] == house_id].iloc[0]


def test_sale_year_and_month(raw_df):
    row = features_for(raw_df, 3)
    assert (row["sale_year"], row["sale_month"]) == (2014, 10)


def test_house_age(raw_df):
    assert features_for(raw_df, 3)["house_age"] == 2014 - 1990


def test_pre_construction_sale_gets_age_zero(raw_df):
    assert features_for(raw_df, 4)["house_age"] == 0


def test_renovation_after_sale_is_ignored(raw_df):
    row = features_for(raw_df, 5)
    assert row["is_renovated"] == 0
    assert row["years_since_update"] == row["house_age"]


def test_renovation_before_sale_counts(raw_df):
    row = features_for(raw_df, 6)
    assert row["is_renovated"] == 1
    assert row["years_since_update"] == 2014 - 2005


def test_has_basement(raw_df):
    assert features_for(raw_df, 3)["has_basement"] == 1
    assert features_for(raw_df, 2)["has_basement"] == 0


# ---------------------------------------------------------------------------
# Full pipeline on the fake data
# ---------------------------------------------------------------------------
def test_clean_data_keeps_all_rows_and_drops_redundant_columns(raw_df):
    clean_df, report = data_prep.clean_data(raw_df)
    assert len(clean_df) == len(raw_df)
    assert list(clean_df.columns) == data_prep.CLEAN_COLUMNS
    for col in data_prep.REDUNDANT_COLUMNS:
        assert col not in clean_df.columns
    assert report["decisions"]["rows_excluded"] == 0


def test_save_outputs_refuses_to_overwrite_raw(raw_df, tmp_path):
    clean_df, report = data_prep.clean_data(raw_df)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        data_prep.save_outputs(clean_df, report,
                               data_path=config.RAW_DATA_PATH,
                               report_path=tmp_path / "r.json")


# ---------------------------------------------------------------------------
# Checks on the real files
# ---------------------------------------------------------------------------
def test_raw_file_is_untouched():
    raw = pd.read_csv(config.RAW_DATA_PATH)
    assert raw.shape == (21613, 22)
    assert "sqft_above" in raw.columns  # still the original schema


@pytest.mark.skipif(not config.CLEAN_DATA_PATH.exists(),
                    reason="run `python -m src.data_prep` first")
def test_real_clean_file_is_valid():
    clean = pd.read_csv(config.CLEAN_DATA_PATH, parse_dates=["sale_date"])
    data_prep.validate_clean_data(clean, n_raw_rows=21613)
    assert data_prep.summarise_missing_values(clean) == {"bedrooms": 14, "bathrooms": 10}
