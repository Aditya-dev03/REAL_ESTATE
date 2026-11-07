"""Data cleaning and feature engineering for the house sales dataset.

Run from the project root (REAL_ESTATE/):

    python -m src.data_prep

This reads  data/raw/real_estate_data.csv        (never modified)
and writes  data/processed/house_sales_clean.csv
            data/processed/cleaning_report.json

Key principles
--------------
1. Never change the raw file. All cleaning happens on a copy in memory.
2. Never drop rows silently. Every decision is logged in a cleaning report.
3. No data leakage. This script applies only fixed, row-by-row rules
   (e.g. "house age = sale year - year built"). Anything that *learns* from
   the data, such as filling missing values with a median or scaling, is left
   for the modelling pipeline, where it is fitted on the training set only.
"""
import json

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Expected raw schema: column name -> "numeric" or "text"
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = {
    "id": "numeric",
    "date": "text",
    "price": "numeric",
    "bedrooms": "numeric",
    "bathrooms": "numeric",
    "sqft_living": "numeric",
    "sqft_lot": "numeric",
    "floors": "numeric",
    "waterfront": "numeric",
    "view": "numeric",
    "condition": "numeric",
    "grade": "numeric",
    "sqft_above": "numeric",
    "sqft_basement": "numeric",
    "yr_built": "numeric",
    "yr_renovated": "numeric",
    "zipcode": "numeric",
    "lat": "numeric",
    "long": "numeric",
    "sqft_living15": "numeric",
    "sqft_lot15": "numeric",
}

# Raw dates look like "20141013T000000".
DATE_FORMAT = "%Y%m%dT%H%M%S"

# A home whose total living area divided by its bedroom count is below this
# value cannot physically contain that many bedrooms. This is a conservative
# rule of thumb. In this dataset it only catches the 33-bedroom / 1,620 sq ft row.
MIN_SQFT_PER_BEDROOM = 100

# Columns removed after feature engineering, and why:
REDUNDANT_COLUMNS = {
    "date": "replaced by the parsed `sale_date` column",
    "sqft_above": "always equals sqft_living - sqft_basement (perfect linear combination)",
    "yr_built": "replaced by `house_age` (= sale_year - yr_built), keeping both is a linear combination",
    "yr_renovated": "replaced by `is_renovated` and `years_since_update` (0 means 'never', not year 0)",
}

# Final column order of the cleaned dataset.
CLEAN_COLUMNS = [
    "id", "sale_date", "sale_year", "sale_month", "price",
    "bedrooms", "bathrooms", "sqft_living", "sqft_basement", "has_basement",
    "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
    "house_age", "is_renovated", "years_since_update",
    "zipcode", "lat", "long", "sqft_living15", "sqft_lot15",
]


# ---------------------------------------------------------------------------
# Small helper
# ---------------------------------------------------------------------------
def _check(condition, message):
    """Raise a clear error if a validation rule fails.

    We use this instead of `assert` because Python can switch asserts off.
    """
    if not condition:
        raise ValueError(f"Validation failed: {message}")


# ---------------------------------------------------------------------------
# 1. Loading and basic validation
# ---------------------------------------------------------------------------
def load_raw_data(path=config.RAW_DATA_PATH):
    """Read the raw CSV. `date` is read as text so we can parse it ourselves."""
    return pd.read_csv(path, dtype={"date": str})


def drop_index_column(df):
    """Remove the unnamed index column that pandas saved into the CSV."""
    index_cols = [col for col in df.columns if col.startswith("Unnamed:")]
    return df.drop(columns=index_cols)


def validate_raw_schema(df):
    """Check that all expected columns exist and have the expected type."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    _check(not missing, f"missing columns: {missing}")

    unexpected = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    _check(not unexpected, f"unexpected columns: {unexpected}")

    for col, kind in EXPECTED_COLUMNS.items():
        if kind == "numeric":
            _check(pd.api.types.is_numeric_dtype(df[col]), f"`{col}` should be numeric")
        else:
            _check(not pd.api.types.is_numeric_dtype(df[col]), f"`{col}` should be text")

    _check(len(df) > 0, "dataset is empty")


def parse_sale_date(df):
    """Convert `date` text into a real datetime column called `sale_date`.

    errors="coerce" turns unreadable dates into NaT instead of crashing, and
    then we fail loudly if any exist, so bad dates are never hidden.
    """
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")
    n_bad = int(df["sale_date"].isna().sum())
    _check(n_bad == 0, f"{n_bad} sale dates could not be parsed")
    return df


# ---------------------------------------------------------------------------
# 2. Data-quality reporting (read-only: these functions never change data)
# ---------------------------------------------------------------------------
def summarise_missing_values(df):
    """Return {column: number_of_missing_values} for columns with gaps."""
    counts = df.isna().sum()
    return {col: int(n) for col, n in counts.items() if n > 0}


def investigate_repeat_sales(df):
    """Check whether repeated `id`s are true duplicates or repeat sales.

    A true duplicate would have the same id AND the same sale date.
    A repeat sale has the same id, the same house attributes, but a different date.
    """
    repeated = df[df["id"].duplicated(keep=False)]
    house_attributes = [
        col for col in EXPECTED_COLUMNS if col not in ("id", "date", "price")
    ]
    # For each repeated id, does any house attribute change between sales?
    attrs_changed = (repeated.groupby("id")[house_attributes].nunique() > 1).any(axis=1)
    price_changed = repeated.groupby("id")["price"].nunique() > 1

    same_id_same_date = int(df.duplicated(subset=["id", "date"]).sum())

    return {
        "ids_sold_more_than_once": int(repeated["id"].nunique()),
        "rows_with_repeated_id": int(len(repeated)),
        "extra_rows_beyond_first_sale": int(df["id"].duplicated().sum()),
        "max_sales_per_id": int(repeated["id"].value_counts().max()) if len(repeated) else 0,
        "same_id_and_same_date": same_id_same_date,
        "ids_with_changed_house_attributes": int(attrs_changed.sum()),
        "ids_with_different_prices": int(price_changed.sum()),
        "conclusion": (
            "repeat sales - kept"
            if same_id_same_date == 0
            else "possible true duplicates - investigate"
        ),
    }


def find_suspicious_values(df):
    """Count values that are impossible or worth a second look.

    Expects `sale_date` to exist (call `parse_sale_date` first).
    """
    sale_year = df["sale_date"].dt.year
    sqft_per_bedroom = df["sqft_living"] / df["bedrooms"]

    # Interquartile-range rule, used here only to *describe* price outliers.
    # These rows are real sales (luxury homes) and are not removed.
    q1, q3 = df["price"].quantile([0.25, 0.75])
    high_price_cutoff = q3 + 3 * (q3 - q1)

    return {
        "non_positive_price": int((df["price"] <= 0).sum()),
        "non_positive_sqft_living": int((df["sqft_living"] <= 0).sum()),
        "bedrooms_over_10": int((df["bedrooms"] > 10).sum()),
        f"sqft_per_bedroom_below_{MIN_SQFT_PER_BEDROOM}": int(
            (sqft_per_bedroom < MIN_SQFT_PER_BEDROOM).sum()
        ),
        "built_after_sale_year": int((df["yr_built"] > sale_year).sum()),
        "renovated_after_sale_year": int(
            ((df["yr_renovated"] > 0) & (df["yr_renovated"] > sale_year)).sum()
        ),
        "area_mismatch_above_plus_basement": int(
            (df["sqft_above"] + df["sqft_basement"] != df["sqft_living"]).sum()
        ),
        "price_above_q3_plus_3iqr": int((df["price"] > high_price_cutoff).sum()),
        "price_q3_plus_3iqr_cutoff": float(high_price_cutoff),
    }


def build_quality_report(df):
    """Combine all read-only checks into one dictionary."""
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "sale_date_range": [
            str(df["sale_date"].min().date()),
            str(df["sale_date"].max().date()),
        ],
        "missing_values": summarise_missing_values(df.drop(columns=["sale_date"])),
        "duplicate_rows": int(df.drop(columns=["sale_date"]).duplicated().sum()),
        "repeat_sales": investigate_repeat_sales(df),
        "suspicious_values": find_suspicious_values(df),
    }


# ---------------------------------------------------------------------------
# 3. Cleaning decisions
# ---------------------------------------------------------------------------
def find_invalid_bedrooms(df, min_sqft_per_bedroom=MIN_SQFT_PER_BEDROOM):
    """Return a True/False mask of rows whose bedroom count is physically implausible."""
    sqft_per_bedroom = df["sqft_living"] / df["bedrooms"]
    return sqft_per_bedroom < min_sqft_per_bedroom


def set_invalid_bedrooms_to_missing(df):
    """Replace implausible bedroom counts with NaN instead of dropping the row.

    Why not drop the row?  The rest of the record (price, size, location)
    looks valid, so we keep it.
    Why not "fix" 33 -> 3?  That is a guess. Marking it missing is honest,
    and the modelling pipeline will impute it like any other missing value.
    """
    df = df.copy()
    mask = find_invalid_bedrooms(df)
    changed_ids = df.loc[mask, "id"].tolist()
    df.loc[mask, "bedrooms"] = np.nan
    return df, changed_ids


# ---------------------------------------------------------------------------
# 4. Feature engineering (fixed row-by-row rules -> no leakage)
# ---------------------------------------------------------------------------
def add_features(df):
    """Create new columns from existing ones.

    Note: we deliberately do NOT create price-based features such as
    price_per_sqft. They use the target and would leak the answer
    into the model.
    """
    df = df.copy()

    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month

    # Age at the time of sale. 12 homes were sold the year before they were
    # built (pre-construction sales), which gives -1. We set those to 0 (new).
    df["house_age"] = (df["sale_year"] - df["yr_built"]).clip(lower=0)

    # A renovation only counts if it happened on or before the sale year.
    # A few homes list a renovation year AFTER the sale. Using that would
    # leak future information into the model, so those count as not renovated.
    renovated_by_sale = (df["yr_renovated"] > 0) & (df["yr_renovated"] <= df["sale_year"])
    df["is_renovated"] = renovated_by_sale.astype(int)

    # Years since the home was built or last renovated, whichever is more recent.
    last_update_year = np.where(renovated_by_sale, df["yr_renovated"], df["yr_built"])
    df["years_since_update"] = (df["sale_year"] - last_update_year).clip(lower=0)

    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    return df


def drop_redundant_columns(df):
    """Remove columns that are duplicated information (see REDUNDANT_COLUMNS)."""
    return df.drop(columns=list(REDUNDANT_COLUMNS))


# ---------------------------------------------------------------------------
# 5. Output validation
# ---------------------------------------------------------------------------
def validate_clean_data(df, n_raw_rows):
    """Sanity-check the cleaned dataset before saving it."""
    _check(list(df.columns) == CLEAN_COLUMNS, "columns do not match CLEAN_COLUMNS")
    _check(len(df) == n_raw_rows, f"row count changed: {n_raw_rows} -> {len(df)}")
    _check(not df.duplicated(subset=["id", "sale_date"]).any(), "duplicate (id, sale_date) pairs")

    # Only bedrooms/bathrooms may still have gaps. The model pipeline imputes them.
    allowed_missing = {"bedrooms", "bathrooms"}
    cols_with_missing = set(summarise_missing_values(df))
    _check(cols_with_missing <= allowed_missing,
           f"unexpected missing values in {cols_with_missing - allowed_missing}")

    _check((df["price"] > 0).all(), "non-positive prices")
    _check((df["house_age"] >= 0).all(), "negative house_age")
    _check((df["years_since_update"] >= 0).all(), "negative years_since_update")
    _check((df["years_since_update"] <= df["house_age"]).all(),
           "years_since_update greater than house_age")
    _check(df["sale_month"].between(1, 12).all(), "sale_month outside 1-12")
    for col in ["is_renovated", "has_basement", "waterfront"]:
        _check(df[col].isin([0, 1]).all(), f"`{col}` is not 0/1")
    _check(not find_invalid_bedrooms(df).any(), "implausible bedroom counts remain")


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------
def clean_data(raw_df):
    """Run every cleaning step and return (clean_df, report)."""
    df = drop_index_column(raw_df)
    validate_raw_schema(df)
    df = parse_sale_date(df)

    report = {"before_cleaning": build_quality_report(df)}

    df, invalid_bedroom_ids = set_invalid_bedrooms_to_missing(df)
    df = add_features(df)
    df = drop_redundant_columns(df)
    df = df[CLEAN_COLUMNS]

    validate_clean_data(df, n_raw_rows=len(raw_df))

    report["decisions"] = {
        "rows_excluded": 0,
        "rows_excluded_reason": "none - suspicious values are fixed or flagged, never silently dropped",
        "bedrooms_set_to_missing_ids": [int(i) for i in invalid_bedroom_ids],
        "bedrooms_set_to_missing_reason":
            f"sqft_living / bedrooms < {MIN_SQFT_PER_BEDROOM} (physically implausible)",
        "missing_value_strategy":
            "left as NaN here; impute with the training-set median inside the model pipeline",
        "repeat_sales": "kept - same house sold on different dates is valid market data",
        "columns_dropped": REDUNDANT_COLUMNS,
        "features_created": [
            "sale_date", "sale_year", "sale_month", "house_age",
            "is_renovated", "years_since_update", "has_basement",
        ],
    }
    report["after_cleaning"] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_values": summarise_missing_values(df),
        "pre_construction_sales_age_set_to_0": report["before_cleaning"]
            ["suspicious_values"]["built_after_sale_year"],
        "renovations_after_sale_ignored": report["before_cleaning"]
            ["suspicious_values"]["renovated_after_sale_year"],
    }
    return df, report


def save_outputs(df, report,
                 data_path=config.CLEAN_DATA_PATH,
                 report_path=config.CLEANING_REPORT_PATH):
    """Write the cleaned CSV and the JSON report (never touches data/raw/)."""
    _check(data_path.resolve() != config.RAW_DATA_PATH.resolve(),
           "refusing to overwrite the raw dataset")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False, date_format="%Y-%m-%d")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


def main():
    raw_df = load_raw_data()
    clean_df, report = clean_data(raw_df)
    save_outputs(clean_df, report)

    before, decisions, after = report["before_cleaning"], report["decisions"], report["after_cleaning"]
    print("=== Data cleaning complete ===")
    print(f"Raw rows            : {len(raw_df):,}")
    print(f"Clean rows          : {after['n_rows']:,}  (excluded: {decisions['rows_excluded']})")
    print(f"Clean columns       : {after['n_columns']}")
    print(f"Missing before      : {before['missing_values']}")
    print(f"Missing after       : {after['missing_values']}  (imputed later, inside the model pipeline)")
    print(f"Bedrooms -> NaN     : ids {decisions['bedrooms_set_to_missing_ids']}")
    print(f"Repeat sales        : {before['repeat_sales']['ids_sold_more_than_once']} homes, "
          f"{before['repeat_sales']['extra_rows_beyond_first_sale']} extra rows "
          f"-> {before['repeat_sales']['conclusion']}")
    print(f"Pre-construction    : {after['pre_construction_sales_age_set_to_0']} sales (house_age set to 0)")
    print(f"Reno after sale     : {after['renovations_after_sale_ignored']} rows (treated as not renovated)")
    print(f"Features created    : {decisions['features_created']}")
    print(f"Columns dropped     : {list(decisions['columns_dropped'])}")
    print(f"Saved               : {config.CLEAN_DATA_PATH.relative_to(config.PROJECT_ROOT)}")
    print(f"Report              : {config.CLEANING_REPORT_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
