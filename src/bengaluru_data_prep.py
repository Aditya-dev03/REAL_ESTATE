"""Data preparation and feature engineering module for Bengaluru House Price data.

Provides modular, robust functions to:
- Validate raw schema.
- Safely parse mixed-format total_sqft (ranges, unit conversions, and direct floats).
- Parse unit size strings into integer BHK (with 1 RK documented as 1 room studio).
- Transparently handle missing values without fabricating society names.
- Deduplicate repeated portal listings (explaining SQL vs. Pandas counts).
- Engineer domain features and extreme-value review flags.
- Export cleaned dataset to data/processed/bengaluru_clean.csv.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Expected raw columns from Kaggle / public mirror
EXPECTED_BENGALURU_COLUMNS = [
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

# Standard unit conversion factors to square feet
# Documented real estate metric conversions:
# - 1 Sq. Meter = 10.7639 Sq. Ft.
# - 1 Sq. Yard = 9.0 Sq. Ft.
# - 1 Acre = 43,560.0 Sq. Ft.
# - 1 Guntha = 1,089.0 Sq. Ft. (1/40th of an acre, standard Karnataka measure)
# - 1 Cent = 435.6 Sq. Ft. (1/100th of an acre, standard South India measure)
# - 1 Ground = 2,400.0 Sq. Ft. (traditional Tamil Nadu / South India urban plot unit)
# - 1 Perch = 272.25 Sq. Ft. (traditional Commonwealth measure, 1/160th acre)
UNIT_CONVERSION_FACTORS: Dict[str, float] = {
    "sqmeter": 10.7639,
    "sqmeters": 10.7639,
    "sqm": 10.7639,
    "sqyard": 9.0,
    "sqyards": 9.0,
    "sqyd": 9.0,
    "acre": 43560.0,
    "acres": 43560.0,
    "guntha": 1089.0,
    "gunthas": 1089.0,
    "cent": 435.6,
    "cents": 435.6,
    "ground": 2400.0,
    "grounds": 2400.0,
    "perch": 272.25,
    "perches": 272.25,
}


def validate_raw_bengaluru_schema(df: pd.DataFrame) -> None:
    """Validate that input DataFrame contains all expected raw Bengaluru columns."""
    missing = [c for c in EXPECTED_BENGALURU_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw Bengaluru dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def parse_total_sqft(val: Any) -> Optional[float]:
    """Parse mixed-format total_sqft into numeric square feet.

    Supported patterns:
    1. Direct numeric float or int (e.g., '1200', 1056.5).
    2. Hyphenated range (e.g., '2100 - 2850' -> average 2475.0).
    3. Supported unit conversions (e.g., '34.46Sq. Meter' -> 370.92 sqft).

    Returns:
        float square footage if parsed successfully, or None if malformed/unrecognized.
    """
    if pd.isna(val):
        return None

    s = str(val).strip().replace(",", "")
    if not s:
        return None

    # Case 1: Direct numeric
    try:
        f_val = float(s)
        if f_val > 0:
            return round(f_val, 2)
        return None
    except ValueError:
        pass

    # Case 2: Hyphenated range (e.g., '2100 - 2850')
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2:
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                if low > 0 and high >= low:
                    return round((low + high) / 2.0, 2)
            except ValueError:
                return None
        return None

    # Case 3: Unit conversions (e.g. '34.46Sq. Meter', '1100Sq. Yards')
    match = re.match(r"^([\d.]+)\s*([a-zA-Z\s.]+)$", s)
    if match:
        try:
            num = float(match.group(1))
            unit_raw = match.group(2).lower().replace(" ", "").replace(".", "")
            if unit_raw in UNIT_CONVERSION_FACTORS and num > 0:
                factor = UNIT_CONVERSION_FACTORS[unit_raw]
                return round(num * factor, 2)
        except (ValueError, KeyError):
            return None

    return None


def parse_bhk(val: Any) -> Optional[int]:
    """Parse unit size string into an integer bedroom count (BHK).

    Terminology & Representation:
    - BHK: Bedroom, Hall (living room), Kitchen. E.g. '2 BHK' -> 2.
    - Bedroom: Independent house / villa bedroom count. E.g. '4 Bedroom' -> 4.
    - RK: Room, Kitchen (studio apartment without a separate hall).
      In Indian real estate, 1 RK comprises 1 combined living/sleeping room plus
      a kitchen. It represents 1 room unit (BHK equivalent = 1). E.g. '1 RK' -> 1.

    Returns:
        int bedroom count or None if missing or unparseable.
    """
    if pd.isna(val):
        return None

    s = str(val).strip()
    match = re.match(r"^(\d+)\s*(?:BHK|Bedroom|RK)?", s, re.IGNORECASE)
    if match:
        try:
            bhk_val = int(match.group(1))
            if bhk_val > 0:
                return bhk_val
        except ValueError:
            return None
    return None


def deduplicate_bengaluru_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Deduplicate exact duplicate listings across all 9 raw fields.

    Discrepancy Investigation Note (SQL vs Pandas):
    - In SQL concatenation `col1 || '|' || col2...`, if any field (such as location
      or size) is NULL, the entire concatenation evaluates to NULL. SQLite's
      `COUNT(DISTINCT NULL)` ignores NULLs, causing all NULL-bearing rows to collapse
      and reporting 546 apparent duplicates.
    - When grouped via SQL `GROUP BY col1, col2, ... col9 HAVING count(*) > 1`,
      SQLite reports exactly 529 excess duplicate rows, perfectly matching
      Pandas `df.duplicated().sum()` (529 rows).
    """
    initial_rows = len(df)
    df_dedup = df.drop_duplicates().copy()
    final_rows = len(df_dedup)
    removed_rows = initial_rows - final_rows

    report = {
        "initial_rows": initial_rows,
        "removed_duplicates": removed_rows,
        "final_dedup_rows": final_rows,
    }
    return df_dedup, report


def engineer_bengaluru_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer clean numerical fields, domain features, and review flags.

    Missing Value Policy:
    - location: 1 missing row is labeled 'Other' to preserve the listing.
    - size / bhk: 16 missing values parsed as NaN (imputed in pipeline by median).
    - bath: 73 missing values left as NaN (imputed in pipeline by BHK median).
    - balcony: 609 missing values left as NaN (imputed in pipeline).
    - society: 5,502 missing values (41.3%). In accordance with data governance rules,
      society names are NEVER fabricated. Instead, a binary indicator `is_in_named_society`
      is engineered (1 if society exists, 0 otherwise).
    """
    df = df.copy()

    # Clean location text
    df["location"] = df["location"].fillna("Other").astype(str).str.strip()
    df["location"] = df["location"].replace("", "Other")

    # Group rare locations (frequency <= 10) into 'Other'
    loc_counts = df["location"].value_counts()
    rare_locs = set(loc_counts[loc_counts <= 10].index)
    df["location_clean"] = df["location"].apply(lambda x: "Other" if x in rare_locs else x)

    # Clean area_type
    df["area_type"] = df["area_type"].astype(str).str.strip()

    # Parse total_sqft and bhk
    df["total_sqft_clean"] = df["total_sqft"].apply(parse_total_sqft)
    df["bhk"] = df["size"].apply(parse_bhk)

    # Economic metrics
    df["sqft_per_bhk"] = np.where(
        (df["bhk"].notna()) & (df["bhk"] > 0) & (df["total_sqft_clean"].notna()),
        df["total_sqft_clean"] / df["bhk"],
        np.nan,
    )
    # Price per sq. ft. in real INR: (price in Lakhs * 100,000) / total_sqft
    df["price_per_sqft"] = np.where(
        (df["total_sqft_clean"].notna()) & (df["total_sqft_clean"] > 0),
        (df["price"] * 100000.0) / df["total_sqft_clean"],
        np.nan,
    )

    # Binary indicators
    df["is_ready_to_move"] = (df["availability"].astype(str).str.strip() == "Ready To Move").astype(int)
    df["is_in_named_society"] = df["society"].notna().astype(int)

    # Review Flags (Non-destructive diagnostic markers)
    # 1. Obvious data entry area typos (< 100 sq. ft. e.g. 1 sqft, 11 sqft)
    df["flag_area_typo"] = (df["total_sqft_clean"].notna()) & (df["total_sqft_clean"] < 100)
    # 2. Unusually low area per BHK (< 250 sqft/BHK)
    df["flag_sqft_per_bhk_low"] = (df["sqft_per_bhk"].notna()) & (df["sqft_per_bhk"] < 250)
    # 3. Excessive bathrooms relative to bedrooms
    df["flag_excessive_bathrooms"] = (df["bath"].notna()) & (df["bhk"].notna()) & (df["bath"] > (df["bhk"] + 2))
    # 4. Extreme unit size
    df["flag_extreme_bhk"] = (df["bhk"].notna()) & (df["bhk"] > 10)
    df["flag_extreme_bath"] = (df["bath"].notna()) & (df["bath"] > 10)

    return df


def clean_bengaluru_data(
    raw_path: Optional[Path] = None,
    filter_severe_anomalies: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete end-to-end data cleaning for Bengaluru housing data.

    Args:
        raw_path: Path to raw CSV. Defaults to config.BENGALURU_RAW_DATA_PATH.
        filter_severe_anomalies: If True, filters out physical impossibility data typos:
          - Total area < 100 sq. ft. (7 clear typos like 1 sq. ft. for 4 bedrooms).
          - Extreme sqft/BHK < 150 where bedroom packing is physically impossible.
          Does NOT automatically drop valid luxury estates with bath > bhk + 2.

    Returns:
        Tuple of (cleaned DataFrame, audit report dictionary).
    """
    raw_file = raw_path or config.BENGALURU_RAW_DATA_PATH
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw Bengaluru dataset not found at {raw_file}")

    df_raw = pd.read_csv(raw_file)
    validate_raw_bengaluru_schema(df_raw)

    initial_row_count = len(df_raw)
    raw_missing = df_raw.isnull().sum().to_dict()

    # Step 1: Deduplication
    df_dedup, dedup_stats = deduplicate_bengaluru_data(df_raw)

    # Step 2: Feature Engineering & Flags
    df_engineered = engineer_bengaluru_features(df_dedup)

    # Step 3: Diagnostic Flag Summaries
    flags_summary = {
        "flag_area_typo_count": int(df_engineered["flag_area_typo"].sum()),
        "flag_sqft_per_bhk_low_count": int(df_engineered["flag_sqft_per_bhk_low"].sum()),
        "flag_excessive_bathrooms_count": int(df_engineered["flag_excessive_bathrooms"].sum()),
        "flag_extreme_bhk_count": int(df_engineered["flag_extreme_bhk"].sum()),
        "flag_extreme_bath_count": int(df_engineered["flag_extreme_bath"].sum()),
    }

    # Step 4: Selective Anomaly Filtering
    # Justification:
    # - total_sqft_clean < 100: Exactly 7 records have total areas between 1 and 60 sq. ft.
    #   for 1 to 7 bedrooms. These are indisputable data entry typos.
    # - sqft_per_bhk < 150: Properties with less than 150 sq. ft. per bedroom (such as 8 bedrooms
    #   in 600 sq. ft.) distort price-per-sqft estimation.
    # - Bath > bhk + 2 records are RETAINED with flags, as luxury mansions (e.g. 11,338 sqft 6 BHK
    #   with 9 baths) represent legitimate market properties.
    excluded_rows_count = 0
    if filter_severe_anomalies:
        severe_typos = df_engineered["flag_area_typo"]
        severe_sqft_bhk = (df_engineered["sqft_per_bhk"].notna()) & (df_engineered["sqft_per_bhk"] < 150)
        exclude_mask = severe_typos | severe_sqft_bhk
        excluded_rows_count = int(exclude_mask.sum())
        df_clean = df_engineered[~exclude_mask].copy()
    else:
        df_clean = df_engineered.copy()

    clean_missing = df_clean.isnull().sum().to_dict()

    report: Dict[str, Any] = {
        "initial_raw_rows": initial_row_count,
        "raw_missing_values": raw_missing,
        "deduplication": dedup_stats,
        "flags_summary": flags_summary,
        "severe_anomalies_filtered": excluded_rows_count,
        "final_clean_rows": len(df_clean),
        "clean_missing_values": clean_missing,
        "price_summary_lakhs": {
            "min": float(df_clean["price"].min()),
            "median": float(df_clean["price"].median()),
            "mean": round(float(df_clean["price"].mean()), 2),
            "max": float(df_clean["price"].max()),
        },
    }

    return df_clean, report


def save_clean_bengaluru_dataset(
    df: pd.DataFrame,
    report: Dict[str, Any],
    output_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> None:
    """Save cleaned Bengaluru dataset to CSV and report to JSON, refusing to overwrite raw data."""
    out_file = output_path or config.BENGALURU_CLEAN_DATA_PATH
    rep_file = report_path or config.BENGALURU_CLEANING_REPORT_PATH

    # Safety guard: Never overwrite raw data files
    if out_file.resolve() == config.BENGALURU_RAW_DATA_PATH.resolve() or out_file.resolve() == config.RAW_DATA_PATH.resolve():
        raise PermissionError(f"Safety violation: Refusing to overwrite raw dataset at {out_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info(f"Saved cleaned Bengaluru dataset to {out_file} ({len(df):,} rows)")

    with open(rep_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved Bengaluru cleaning report to {rep_file}")


def main() -> None:
    """CLI entrypoint for Bengaluru data cleaning."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Executing Bengaluru data preparation pipeline...")

    df_clean, report = clean_bengaluru_data()
    save_clean_bengaluru_dataset(df_clean, report)

    print("\n" + "=" * 60)
    print("BENGALURU DATA PREPARATION AUDIT REPORT")
    print("=" * 60)
    print(f"Initial raw listings:          {report['initial_raw_rows']:,}")
    print(f"Duplicate listings removed:    {report['deduplication']['removed_duplicates']:,}")
    print(f"Deduplicated listings:         {report['deduplication']['final_dedup_rows']:,}")
    print(f"Review Flags Identified:")
    print(f"  - Obvious area typos (<100 sqft):   {report['flags_summary']['flag_area_typo_count']}")
    print(f"  - Low sqft per BHK (<250 sqft):     {report['flags_summary']['flag_sqft_per_bhk_low_count']}")
    print(f"  - Excessive bathrooms (bath>bhk+2): {report['flags_summary']['flag_excessive_bathrooms_count']}")
    print(f"  - Extreme bedrooms (>10 BHK):       {report['flags_summary']['flag_extreme_bhk_count']}")
    print(f"Severe anomalies excluded:     {report['severe_anomalies_filtered']:,}")
    print(f"Final clean listings:          {report['final_clean_rows']:,}")
    print(f"Price range (₹ Lakhs):         ₹{report['price_summary_lakhs']['min']}L – ₹{report['price_summary_lakhs']['max']}L")
    print(f"Median sale price:             ₹{report['price_summary_lakhs']['median']} Lakhs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
