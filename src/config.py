"""Central place for file paths and project-wide constants.

Every script imports paths from here, so nothing is hard-coded twice.
"""
from pathlib import Path

# REAL_ESTATE/ (this file lives in REAL_ESTATE/src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "real_estate_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = PROCESSED_DIR / "house_sales_clean.csv"
CLEANING_REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODEL_PATH = MODELS_DIR / "price_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Used later for train/test splits and models so results are reproducible.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Bengaluru Housing Dataset paths
BENGALURU_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bengaluru_house_data.csv"
BENGALURU_CLEAN_DATA_PATH = PROCESSED_DIR / "bengaluru_clean.csv"
BENGALURU_CLEANING_REPORT_PATH = PROCESSED_DIR / "bengaluru_cleaning_report.json"
BENGALURU_MODELS_DIR = MODELS_DIR / "bengaluru"
BENGALURU_SAVED_MODEL_PATH = BENGALURU_MODELS_DIR / "price_model.joblib"
BENGALURU_METRICS_PATH = BENGALURU_MODELS_DIR / "model_metrics.json"

# Bengaluru Feature Specifications (Zero Leakage)
BENGALURU_NUMERICAL_FEATURES = [
    "total_sqft_clean",
    "bhk",
    "bath",
    "balcony",
    "is_ready_to_move",
    "is_in_named_society",
]

BENGALURU_CATEGORICAL_FEATURES = [
    "area_type",
    "location",
]

# Columns strictly excluded from model training to prevent target leakage or review bias
BENGALURU_EXCLUDED_COLUMNS = [
    "price",
    "price_per_sqft",
    "size",
    "society",
    "total_sqft",
    "availability",
    "sqft_per_bhk",
    "location_clean",
    "flag_area_typo",
    "flag_sqft_per_bhk_low",
    "flag_excessive_bathrooms",
    "flag_extreme_bhk",
    "flag_extreme_bath",
]


