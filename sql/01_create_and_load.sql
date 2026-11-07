-- =====================================================================
-- 01_create_and_load.sql
-- Creates the `house_sales` table in SQLite and loads the raw CSV.
--
-- Run from the project root (REAL_ESTATE/):
--     sqlite3 data/real_estate.db < sql/01_create_and_load.sql
--
-- Note: lines starting with "." are sqlite3 command-line instructions,
-- not standard SQL.
-- =====================================================================

DROP TABLE IF EXISTS house_sales;

CREATE TABLE house_sales (
    row_index      INTEGER,   -- unnamed index column saved in the CSV
    id             INTEGER,   -- property id (not unique: some homes sold twice)
    date           TEXT,      -- sale date, format YYYYMMDDT000000
    price          REAL,      -- sale price in USD (target variable)
    bedrooms       REAL,
    bathrooms      REAL,
    sqft_living    INTEGER,
    sqft_lot       INTEGER,
    floors         REAL,
    waterfront     INTEGER,   -- 1 = waterfront, 0 = not
    view           INTEGER,   -- 0-4
    condition      INTEGER,   -- 1-5
    grade          INTEGER,   -- 1-13
    sqft_above     INTEGER,
    sqft_basement  INTEGER,
    yr_built       INTEGER,
    yr_renovated   INTEGER,   -- 0 = never renovated
    zipcode        INTEGER,
    lat            REAL,
    long           REAL,
    sqft_living15  INTEGER,
    sqft_lot15     INTEGER
);

-- Load the CSV, skipping its header row.
.import --csv --skip 1 data/raw/real_estate_data.csv house_sales

-- sqlite3 .import loads empty CSV fields as empty strings (''), not NULL.
-- Turn them into proper NULLs so missing-value checks work.
UPDATE house_sales SET bedrooms  = NULL WHERE bedrooms  = '';
UPDATE house_sales SET bathrooms = NULL WHERE bathrooms = '';

SELECT 'Loaded rows: ' || COUNT(*) FROM house_sales;
