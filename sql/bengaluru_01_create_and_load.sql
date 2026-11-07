-- =====================================================================
-- bengaluru_01_create_and_load.sql
-- Creates the `bengaluru_raw` table in SQLite and loads the raw CSV.
--
-- Run from the project root (REAL_ESTATE/):
--     sqlite3 data/bengaluru.db < sql/bengaluru_01_create_and_load.sql
-- =====================================================================

DROP TABLE IF EXISTS bengaluru_raw;

CREATE TABLE bengaluru_raw (
    area_type     TEXT,
    availability  TEXT,
    location      TEXT,
    size          TEXT,
    society       TEXT,
    total_sqft    TEXT,    -- Stored as TEXT due to range strings (e.g., '2100 - 2850') and unit conversions
    bath          REAL,
    balcony       REAL,
    price         REAL     -- Target variable: price in Lakhs INR (1 Lakh = 100,000 INR)
);

-- Load the CSV, skipping the header line
.import --csv --skip 1 data/raw/bengaluru_house_data.csv bengaluru_raw

-- Normalize empty CSV fields (which SQLite imports as empty strings '') to proper NULLs
UPDATE bengaluru_raw SET location     = NULL WHERE location     = '';
UPDATE bengaluru_raw SET size         = NULL WHERE size         = '';
UPDATE bengaluru_raw SET society      = NULL WHERE society      = '';
UPDATE bengaluru_raw SET total_sqft   = NULL WHERE total_sqft   = '';
UPDATE bengaluru_raw SET bath         = NULL WHERE bath         = '';
UPDATE bengaluru_raw SET balcony      = NULL WHERE balcony      = '';
UPDATE bengaluru_raw SET price        = NULL WHERE price        = '';

SELECT 'Loaded rows into bengaluru_raw: ' || COUNT(*) FROM bengaluru_raw;
