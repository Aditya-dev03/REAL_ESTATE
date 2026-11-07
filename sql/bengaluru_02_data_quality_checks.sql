-- =====================================================================
-- bengaluru_02_data_quality_checks.sql
-- Ingestion & Data Quality audit for the raw Bengaluru housing dataset.
--
-- Run from project root (REAL_ESTATE/):
--     sqlite3 -header -column data/bengaluru.db < sql/bengaluru_02_data_quality_checks.sql
-- =====================================================================

.mode box

-- 1. TOTAL ROW COUNT
SELECT COUNT(*) AS total_rows FROM bengaluru_raw;

-- 2. MISSING VALUES PER COLUMN
SELECT
    SUM(area_type    IS NULL) AS area_type_nulls,
    SUM(availability IS NULL) AS availability_nulls,
    SUM(location     IS NULL) AS location_nulls,
    SUM(size         IS NULL) AS size_nulls,
    SUM(society      IS NULL) AS society_nulls,
    SUM(total_sqft   IS NULL) AS total_sqft_nulls,
    SUM(bath         IS NULL) AS bath_nulls,
    SUM(balcony      IS NULL) AS balcony_nulls,
    SUM(price        IS NULL) AS price_nulls
FROM bengaluru_raw;

-- 3. DUPLICATE LISTINGS (Exact identical records across all 9 fields)
SELECT 
    COUNT(*) - COUNT(DISTINCT area_type || '|' || availability || '|' || location || '|' || size || '|' || IFNULL(society, '') || '|' || total_sqft || '|' || IFNULL(bath, '') || '|' || IFNULL(balcony, '') || '|' || price) AS exact_duplicate_rows
FROM bengaluru_raw;

-- 4. TARGET VARIABLE AUDIT: PRICE IN LAKHS INR
SELECT
    MIN(price)                           AS min_price_lakhs,
    ROUND(AVG(price), 2)                 AS mean_price_lakhs,
    MAX(price)                           AS max_price_lakhs,
    SUM(price <= 0 OR price IS NULL)     AS non_positive_or_null_prices
FROM bengaluru_raw;

-- 5. AREA FORMAT ANOMALIES (total_sqft text format inspection)
SELECT
    COUNT(*) AS total_records,
    SUM(CASE WHEN total_sqft LIKE '%-%' THEN 1 ELSE 0 END) AS range_strings_count,
    SUM(CASE WHEN total_sqft GLOB '*[a-zA-Z]*' AND total_sqft NOT LIKE '%-%' THEN 1 ELSE 0 END) AS unit_conversion_strings_count
FROM bengaluru_raw;

-- 5b. Samples of non-numeric area entries
SELECT total_sqft, COUNT(*) as frequency
FROM bengaluru_raw
WHERE total_sqft LIKE '%-%' OR total_sqft GLOB '*[a-zA-Z]*'
GROUP BY total_sqft
ORDER BY frequency DESC
LIMIT 10;

-- 6. CATEGORICAL DISTRIBUTIONS
-- 6a. Area Type breakdown
SELECT 
    area_type, 
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM bengaluru_raw), 2) AS percentage
FROM bengaluru_raw
GROUP BY area_type
ORDER BY count DESC;

-- 6b. Availability status (Top 5)
SELECT 
    availability, 
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM bengaluru_raw), 2) AS percentage
FROM bengaluru_raw
GROUP BY availability
ORDER BY count DESC
LIMIT 5;

-- 6c. Top 10 Micro-markets / Localities
SELECT 
    location, 
    COUNT(*) AS listing_count,
    ROUND(AVG(price), 2) AS avg_price_lakhs
FROM bengaluru_raw
WHERE location IS NOT NULL
GROUP BY location
ORDER BY listing_count DESC
LIMIT 10;

-- 7. DOMAIN ANOMALIES & SENSITIVITY CHECKS
SELECT
    SUM(bath IS NOT NULL AND bath > 10)  AS properties_with_over_10_baths,
    SUM(bath IS NOT NULL AND bath = 0)   AS zero_bathrooms,
    SUM(balcony IS NOT NULL AND balcony > 4) AS suspicious_balcony_count
FROM bengaluru_raw;
