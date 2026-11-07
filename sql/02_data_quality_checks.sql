-- =====================================================================
-- 02_data_quality_checks.sql
-- Basic data-quality checks on the `house_sales` table.
-- Run 01_create_and_load.sql first.
--
-- Run from the project root (REAL_ESTATE/):
--     sqlite3 -header -column data/real_estate.db < sql/02_data_quality_checks.sql
-- =====================================================================

-- 1. Size of the dataset
SELECT COUNT(*) AS total_rows
FROM house_sales;

-- 2. Missing values per column (only columns that can realistically be empty)
SELECT
    SUM(price       IS NULL) AS price_nulls,
    SUM(bedrooms    IS NULL) AS bedrooms_nulls,
    SUM(bathrooms   IS NULL) AS bathrooms_nulls,
    SUM(sqft_living IS NULL) AS sqft_living_nulls,
    SUM(zipcode     IS NULL) AS zipcode_nulls,
    SUM(date        IS NULL) AS date_nulls
FROM house_sales;

-- 3. Duplicate property ids (expected: repeat sales of the same home)
SELECT
    COUNT(*) - COUNT(DISTINCT id) AS duplicate_id_rows,
    COUNT(DISTINCT id)            AS unique_properties
FROM house_sales;

-- 3b. Example of a property sold more than once
SELECT id, date, price
FROM house_sales
WHERE id IN (
    SELECT id FROM house_sales GROUP BY id HAVING COUNT(*) > 1 LIMIT 3
)
ORDER BY id, date;

-- 4. Target variable summary
SELECT
    MIN(price)            AS min_price,
    ROUND(AVG(price), 0)  AS avg_price,
    MAX(price)            AS max_price
FROM house_sales;

-- 5. Sale date range (dates are stored as YYYYMMDDT000000)
SELECT
    MIN(SUBSTR(date, 1, 8)) AS first_sale,
    MAX(SUBSTR(date, 1, 8)) AS last_sale
FROM house_sales;

-- 6. Impossible or suspicious values
SELECT
    SUM(price <= 0)                                   AS non_positive_price,
    SUM(sqft_living <= 0)                             AS non_positive_living_area,
    SUM(bedrooms = 0)                                 AS zero_bedrooms,
    SUM(bedrooms > 10)                                AS more_than_10_bedrooms,
    SUM(bathrooms = 0)                                AS zero_bathrooms,
    SUM(yr_renovated > 0 AND yr_renovated < yr_built) AS renovated_before_built,
    SUM(yr_built > CAST(SUBSTR(date, 1, 4) AS INTEGER) + 1) AS built_after_sale
FROM house_sales;

-- 6b. Look at the extreme bedroom counts directly
SELECT id, price, bedrooms, bathrooms, sqft_living
FROM house_sales
WHERE bedrooms > 10
ORDER BY bedrooms DESC;

-- 7. Categorical columns stay inside their documented ranges
SELECT
    SUM(waterfront NOT IN (0, 1))        AS bad_waterfront,
    SUM(view       NOT BETWEEN 0 AND 4)  AS bad_view,
    SUM(condition  NOT BETWEEN 1 AND 5)  AS bad_condition,
    SUM(grade      NOT BETWEEN 1 AND 13) AS bad_grade
FROM house_sales;

-- 8. Consistency check: above-ground + basement area should equal living area
SELECT
    SUM(sqft_above + sqft_basement <> sqft_living) AS inconsistent_area_rows
FROM house_sales;

-- 9. Geographic coverage
SELECT
    COUNT(DISTINCT zipcode)   AS distinct_zipcodes,
    MIN(lat)  AS min_lat,  MAX(lat)  AS max_lat,
    MIN(long) AS min_long, MAX(long) AS max_long
FROM house_sales;
