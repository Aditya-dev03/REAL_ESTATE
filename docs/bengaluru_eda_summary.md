# Bengaluru Housing Market: Exploratory Data Analysis & Empirical Findings Summary

This document presents the verified empirical findings from exploratory data analysis conducted on `data/processed/bengaluru_clean.csv` (12,640 clean listings). All statistics were calculated directly from the clean data and generated figures in `reports/figures/bengaluru/`.

---

## 1. Target Variable: Asking Price Distribution (₹ Lakhs & ₹ Crores)

| Statistical Metric | Clean Listings ($n = 12,640$) | Interpretation & Modeling Impact |
| :--- | :--- | :--- |
| **Minimum** | **₹8.0 Lakh** (₹800,000) | Compact studio flat in peripheral micro-market |
| **25th Percentile ($Q_1$)** | **₹50.0 Lakh** (₹5,000,000) | Entry-level 2 BHK threshold in suburban corridors |
| **Median ($Q_2$)** | **₹72.0 Lakh** (₹7,200,000) | **Robust central tendency of the Bengaluru market** |
| **Mean** | **₹114.1 Lakh** (₹11,413,236) | Elevated 58.5% above median by luxury right-tail |
| **75th Percentile ($Q_3$)** | **₹120.0 Lakh** (₹12,000,000) | Upper-quartile 3 BHK / premium gated apartments |
| **Maximum** | **₹3,600.0 Lakh** (₹36.0 Crore) | Luxury sprawling mansion / prime land parcel |
| **Raw Price Skewness** | **+7.95** | Severe positive skew; violates linear normal error assumptions |
| **$\log_{1p}(\text{Price})$ Skewness** | **+0.86** | Near-normal symmetry; justifies log-transformed target modeling |

---

## 2. Structural Property Drivers

### A. Bedroom Configuration (BHK) Stratification
| Configuration | Listing Count | Percentage | Median Price (₹ Lakhs) | Mean Price (₹ Lakhs) | Interquartile Range (₹ Lakhs) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 BHK** | 636 | 5.0% | **₹34.8L** | ₹45.3L | ₹25.0L – ₹55.0L |
| **2 BHK** | 5,244 | 41.5% | **₹54.0L** | ₹59.9L | ₹42.0L – ₹70.0L |
| **3 BHK** | 4,651 | 36.8% | **₹90.0L** | ₹112.2L | ₹66.0L – ₹130.0L |
| **4 BHK** | 1,389 | 11.0% | **₹205.0L** | ₹273.2L | ₹135.0L – ₹330.0L |
| **5 BHK** | 333 | 2.6% | **₹180.0L** | ₹291.1L | ₹115.0L – ₹360.0L |
| **6 BHK** | 183 | 1.4% | **₹175.0L** | ₹248.2L | ₹125.0L – ₹300.0L |

*Key Finding*: The core residential market is concentrated in **2 BHK and 3 BHK configurations** (accounting for 78.3% of all listings). While median prices step up systematically from 1 BHK to 4 BHK, 5+ BHK properties exhibit wider variance because they predominantly represent independent houses and multi-unit builder floors on varied plot dimensions in non-core locations.

---

### B. Development Typology: Area Type Economics
| Development Typology | Listings ($n$) | Market Share | Median Price | Mean Price | Typical Property Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Super built-up Area** | 8,312 | 65.8% | **₹65.3 Lakhs** | ₹94.3 Lakhs | Multistory society flats with shared amenities |
| **Built-up Area** | 2,381 | 18.8% | **₹70.0 Lakhs** | ₹104.7 Lakhs | Independent floors & older apartments |
| **Carpet Area** | 85 | 0.7% | **₹70.0 Lakhs** | ₹90.1 Lakhs | Direct usable living area listings |
| **Plot Area** | 1,862 | 14.7% | **₹145.0 Lakhs** | **₹216.0 Lakhs** | **Freehold land / independent villas** |

*Economic Insight*: Freehold land (`Plot Area`) commands more than **double the median price** of apartment units (₹145.0L vs. ₹65.3L), reflecting high land scarcity premiums in urban Bengaluru.

---

## 3. Micro-Market & Locality Disparity

To ensure statistical reliability and prevent small-sample distortion, localities are evaluated only if they possess **at least 30 verified listings** ($n \ge 30$).

### Top 15 Premium Micro-Markets (by Median Asking Price)
1. **Malleshwaram**: Median **₹261.0 Lakhs** ($n = 54$, Mean ₹366.8L) — Historic, highly established North-Central residential hub.
2. **Rajaji Nagar**: Median **₹251.0 Lakhs** ($n = 105$, Mean ₹331.6L) — Central commercial and high-density luxury residential zone.
3. **Frazer Town**: Median **₹210.0 Lakhs** ($n = 36$, Mean ₹246.7L) — Prime Cantonment residential enclave.
4. **Indira Nagar**: Median **₹191.0 Lakhs** ($n = 43$, Mean ₹288.3L) — Premier East Bengaluru lifestyle and tech corridor.
5. **Hebbal Kempapura**: Median **₹170.0 Lakhs** ($n = 33$, Mean ₹197.3L) — Airport corridor with lakefront luxury developments.
6. **Thigalarapalya**: Median **₹150.5 Lakhs** ($n = 48$, Mean ₹157.7L)
7. **Hosakerehalli**: Median **₹150.0 Lakhs** ($n = 36$, Mean ₹179.6L)
8. **Basavangudi**: Median **₹150.0 Lakhs** ($n = 31$, Mean ₹180.6L) — Heritage South Bengaluru cultural hub.
9. **Koramangala**: Median **₹145.0 Lakhs** ($n = 70$, Mean ₹205.2L) — Landmark startup & executive residential hub.
10. **Hegde Nagar**: Median **₹123.0 Lakhs** ($n = 47$, Mean ₹139.9L)
11. **Talaghattapura**: Median **₹120.0 Lakhs** ($n = 36$, Mean ₹102.3L)
12. **R.T. Nagar**: Median **₹120.0 Lakhs** ($n = 34$, Mean ₹138.4L)
13. **Green Glen Layout**: Median **₹115.0 Lakhs** ($n = 39$, Mean ₹118.0L)
14. **Hebbal**: Median **₹111.5 Lakhs** ($n = 160$, Mean ₹164.5L)
15. **Kundalahalli**: Median **₹105.0 Lakhs** ($n = 46$, Mean ₹133.3L)

*Key Observation*: Prime heritage and central IT micro-markets command 3x to 4x the median price of peripheral IT hubs like Electronic City (Median: ₹55.0L) and Uttarahalli (Median: ₹61.3L).

---

## 4. Rank Correlation Matrix (Spearman Rank $\rho$)

Because real estate metrics exhibit non-linearities and outliers, Spearman rank correlation was calculated across numerical features:

| Feature Pair | Spearman $\rho$ | Observed Market Association (Non-Causal) |
| :--- | :---: | :--- |
| `price` vs. `price_per_sqft` | **+0.81** | Descriptive only: unit rate intensity correlates with total property valuation |
| `price` vs. `total_sqft_clean` | **+0.75** | Physical footprint is the primary structural attribute associated with price |
| `price` vs. `bath` | **+0.72** | High bathroom counts co-occur with large floorplans and luxury tiers |
| `price` vs. `bhk` | **+0.69** | Bedroom scale tracks overall residential accommodation size |
| `price` vs. `balcony` | **+0.20** | Modest positive association across standard multi-family tiers |
| `price` vs. `is_ready_to_move` | **+0.09** | Slight observed price premium for immediate possession vs completion risk |
| `price` vs. `is_in_named_society` | **-0.06** | Slight negative raw association reflecting high volume of affordable suburban township apartments |

> [!NOTE]
> **Correlation vs. Causation Disclaimer**: Spearman rank coefficients measure monotonic statistical associations across listings, not causal mechanisms. For instance, high bathroom counts do not causally drive property prices upward in isolation; rather, luxury villas and large penthouses simultaneously feature both expansive footprints and numerous ensuite bathrooms. Similarly, `price_per_sqft` is derived directly from the target and is strictly an exploratory tool, never a predictive feature.

---

## 5. Data Quality Audit: Sensitivity Analysis of 151 Excluded Anomaly Records

| Metric | Clean Dataset ($n = 12,640$) | Excluded Anomalies ($n = 151$) | Analytical Impact of Exclusion |
| :--- | :---: | :---: | :--- |
| **Total Area Range** | $250\text{ sq. ft.} - 1.3\text{M sq. ft.}$ | **$1\text{ sq. ft.} - 2,400\text{ sq. ft.}$** | Excludes 7 clear typos ($<100$ sq. ft.) e.g. 1 sq. ft. for 4 bedrooms |
| **Median Area / BHK** | **$495.0\text{ sq. ft./BHK}$** | **$83.3\text{ sq. ft./BHK}$** | Excludes physically impossible packing (e.g. 8 beds in 600 sq. ft.) |
| **Median Price** | **₹72.0 Lakhs** | **₹135.0 Lakhs** | Prevents distortion in high-value plot typos |
| **Median Price / Sq. Ft.** | **₹5,445 / sq. ft.** | **₹14,333 / sq. ft.** | Eliminates severe unit-rate inflation |
| **Maximum Price / Sq. Ft.** | **₹176,471 / sq. ft.** | **₹12,000,000 / sq. ft.** | **Exclusion avoids artificial ₹1.2 Crore/sq. ft. anomaly!** |

> [!IMPORTANT]
> **Heuristic Cutoff Rationale**: The $150\text{ sq. ft./BHK}$ threshold is a documented domain heuristic for data cleanliness rather than a physical law. It isolates severe data entry errors (e.g., 8 bedrooms in 600 sq. ft. at $75\text{ sqft/BHK}$, or 43 bedrooms in 2,400 sq. ft. at $55.8\text{ sqft/BHK}$) while retaining legitimate compact studio and 1–2 BHK apartments ($200\text{--}250\text{ sqft/BHK}$) with review flags for downstream scrutiny.

---

## 6. Critical Modeling Considerations & Questions to Investigate

1. **Target Scaling**: Positive skewness (+7.95) mandates evaluating both original-scale and log-transformed target modeling via `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`.
2. **Strict Zero-Leakage Policy**: `price_per_sqft` is permanently excluded from all predictive modeling feature sets.
3. **Training-Only Locality Consolidation**: To prevent data leakage, rare locality grouping (e.g., minimum frequency thresholds) will be fitted strictly on the training partition inside the scikit-learn pipeline using `OneHotEncoder(min_frequency=10, handle_unknown='ignore')`.
4. **Reproducible Partitioning**: A fixed random seed 80/20 train/test split with cross-validation on the training set protects against test-set contamination during algorithm selection.
5. **Portal Asking Price Scope**: Models predict online asking prices, not registered municipal deed prices; closing prices typically reflect buyer-seller negotiations and builder incentives.
