# Real Estate Market Intelligence & Predictive Valuation Platforms

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-darkblue.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/pytest-101%20passed-brightgreen.svg)](https://docs.pytest.org/)

An enterprise-grade, end-to-end data analytics, machine learning, and interactive visualization repository hosting two fully isolated market intelligence products:

1. **Bengaluru Real Estate Terminal (`app/bengaluru_app.py`)**: Interactive **Plotly** analytics, market exploration, and ML-driven valuation for 12,640 residential properties across Bengaluru, India (denominated in ₹ Lakhs & ₹ Crores).
2. **King County Real Estate Terminal (`app/streamlit_app.py`)**: Editorial analytical dashboard and predictive valuation engine for 21,613 residential property transactions in King County, Washington, USA (denominated in USD $).

---

## 1. Quick Start & Execution Commands

### Launch the Interactive Dashboards

#### A. Bengaluru Real Estate Analytics & Valuation Terminal (India)
```bash
streamlit run app/bengaluru_app.py
```
*Opens at `http://localhost:8501`. Features interactive Plotly histograms, scatter plots with hover metadata, BHK box plots, locality aggregators with volume sliders, and the validated Random Forest (Log) valuation engine.*

#### B. King County Real Estate Market Terminal (USA)
```bash
streamlit run app/streamlit_app.py
```
*Preserved historical US sales intelligence dashboard and valuation interface.*

---

### Run Automated Unit and Integration Tests (101 Passing Tests)
```bash
python -m pytest -v
```
*Validates data parsing, total sqft conversions, zero-leakage safeguards, pipeline serialization, Plotly figure builders, input validators, and currency formatters.*

---

### Execute End-to-End Data Preparation & Model Pipelines

```bash
# 1. Clean raw Bengaluru dataset (13,320 -> 12,640 rows)
python -m src.bengaluru_data_prep

# 2. Train, benchmark, audit, and serialize Bengaluru production model
python -m src.bengaluru_train

# 3. Clean King County dataset & train US models (Preserved baseline)
python -m src.data_prep
python -m src.train
```

---

## 2. Bengaluru Platform Architecture & Capabilities

### Interactive Dashboard Features (`app/bengaluru_app.py`)
- **Market Overview**: Real-time KPI scorecards displaying active listings, median asking price (₹ Lakhs), mean asking price, and median price per square foot (₹/sq.ft). Clear disclosures that listings represent online portal asking prices rather than registered transfer deed records.
- **Interactive Plotly Diagnostics**:
  - *Price Distribution*: Adjustable bin-width slider and linear vs. logarithmic scale toggles with median/mean reference annotations.
  - *Area vs. Asking Price*: Interactive scatter plot with full hover tooltips (locality, configuration, bathrooms, clean area, and asking price) colored by area classification.
  - *BHK Box Plot*: Price dispersion across 1 to 5+ BHK properties.
  - *Locality Ranking*: Horizontal bar chart ranking micro-markets by median asking price or volume, equipped with a minimum listing frequency filter (e.g. min 15 listings) to prevent small-sample bias.
  - *Area Type Dynamics*: Grouped bar charts comparing Super built-up, Built-up, Plot, and Carpet areas.
- **Shared Sidebar Filters**: Dynamic multi-selects and range sliders for Locality, BHK, Asking Price Range, Property Area, and Possession Status. All charts and KPIs respond instantly, handling empty queries gracefully.
- **Predictive Valuation Engine**:
  - Validates user input bounds and room-to-area heuristics.
  - Uses the production **Random Forest (Log)** pipeline to estimate fair-market asking price in ₹ Lakhs and ₹ Crores.
  - Provides implied ₹/sq.ft rate, city-wide and locality benchmark comparisons, model uncertainty intervals ($\pm\text{MedAE} = \text{₹}12.95\text{L}$, $\text{MAE} = \text{₹}32.60\text{L}$), and disclaimers.
- **Model Performance & Verification**:
  - Test scorecards (Test MAE, MedAE, RMSE, $R^2$) with plain-language definitions.
  - Interactive Predicted vs. Actual scatter plot and Residual error distribution on the locked test partition ($n = 2,528$).
  - Full candidate model benchmark comparison chart.

---

## 3. Machine Learning Methodology & Zero-Leakage Audit

### Zero-Leakage Policy
- **Excluded Features**: `price_per_sqft` is strictly excluded from all predictive models. In real estate pricing, price per square foot is a direct mathematical derivative of the target ($y / \text{sqft}$); using it as an input causes circular target leakage.
- **Training-Only Preprocessing**: All locality frequency aggregations (`OneHotEncoder(min_frequency=10)`) and scalers were fitted exclusively on training folds. Unseen localities or rare micro-markets map safely to an infrequent category representation.

### Bengaluru Model Benchmark (5-Fold CV on Training Set vs. Held-Out Test Set $n = 2,528$)

| Architecture | Target Scale | 5-Fold CV MAE (₹L) | Test MAE (₹L) | Test MAE (INR) | Test MedAE (₹L) | Test RMSE (₹L) | Test $R^2$ | Overfit Gap (MAE) |
|---|---|---|---|---|---|---|---|---|
| **Baseline (Median)** | Raw | ₹65.15L | ₹63.27L | ₹63,27,136 | ₹27.26L | ₹148.87L | -0.0805 | -₹1.85L |
| **Linear Regression** | Raw | ₹54.31L | ₹51.44L | ₹51,43,819 | ₹27.16L | ₹119.73L | 0.3011 | -₹0.96L |
| **Linear Regression** | $\log_{1p}$ | ₹78.39L | ₹48.78L | ₹48,77,882 | ₹17.27L | ₹162.59L | -0.2889 | -₹14.16L |
| **Ridge Regression** | Raw | ₹53.44L | ₹50.80L | ₹50,79,652 | ₹25.99L | ₹119.87L | 0.2995 | -₹1.18L |
| **Ridge Regression** | $\log_{1p}$ | ₹85.38L | ₹49.80L | ₹49,80,488 | ₹17.13L | ₹172.62L | -0.4527 | -₹17.88L |
| **Random Forest** | Raw | ₹34.57L | ₹34.01L | ₹34,00,520 | ₹13.41L | ₹89.84L | 0.6065 | +₹12.66L |
| **Random Forest (Selected)** | **$\log_{1p}$** | **₹32.98L** | **₹32.60L** | **₹32,59,818** | **₹12.95L** | **₹86.05L** | **0.6390** | **+₹10.71L** |
| **HistGradientBoosting** | Raw | ₹36.36L | ₹34.09L | ₹34,08,793 | ₹15.20L | ₹83.12L | 0.6632 | +₹1.11L |
| **HistGradientBoosting** | $\log_{1p}$ | ₹33.45L | ₹32.15L | ₹32,15,248 | ₹12.46L | ₹90.46L | 0.6011 | +₹1.28L |

### Key Methodological Insights:
1. **Pre-Specified Training Selection**: Model selection was determined strictly by **lowest 5-fold CV MAE on the training set**. `Random Forest (Log)` won on every training CV metric (CV MAE ₹32.98L, CV MedAE ₹12.49L, CV RMSE ₹96.43L, CV $R^2$ 0.6075).
2. **Generalization vs. Train-Test Gap**: A small train-test gap alone is not proof of superior generalization; it reflects the bias-variance profile of the algorithm. Random Forest's deep trees fit localized neighborhood nuances with lower training bias, yielding lower out-of-sample CV error.
3. **Catastrophic Failure of Linear Log Models**: Additive linear models in log space exponentiate extreme outlier features (e.g. 27 BHK properties in Electronic City), producing multi-thousand-crore predictions that ruin fold $R^2$. Tree models are structurally immune because predictions are bounded by training leaf constants.

---

## 4. Repository Structure

```
REAL_ESTATE/
├── README.md                                  # Repository documentation
├── requirements.txt                           # Project dependencies (incl. plotly)
├── data/
│   ├── raw/
│   │   ├── bengaluru_house_data.csv           # Raw Bengaluru CSV (13,320 rows)
│   │   └── real_estate_data.csv               # Raw King County CSV (21,613 rows)
│   └── processed/
│       ├── bengaluru_clean.csv                # Cleaned Bengaluru dataset (12,640 rows)
│       └── house_sales_clean.csv              # Cleaned King County dataset (21,613 rows)
├── models/
│   ├── bengaluru/
│   │   ├── price_model.joblib                 # Serialized Bengaluru pipeline (5.07 MB)
│   │   └── model_metrics.json                 # Machine-readable evaluation report
│   ├── price_model.joblib                     # Serialized King County model
│   └── model_metrics.json                     # King County metrics report
├── notebooks/
│   ├── bengaluru_01_data_cleaning.ipynb       # Bengaluru data ingestion & audit
│   ├── bengaluru_02_eda_market_insights.ipynb # Bengaluru exploratory analysis
│   ├── bengaluru_03_model_training_evaluation.ipynb # Bengaluru training & evaluation
│   ├── 01_data_cleaning.ipynb                 # King County cleaning notebook
│   ├── 02_eda_market_insights.ipynb           # King County EDA notebook
│   └── 03_modeling.ipynb                      # King County modeling notebook
├── src/
│   ├── bengaluru_dashboard_data.py            # Dashboard data loading & Plotly builders
│   ├── bengaluru_data_prep.py                 # Bengaluru parsing & cleaning engine
│   ├── bengaluru_predict.py                   # Bengaluru inference & INR formatters
│   ├── bengaluru_train.py                     # Bengaluru ML pipeline & benchmark suite
│   ├── config.py                              # Centralized configuration & filepaths
│   ├── data_prep.py                           # King County cleaning module
│   ├── predict.py                             # King County prediction module
│   └── train.py                               # King County training module
├── tests/
│   ├── test_bengaluru_dashboard.py            # 16 tests for dashboard filters & Plotly charts
│   ├── test_bengaluru_data_prep.py            # 44 tests for Bengaluru cleaning & parsing
│   ├── test_bengaluru_train.py                # 10 tests for Bengaluru ML & zero-leakage
│   ├── test_data_prep.py                      # 20 tests for King County data prep
│   ├── test_train.py                          # 6 tests for King County models
│   └── test_app.py                            # 5 tests for King County app validation
├── reports/figures/
│   └── bengaluru/                             # 10 diagnostic figures (PNG)
└── app/
    ├── bengaluru_app.py                       # Bengaluru Plotly interactive dashboard
    └── streamlit_app.py                       # King County Streamlit dashboard
```

---

## 5. Dataset Attribution & Scope Limitations

- **Bengaluru House Price Dataset**: Sourced from Amitabha Joy on Kaggle. Represents online real estate portal asking prices in Bengaluru, Karnataka, India.
- **King County Dataset**: Historical residential home sales from 2014–2015 in King County, Washington, USA.
- **Portal Asking Prices vs. Registered Valuations**: Predictions reflect seller-quoted online listing portal asking prices, which typically embed unstandardized negotiation margins (5%–15%), GST, parking charges, and amenity premiums not captured in official sub-registrar registry deeds (guidance values).
- **Untracked Physical Factors**: The datasets do not record floor rise (e.g. 25th floor vs ground floor), private terrace amenities, interior woodwork quality, or builder brand tier (e.g. Sobha vs. unbranded local builders).
- **Regulatory Status**: Bengaluru property prices are sensitive to municipal title documentation (e.g., BBMP 'A' Khata vs. 'B' Khata vs. BDA layout vs. Gram Panchayat). These regulatory statuses are not captured in portal listings and must be independently verified during physical appraisal.

---

## Author

**Aditya Pratap Singh**  
GitHub: [@Aditya-dev03](https://github.com/Aditya-dev03)
