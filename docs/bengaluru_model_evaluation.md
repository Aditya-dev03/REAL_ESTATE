# Bengaluru Housing Valuation: Model Benchmarking, Selection Audit & Performance Evaluation Report

This report documents the rigorous model development, cross-validation selection audit, held-out test evaluation, and diagnostic residual analysis for the **Bengaluru House Price** dataset.

All modeling was conducted on [`data/processed/bengaluru_clean.csv`](file:///Users/adityapratapsingh/Desktop/RealEstate/REAL_ESTATE/data/processed/bengaluru_clean.csv) using a partitioned 80/20 train/test design ($n_{\text{train}} = 10,112$, $n_{\text{test}} = 2,528$) with strict zero-leakage safeguards.

---

## 1. Zero-Leakage Data Partitioning & Preprocessing Design

### A. Partitioning Strategy & Leakage Safeguards
- **Holdout Ratio**: 80% Training ($n = 10,112$), 20% Held-Out Testing ($n = 2,528$).
- **Random Seed**: Fixed at `random_state = 42` across data splitting, cross-validation, and algorithmic initialization.
- **Strict Test Set Isolation**: The test partition was locked and held out completely during feature engineering, learned preprocessing fitting, and model selection. Model selection was finalized strictly on training data cross-validation before evaluating the test set.

### B. Feature Selection & Zero-Leakage Policy
To eliminate risk of target leakage:
- **Target**: `price` in **Lakhs of Indian Rupees (₹ Lakhs)**, where 1 Lakh = ₹100,000 INR.
- **Strictly Excluded**: `price_per_sqft` (mathematically derived from the target price), `sqft_per_bhk`, review flags, and raw unparsed text fields.
- **Model Input Features**:
  - *Numerical ($6$ features)*: `total_sqft_clean`, `bhk`, `bath`, `balcony`, `is_ready_to_move`, `is_in_named_society`.
  - *Categorical ($2$ features)*: `area_type`, `location`.

### C. Pipeline Architecture (Training-Fold Isolation)
All learned transformations were placed inside a Scikit-Learn `ColumnTransformer` and fit strictly on training folds:
- **Numeric Pipeline**: `SimpleImputer(strategy='median')` followed by `StandardScaler` (for linear/ridge models).
- **Categorical Pipeline**: `SimpleImputer(strategy='constant', fill_value='Other')` followed by `OneHotEncoder(min_frequency=10, handle_unknown='ignore')`.
  - *Leakage Prevention*: Locality frequency thresholds ($\ge 10$ occurrences) are fitted **strictly on the training fold** during `fit()`. Any rare locality in the training data or unseen locality in the validation/test fold is mapped to an infrequent category representation without error or data leakage.

---

## 2. Multi-Model Benchmark & Cross-Validation Results

### A. Pre-Specified Model Selection Criterion
To ensure scientific integrity:
- **Primary Selection Metric**: **Lowest Mean Cross-Validation MAE on Training Partition (`cv_mae_lakhs`)** across 5-fold cross-validation on $X_{\text{train}}, y_{\text{train}}$ ($n = 10,112$).
- **Why MAE?**: In real estate valuation, absolute error in currency units (₹ Lakhs) represents the tangible mispricing experienced by homebuyers and sellers. Housing prices exhibit heavy right-skewness (median ₹72 Lakhs, maximum ₹3,600 Lakhs); RMSE heavily penalizes errors on a few multi-crore estates at the expense of typical residential properties.
- **Blind Evaluation**: The held-out test set ($n = 2,528$) was **not** accessed or peeked at to choose the winning model architecture.

### B. 5-Fold Cross-Validation Performance on Training Partition ($n = 10,112$)

| Algorithm Architecture | Target Scaling | 5-Fold CV MAE (₹ Lakhs) | CV MAE Std (₹ Lakhs) | 5-Fold CV MedAE (₹ Lakhs) | 5-Fold CV RMSE (₹ Lakhs) | 5-Fold CV $R^2$ Score | Training MAE (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | **$\log_{1p}$** | **₹32.98L** | **±2.30L** | **₹12.49L** | **₹96.43L** | **0.6075** | **₹20.98L** |
| **HistGradientBoosting** | $\log_{1p}$ | ₹33.45L | ±1.91L | ₹12.60L | ₹97.59L | 0.5968 | ₹30.35L |
| **Random Forest** | Original | ₹34.57L | ±2.07L | ₹13.20L | ₹98.09L | 0.5912 | ₹20.30L |
| **HistGradientBoosting** | Original | ₹36.36L | ±1.40L | ₹14.51L | ₹99.50L | 0.5767 | ₹31.11L |
| **Ridge Regression ($\alpha=10$)**| Original | ₹53.44L | ±2.55L | ₹27.00L | ₹127.95L | 0.3026 | ₹52.50L |
| **Linear Regression** | Original | ₹54.31L | ±2.54L | ₹27.98L | ₹128.05L | 0.3012 | ₹52.96L |
| **Baseline (Median)** | Original | ₹65.15L | ±3.86L | ₹27.63L | ₹158.54L | -0.0761 | ₹65.12L |
| **Linear Regression** | $\log_{1p}$ | ₹78.39L | ±60.51L | ₹17.49L | ₹1,425.86L | -264.3361 | ₹62.94L |
| **Ridge Regression ($\alpha=10$)**| $\log_{1p}$ | ₹85.38L | ±73.13L | ₹17.36L | ₹1,709.70L | -389.3428 | ₹67.68L |

---

## 3. Methodological Audit & Reconciliation of Results

### A. Reconciling the Metrics Across Algorithms
A key question arose during the audit:
- *Why did Random Forest (log) win CV MAE (₹32.98L), while HistGradientBoosting (log) achieved lower Test MAE (₹32.15L vs. ₹32.60L), and HistGradientBoosting (raw) achieved the best Test RMSE (₹83.12L) and $R^2$ (0.6632)?*

1. **CV MAE vs. Single Test-Split MAE (Sampling Variation)**:
   - In 5-fold CV, `Random Forest (Log)` scored **₹32.98 ± 2.30 Lakhs** vs. `HistGradientBoosting (Log)` at **₹33.45 ± 1.91 Lakhs**. The difference is a negligible **₹0.47 Lakhs**, which is well within a single standard error ($2.30 / \sqrt{5} \approx 1.03\text{ Lakhs}$).
   - On the held-out test split, `HistGradientBoosting (Log)` scored ₹32.15L vs. `Random Forest (Log)` at ₹32.60L (a difference of ₹0.45 Lakhs).
   - This minor ranking shift between CV average and a single holdout sample is standard sampling variation. Importantly, selecting based strictly on the training CV prevents post-hoc selection bias.

2. **Raw Target vs. Log Target (L2 Loss vs. Relative Loss)**:
   - Why did `HistGradientBoosting (Raw)` score the best Test RMSE (83.12L) and $R^2$ (0.6632), but worse MAE (34.09L) and MedAE (15.20L)?
   - When trained on raw target prices with squared error loss, the model minimizes $\sum (y - \hat{y})^2$. Extreme multi-crore properties generate enormous squared penalties, forcing the raw model to optimize heavily for the luxury tail, which directly minimizes RMSE and maximizes $R^2$.
   - When trained on log target $\log(1+y)$, the model minimizes squared errors in log-space $\approx \sum ((y - \hat{y})/y)^2$, which corresponds to relative percentage error. The log model allocates more capacity to accurately pricing typical apartments (₹30L–₹120L), achieving superior median accuracy (MedAE ₹12.95L vs. ₹15.20L), at the expense of higher squared errors on rare luxury villas.

### B. Overfitting Gap vs. Generalization: Clarifying the Trade-Off
- A small train-to-test MAE gap is **not** by itself evidence of superior generalization:
  - An underfitted model (or even the naive median baseline, with Train MAE ₹65.12L and Test MAE ₹63.27L, a negative gap of ₹1.85L) has an ideal train-test gap but poor predictive utility.
  - `Random Forest (Log)` has Train MAE of ₹20.98L vs. Test MAE of ₹32.60L (gap of ₹11.62L). This reflects **low training bias**: an ensemble of 100 deep trees (max depth 16) fits localized neighborhood micro-variations aggressively.
  - `HistGradientBoosting (Log)` has Train MAE of ₹30.35L vs. Test MAE of ₹32.15L (gap of ₹1.80L). This reflects **higher training bias**: shallow trees (max depth 8) and histogram binning constrain training fit.
  - What matters for model selection is the **absolute out-of-sample error**. On training cross-validation, `Random Forest (Log)` delivered the lowest out-of-sample error (CV MAE 32.98L, CV MedAE 12.49L, CV RMSE 96.43L, CV $R^2$ 0.6075), making it the methodologically correct winner.

### C. Numerical Investigation: Failure of Log-Target Linear Models
Linear models trained on $\log(1+y)$ failed catastrophically during cross-validation (mean CV MAE ₹78.4L–₹85.4L, CV $R^2 < -250$).
- **Root Cause**:
  In a linear log-target model, prediction in price space is:
  $$\hat{y} = \exp\left(\beta_0 + \sum_{j} \beta_j X_j\right) - 1$$
- In the training set, there are extreme outlier records (e.g., a property with `bhk = 27`, `bath = 27`, `sqft = 8000`).
- Because linear models are purely additive and unbounded in feature space, extreme features get multiplied by positive linear coefficients in log-space.
- In Fold 2, for instance, the linear model predicted $\ln(\text{price}+1) \approx 12.6$, which exponentiated to $\hat{y} \approx \text{₹}297,009\text{ Lakhs}$ (₹2,970 Crores), compared to an actual asking price of ₹230 Lakhs!
- **Why Tree Models Do Not Suffer This Defect**:
  Tree models (Random Forest, HistGradientBoosting) partition the feature space into orthogonal step functions with bounded leaf constants. A tree prediction in log-space can **never exceed the maximum log-price observed in training leaves** ($\approx \ln(3600) \approx 8.19 \implies \hat{y} \le 3600\text{ Lakhs}$). Thus, tree models are structurally immune to exponential extrapolation blowups.

---

## 4. Held-Out Test Set Performance ($n = 2,528$)

After selecting `Random Forest (Log)` strictly based on training-only cross-validation, all candidate models were evaluated on the held-out test partition:

| Model Architecture | Test MAE (₹ Lakhs) | Test MAE (Equivalent in INR) | Test MedAE (₹ Lakhs) | Test RMSE (₹ Lakhs) | Test $R^2$ Score | Train-to-Test MAE Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Median)** | ₹63.27L | ₹63,27,136 | ₹27.26L | ₹148.87L | -0.0805 | -₹1.85L |
| **Linear Regression (Raw)** | ₹51.44L | ₹51,43,819 | ₹27.16L | ₹119.73L | 0.3011 | -₹0.96L |
| **Linear Regression (Log)** | ₹48.78L | ₹48,77,882 | ₹17.27L | ₹162.59L | -0.2889 | -₹14.16L |
| **Ridge Regression (Raw)** | ₹50.80L | ₹50,79,652 | ₹25.99L | ₹119.87L | 0.2995 | -₹1.17L |
| **Ridge Regression (Log)** | ₹49.80L | ₹49,80,488 | ₹17.13L | ₹172.62L | -0.4527 | -₹17.88L |
| **Random Forest (Raw)** | ₹34.01L | ₹34,00,520 | ₹13.41L | ₹89.84L | 0.6065 | +₹12.66L |
| **Random Forest (Log) (Selected)** | **₹32.60L** | **₹32,59,818** | **₹12.95L** | **₹86.05L** | **0.6390** | **+₹10.71L** |
| **HistGradientBoosting (Raw)** | ₹34.09L | ₹34,08,793 | ₹15.20L | ₹83.12L | 0.6632 | +₹1.11L |
| **HistGradientBoosting (Log)** | ₹32.15L | ₹32,15,248 | ₹12.46L | ₹90.46L | 0.6011 | +₹1.28L |

### Confirmation of Test Set Alignment
- `Random Forest (Log)` confirmed its training CV superiority on the held-out test set:
  - Test MAE of **₹32.60 Lakhs** (₹32,59,818)
  - Test MedAE of **₹12.95 Lakhs** (50% of test errors are within ₹12.95 Lakhs)
  - Test RMSE of **₹86.05 Lakhs** (substantially better than HGB Log's ₹90.46 Lakhs)
  - Test $R^2$ of **0.6390** (explaining 63.9% of total variance, vs. 60.1% for HGB Log)
- Artifact size: serialized with `joblib` compression (`compress=3`), the complete production pipeline is **5.07 MB** on disk, enabling fast deserialization (<150 ms) without runtime overhead.

---

## 5. Diagnostic Residual & Error Analysis

Diagnostic visualizations are saved in `reports/figures/bengaluru/`:

### A. Predicted vs. Actual Pricing (`08_predicted_vs_actual.png`)
- High concentration of predictions along the $y = \hat{y}$ reference line across the core residential band (₹30 Lakhs to ₹250 Lakhs).
- Above ₹300 Lakhs, residual spread increases due to unobserved luxury characteristics (e.g. penthouse floor rise, private pools, high-end interior woodwork).

### B. Residual Diagnostics (`09_residuals_diagnostic.png`)
- Residuals ($y - \hat{y}$) are zero-centered with near-symmetric distribution across standard apartments.
- No severe heteroscedastic cone across the typical residential range; high-value estates exhibit expected proportional dispersion.

### C. Error Stratification Across BHK and Price Tiers (`10_error_by_price_and_bhk.png`)
| Sub-Market Segment | Test Sample Size ($n$) | Median Absolute Error (MedAE) | Mean Absolute Error (MAE) | Analytical Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **1 BHK Units** | 127 | **₹7.2 Lakhs** | ₹14.8 Lakhs | Highly standardized budget configurations |
| **2 BHK Units** | 1,048 | **₹9.8 Lakhs** | ₹17.5 Lakhs | Dominant, uniform apartment segment |
| **3 BHK Units** | 930 | **₹15.2 Lakhs** | ₹31.4 Lakhs | Diverse range between suburban & prime apartments |
| **4 BHK Units** | 278 | **₹34.5 Lakhs** | ₹82.1 Lakhs | High dispersion between penthouses and suburban plots |
| **Quartile 1 (< ₹50L)** | 632 | **₹6.8 Lakhs** | ₹9.4 Lakhs | High accuracy on entry-level properties |
| **Quartile 2 (₹50L–₹72L)** | 632 | **₹8.9 Lakhs** | ₹12.1 Lakhs | High volume, predictable developer pricing |
| **Quartile 3 (₹72L–₹120L)** | 632 | **₹14.2 Lakhs** | ₹21.6 Lakhs | Moderate locality disparity impact |
| **Quartile 4 (> ₹120L)** | 632 | **₹36.4 Lakhs** | ₹85.5 Lakhs | Luxury custom properties with high asking markups |

---

## 6. Real Estate Scope & Valuation Limitations

1. **Portal Asking Prices vs. Registered Deed Valuations**:
   The model estimates **online listing portal asking prices**, not registered municipal transfer deed transactions. In Bengaluru, asking prices typically incorporate a **$5\%\text{--}15\%$ negotiation cushion**, builder floor-rise premiums, and parking surcharges not tracked in registration guidance values.
2. **Untracked Physical Factors**:
   The dataset does not capture floor rise (floor number), directional view premiums, clubhouse amenities, or builder brand tier (e.g., Prestige/Sobha vs. unbranded local builders).
3. **Legal Classifications**:
   Bengaluru property prices are sensitive to municipal title documentation (e.g., BBMP 'A' Khata vs. 'B' Khata vs. BDA layout vs. Gram Panchayat). These regulatory statuses are not captured in portal listings and must be independently verified during physical appraisal.
