1: Project Overview – Insurance Risk Analytics**

* Goal: Analyze South African car insurance data
* Dataset: 1,000,098 records, 52 columns
* Focus: Risk profiling, customer segmentation, claim trends
* Tools: Python, Pandas, Seaborn, Matplotlib, Git, GitHub

2: Data Preprocessing – Overview**

* Ensures data quality for modeling
* Key tasks: Handle missing data, encode categories, format columns
* Implemented in: `src/data_preprocessing.py`

3: Handling Missing Values**

* Checked missing counts and percentages
* Applied forward-fill method where applicable
* Categorical placeholders (e.g., "Unknown") used for sparse features
* Ensured domain-reasonable imputations

4: Final Cleaned Dataset**

* Cleaned data saved for consistent usage
* Used in notebooks and Python scripts
* No critical missing values or malformed data remaining

5: Exploratory Data Analysis (EDA) – Overview**

* Objective: Discover patterns and guide feature selection
* Implemented in: `src/eda.py`
* Visuals saved in: `src/visualization/`

6: Distribution Analysis**

* Focused on key numerical features:

  * `TotalClaimAmount`, `TotalPremium`, `SumInsured`
* Used histograms and boxplots
* Detected heavy skewness and outliers

7: Loss Ratio Exploration**

* Defined Loss Ratio = Total Claims / Total Premiums
* Grouped and compared across:

  * Gender
  * Province
  * Vehicle Type
* Identified segments with high loss ratio

8: Temporal Trends**

* Grouped data by transaction month
* Analyzed time-based trends in claims and premiums
* Found seasonal claim spikes and premium fluctuations

9: Correlation Analysis**

* Generated a correlation heatmap
* Focused on numerical features
* Identified potential multicollinearity
* Suggested which features to combine or drop later

10: High-Risk Categories & Segments**

* Top `VehicleMakeModel` types by claim amount
* Top `MainCrestaZone` locations by average loss ratio
* Bar charts used for ranking and segmentation


11: Visual Outputs (Saved Figures)**

* `Distribution_TotalClaims.png`
* `LossRatio_by_Gender.png`
* `LossRatio_by_Province.png`
* `Correlation_Heatmap.png`
* `Temporal_Trend.png`
* All plots saved to `visualization/`

12: Hypothesis Testing & Exploratory Analysis

- **Objective**: Identify key business insights through statistical testing
- **Key Activities**:
  - Performed A/B tests and Chi-Square tests using `src/hypothesis_analysis.py`
  - Analyzed relationships between variables such as claim likelihood and demographics
  - Example:
    - Compared premium values between vehicle types
    - Tested dependency of claim status on marital status

13: Data Preprocessing and Feature Engineering

- **Objective**: Clean and transform data for modeling
- **Steps Taken**:
  - Handled missing values, encoded categorical variables
  - Engineered new features like `PremiumToClaimsRatio`
  - Split data into training and testing sets

14: Predictive Modeling and Explainability

- **Objective**: Predict:
  1. **Claim Severity** (Regression)
  2. **Claim Occurrence** (Classification)
  3. **Risk-Based Premium** (Formula: `Probability × Severity + Expense Loading`)
- **Modeling Techniques**:
  - **Regression**: Linear Regression, Decision Tree, Random Forest, XGBoost
  - **Classification**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Explainability**:
  - SHAP summary plots and top features
- **Modules Used**:
  - `src/Machine_learning_model.py`
- **Key Outputs**:
  - `outputs/risk_based_premium.csv`: Final premiums
  - `outputs/shap_summary.png`: SHAP explainability
  - `outputs/model_comparison.csv`: Model performance
  - Feature importances from XGBoost

## 14: Final Model Results

### Claim Severity (Regression)

| Model            | RMSE      | R²    |
|------------------|-----------|-------|
| Linear Regression | 33,346.75 | 0.309 |
| Decision Tree     | 13,204.97 | 0.892 |
| Random Forest     | 8,974.59  | 0.950 |
| XGBoost           | 7,383.77  | 0.966 |

### Premium Prediction (Regression)

| Model            | RMSE     | R²    |
|------------------|----------|-------|
| Linear Regression | 261.27  | 0.487 |
| Decision Tree     | 9.90    | 0.999 |
| Random Forest     | 7.81    | 1.000 |
| XGBoost           | 28.88   | 0.994 |

### Claim Classification

| Model            | Accuracy | Precision | Recall | F1-score |
|------------------|----------|-----------|--------|----------|
| Logistic Regression | 1.000 | 1.000     | 1.000  | 1.000    |
| Decision Tree       | 1.000 | 1.000     | 1.000  | 1.000    |
| Random Forest       | 1.000 | 1.000     | 1.000  | 1.000    |
| XGBoost             | 1.000 | 1.000     | 0.942  | 0.970    |


