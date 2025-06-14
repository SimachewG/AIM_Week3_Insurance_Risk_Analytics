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
