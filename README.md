# 🧼 Insurance Dataset Preprocessing Script

This module provides functions to **load** and **preprocess** raw insurance claim data from a pipe-delimited text file. It is designed to clean and prepare the dataset for further analysis, such as loss ratio evaluation, modeling, or visualization.

## 📁 Project Structure

```
project/
├── data/
│   └── raw/
│       └── MachineLearningRating_v3.txt      # Raw insurance dataset
│
├── src/
│   └── data_preprocessing.py                 # This script (loading + preprocessing functions)
│
└── README.md

## 📌 Overview

The script contains two main functions:

### 1. `load_data(filepath)`

Loads the insurance dataset from a `|` (pipe) delimited `.txt` file.

* Checks for missing or incorrect files.
* Handles parse errors gracefully.
* Returns a Pandas DataFrame if successful.

### 2. `preprocess_data(data)`

Performs comprehensive data cleaning and preparation.

Steps include:

* Missing values summary and treatment:

  * Drops columns with >60% missing values
  * Fills missing values with **mode** (categorical) or **median** (numerical)
* Date parsing and feature extraction from `TransactionMonth`
* Duplicate row removal

## 📌 Notes

* The column `TransactionMonth` is converted to datetime and used to extract:

  * `TransactionYear`
  * `TransactionQuarter`
  * `TransactionMonthNum`
* Encoding is **optional** and can be customized based on downstream ML models.
* Placeholder blocks for:

  * Outlier detection (using IQR)

## 📁 Project Structure

project/
│
├── data/
│   └── raw/
│       └── MachineLearningRating_v3.txt   # Raw insurance dataset
│
├── scripts/
│   └── run_analysis.py                    # Main script (code above)
│
├── src/
│   └── data_preprocessing.py             # Contains `load_data()` and `preprocess_data()` functions
│
└── README.md                             # This file
```

## 🚀 Features

* Calculates **loss ratio** as `TotalClaims / TotalPremium`
* Plots **loss ratio distribution** across `Province`, `Gender`, and `VehicleType`
* Performs **univariate distribution analysis** for numeric features
* Detects **outliers** using boxplots for key numeric columns
* Analyzes **temporal trends** in claims, premiums, and loss ratios
* Identifies **vehicle make/models** with the highest and lowest claim amounts
* Visualizes **correlations** between numeric variables

## 👨‍💻 Author

Developed by Simachew Gashaw Yaecob: who works on risk analytics and predictive modeling.