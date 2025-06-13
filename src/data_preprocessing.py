import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load historical claim data from a pipe-delimited text file."""
    try:
        data = pd.read_csv(filepath, sep='|', low_memory=False)
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        print(f"Current working directory: {os.getcwd()}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the file: {filepath}. Details: {e}")
        return None


# Example usage
file_path = '../data/raw/MachineLearningRating_v3.txt'
data = load_data(file_path)

if data is not None:
    print("\nFirst few rows of the data:")
    print(data.head())
    print("\nColumn names:")
    print(data.columns.tolist())

def preprocess_data(data):
    """Comprehensive preprocessing: handle missing values, data types, duplicates, and feature engineering."""
    
    print("Initial data shape:", data.shape)
    
    # 1. Overview of missing data
    missing = data.isnull().sum()
    missing_percent = (missing / len(data)) * 100
    print("\nMissing values (%):\n", missing_percent[missing_percent > 0].sort_values(ascending=False))

    # 2. Drop columns with more than 60% missing values
    threshold = 60
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    if cols_to_drop:
        print(f"\nDropping columns with >{threshold}% missing values: {cols_to_drop}")
        data.drop(columns=cols_to_drop, inplace=True)

    # 3. Fill remaining missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna(data[col].mode()[0], inplace=True)  # Fill with mode
        else:
            data[col].fillna(data[col].median(), inplace=True)   # Fill with median for numeric columns

    # 4. Convert date columns
    if 'TransactionMonth' in data.columns:
        try:
            data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], errors='coerce')
        except Exception as e:
            print(f"Error converting 'TransactionMonth' to datetime: {e}")
    else:
        print("Warning: 'TransactionMonth' column does not exist.")

    # 5. Extract date features
    if 'TransactionMonth' in data.columns and pd.api.types.is_datetime64_any_dtype(data['TransactionMonth']):
        data['TransactionYear'] = data['TransactionMonth'].dt.year
        data['TransactionQuarter'] = data['TransactionMonth'].dt.quarter
        data['TransactionMonthNum'] = data['TransactionMonth'].dt.month

    # 6. Drop duplicate rows
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"\nRemoving {duplicates} duplicate rows.")
        data.drop_duplicates(inplace=True)

    # 7. Encode categorical variables (Label Encoding for simplicity)
    #categorical_cols = data.select_dtypes(include='object').columns.tolist()
    #for col in categorical_cols:
    #    le = LabelEncoder()
    #    try:
    #        data[col] = le.fit_transform(data[col])
    #    except Exception as e:
    #        print(f"Error encoding column '{col}': {e}")

    # 8. Placeholder for outlier handling (you can implement based on domain)
    # Example: using IQR or Z-score method to remove/cap outliers

    # 9. Placeholder for feature scaling (if needed for modeling)

    print("\nFinal data shape after preprocessing:", data.shape)

    return data
