import pandas as pd
import os

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

def preprocess_data(data):
    """Comprehensive preprocessing: handle missing values, data types, duplicates, and feature engineering."""
    
    print("Initial data shape:", data.shape)
    
    # Missing value inspection
    missing = data.isnull().sum()
    missing_percent = (missing / len(data)) * 100
    print("\nMissing values (%):\n", missing_percent[missing_percent > 0].sort_values(ascending=False))

    # Drop high-missing columns
    threshold = 60
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    if cols_to_drop:
        print(f"\nDropping columns with >{threshold}% missing values: {cols_to_drop}")
        data.drop(columns=cols_to_drop, inplace=True)

    # Fill remaining missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

    # Convert date column
    if 'TransactionMonth' in data.columns:
        try:
            data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], errors='coerce')
        except Exception as e:
            print(f"Error converting 'TransactionMonth' to datetime: {e}")
    else:
        print("Warning: 'TransactionMonth' column does not exist.")

    # Date features
    if 'TransactionMonth' in data.columns and pd.api.types.is_datetime64_any_dtype(data['TransactionMonth']):
        data['TransactionYear'] = data['TransactionMonth'].dt.year
        data['TransactionQuarter'] = data['TransactionMonth'].dt.quarter
        data['TransactionMonthNum'] = data['TransactionMonth'].dt.month

    # Drop duplicates
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"\nRemoving {duplicates} duplicate rows.")
        data.drop_duplicates(inplace=True)

    print("\nFinal data shape after preprocessing:", data.shape)
    return data

if __name__ == "__main__":
    file_path = '../data/raw/MachineLearningRating_v3.txt'
    data = load_data(file_path)

    if data is not None:
        print("\nFirst few rows of the data:")
        print(data.head())
        print("\nColumn names:")
        print(data.columns.tolist())

        data = preprocess_data(data)
