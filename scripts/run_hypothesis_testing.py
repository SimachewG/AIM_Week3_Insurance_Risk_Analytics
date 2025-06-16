import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.hypothesis_testing import hypothesis_test_pipeline

# Ensure directory for reports
os.makedirs("reports", exist_ok=True)

# Load cleaned data
data_path = os.path.join("data", "processed", "cleaned_data.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing file: {data_path}")

data = pd.read_csv(data_path)

# Define hypotheses to test
hypotheses = [
    ('Gender', 'Claim difference between genders'),
    ('LegalType', 'Claim difference by legal type'),
    ('AccountType', 'Claim difference by account type'),
    ('MaritalStatus', 'Claim difference by marital status'),
    ('Province', 'Claim difference across provinces')
]

# Drop rows with missing values in relevant columns
results = []
for group_col, hypothesis_name in hypotheses:
    if group_col not in data.columns:
        print(f"Warning: Column '{group_col}' not in dataset. Skipping...")
        continue

    filtered = data.dropna(subset=['TotalClaims', 'TotalPremium', group_col])
    print(f"Running test: {hypothesis_name}")
    test_result = hypothesis_test_pipeline(filtered, group_col, hypothesis_name)
    results.append(test_result)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("reports/hypothesis_test_results.csv", index=False)

print("\n✅ Hypothesis testing completed and saved to reports/hypothesis_test_results.csv")
