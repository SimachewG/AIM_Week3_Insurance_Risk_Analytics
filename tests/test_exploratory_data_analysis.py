import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
from src.exploratory_data_analysis import compute_loss_ratio, temporal_trend

class TestEDAFunctions(unittest.TestCase):
    def setUp(self):
        # Test DataFrame with necessary columns and no zero premiums
        self.df = pd.DataFrame({
            'PolicyID': [1, 2, 3, 4],
            'TotalClaims': [100, 200, 0, 50],
            'TotalPremium': [1000, 500, 500, 2000],  # no zero premiums
            'TransactionMonth': ['2021-01', '2021-02', '2021-03', '2021-04'],  # strings, not datetime
            'make': ['A', 'B', 'A', 'C'],
            'Model': ['X', 'Y', 'Z', 'X']
        })

    def test_compute_loss_ratio(self):
        # Calculate loss ratio using the function
        ratio = compute_loss_ratio(self.df)
        # Calculate expected ratio manually
        expected_ratio = (100/1000 + 200/500 + 0/500 + 50/2000) / 4
        self.assertAlmostEqual(ratio, expected_ratio, places=4)

    def test_temporal_trend_runs(self):
        # Just check that temporal_trend runs without error
        try:
            temporal_trend(self.df)
        except Exception as e:
            self.fail(f"temporal_trend raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()