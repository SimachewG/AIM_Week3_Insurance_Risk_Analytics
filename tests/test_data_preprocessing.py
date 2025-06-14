import unittest
import pandas as pd
import tempfile
from src.data_preprocessing import load_data, preprocess_data
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDataFunctions(unittest.TestCase):

    def setUp(self):
        # Create a temporary test file
        self.test_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        self.test_file_path = self.test_file.name

        # Write mock data into the file
        self.test_data = """TotalClaims|TotalPremium|TransactionMonth|make|Model
100|1000|2021-01-01|A|X
200|2000|2021-02-01|B|Y
300|3000|2021-01-01|A|X
400|4000|2021-03-01|B|Y
NaN|5000|2021-04-01|C|Z
600|NaN|2021-01-01|A|X
300|3000|2021-01-01|A|X
"""

        self.test_file.write(self.test_data)
        self.test_file.close()

    def tearDown(self):
        # Remove temporary test file
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_load_data_success(self):
        data = load_data(self.test_file_path)
        self.assertIsNotNone(data)
        self.assertEqual(data.shape[1], 5)

    def test_load_data_file_not_found(self):
        data = load_data("non_existent_file.txt")
        self.assertIsNone(data)

    def test_preprocess_data_cleaning(self):
        data = load_data(self.test_file_path)
        self.assertIsNotNone(data)

        processed_data = preprocess_data(data)

        # Check that NaNs are filled
        self.assertEqual(processed_data['TotalClaims'].isnull().sum(), 0)
        self.assertEqual(processed_data['TotalPremium'].isnull().sum(), 0)

        # Expecting 1 duplicate row removed → from 7 to 6 rows
        self.assertEqual(processed_data.shape[0], 6)


        # Check date feature engineering
        self.assertIn('TransactionYear', processed_data.columns)
        self.assertIn('TransactionQuarter', processed_data.columns)
        self.assertIn('TransactionMonthNum', processed_data.columns)
    def test_preprocess_high_missing_columns_dropped(self):
        data = pd.DataFrame({
            'TotalClaims': [100, 200, None, None],
            'MostlyMissing': [None, None, None, None],
            'TransactionMonth': ['2021-01-01', '2021-02-01', None, None],
            'make': ['A', 'B', None, None],
            'Model': ['X', 'Y', None, None]
        })

        processed_data = preprocess_data(data)
        self.assertNotIn('MostlyMissing', processed_data.columns)


if __name__ == '__main__':
    unittest.main()


