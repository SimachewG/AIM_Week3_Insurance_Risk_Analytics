import unittest
import pandas as pd
import os
import tempfile
from src.data_preprocessing import load_data, preprocess_data

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
        300|3000|2021-01-01|A|X  <-- duplicate of earlier row
        """

        self.test_file.write(self.test_data)
        self.test_file.close()

    def tearDown(self):
        # Remove the temporary file after tests
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_load_data(self):
        data = load_data(self.test_file_path)
        self.assertIsNotNone(data)
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 5)

    def test_load_data_file_not_found(self):
        data = load_data("non_existent_file.txt")
        self.assertIsNone(data)

    def test_preprocess_data(self):
        data = load_data(self.test_file_path)
        self.assertIsNotNone(data)

        processed_data = preprocess_data(data)

        self.assertEqual(processed_data.shape[0], 5)  # Duplicate row removed
        self.assertIn('TransactionYear', processed_data.columns)
        self.assertIn('TransactionQuarter', processed_data.columns)
        self.assertIn('TransactionMonthNum', processed_data.columns)
        self.assertEqual(processed_data['TotalClaims'].isnull().sum(), 0)
        self.assertEqual(processed_data['TotalPremium'].isnull().sum(), 0)

    def test_preprocess_data_with_high_missing(self):
        data = pd.DataFrame({
            'TotalClaims': [100, 200, None, None],
            'TotalPremium': [1000, None, None, None],
            'TransactionMonth': ['2021-01-01', '2021-02-01', None, None],
            'make': ['A', 'B', None, None],
            'Model': ['X', 'Y', None, None]
        })

        processed_data = preprocess_data(data)
        self.assertLessEqual(len(processed_data.columns), 7)


if __name__ == '__main__':
    unittest.main()

