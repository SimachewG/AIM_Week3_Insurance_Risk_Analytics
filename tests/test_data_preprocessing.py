import unittest
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.data = load_data('../data/raw/MachineLearningRating_v3.txt')

    def test_preprocess_data(self):
        cleaned_data = preprocess_data(self.data)
        self.assertFalse(cleaned_data.isnull().values.any())  # Check for missing values
        self.assertEqual(cleaned_data['TransactionMonth'].dtype, 'datetime64[ns]')  # Check date type

if __name__ == '__main__':
    unittest.main()